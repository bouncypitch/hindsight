"""Mine candidate failures from real FrodoBots rides using the robot's own sensors, then cut clips.

Signals (relative to each episode's own baseline, since IMU mounting differs per robot):
  stall  - operator commands forward motion but the wheels stop (blocked, stuck on a curb, high-centered)
  tilt   - gravity direction moves > TILT_DEG away from the episode's normal and stays there (tipped / slope)
  shock  - gyro or accelerometer spikes (curb drops, bumps, collisions)
  brake  - driver slams from >=70% to ~0 command in 0.5 s at speed (reacting to a near miss / wrong move)
  swerve - driver yanks the steering in 0.5 s while moving (avoiding something the robot drove toward)
  jerky  - sustained stop-go throttle or twitchy steering over 3 s (unsure, oscillating, unsafe-looking driving)

uv run python -m data.mine_failures  ->  data/real_clips/*.mp4 + *.imu.csv + candidates.json
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "raw" / "frodobots"
OUT = Path(__file__).parent / "real_clips"
FPS = 10
TILT_DEG, STALL_S, SHOCK_GYRO = 35.0, 2.0, 90.0
PRE, POST = 6.0, 4.0  # clip window around the event, seconds


def load(file_idx: int = 0):
    names = json.load(open(RAW / "meta/info.json"))["features"]["observation.state"]["names"]
    eps = pd.concat([pd.read_parquet(p) for p in sorted((RAW / "meta/episodes/chunk-000").glob("*.parquet"))])
    vcol = "videos/observation.images.front"
    eps = eps[(eps[f"{vcol}/file_index"] == file_idx) & (eps[f"{vcol}/chunk_index"] == 0)]
    data_files = sorted(set(eps["data/file_index"]))
    eps = eps[["episode_index", "length", f"{vcol}/from_timestamp", f"{vcol}/to_timestamp"]]
    eps.columns = ["episode", "length", "v_from", "v_to"]
    parts = [pd.read_parquet(RAW / f"data/chunk-000/file-{i:03d}.parquet") for i in data_files
             if (RAW / f"data/chunk-000/file-{i:03d}.parquet").exists()]
    d = pd.concat(parts)
    d = d[d.episode_index.isin(eps.episode)].reset_index(drop=True)
    S = pd.DataFrame(np.stack(d["observation.state"].values), columns=names, index=d.index)
    A = np.stack(d["action"].values)
    S["cmd_lin"], S["cmd_ang"] = A[:, 0], A[:, 1]
    S["episode"], S["t"] = d.episode_index.values, d.frame_index.values / FPS
    return S, eps.set_index("episode")


def runs(mask: np.ndarray, min_len: int):
    """(start, end) index pairs where mask is True for at least min_len samples."""
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def events(ep: pd.DataFrame) -> list[dict]:
    if len(ep) < 100:
        return []
    acc = ep[["accelerometer_x", "accelerometer_y", "accelerometer_z"]].to_numpy()
    g = np.median(acc, axis=0)
    cosang = (acc @ g) / (np.linalg.norm(acc, axis=1) * np.linalg.norm(g) + 1e-9)
    tilt = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    tilt_s = pd.Series(tilt).rolling(5, center=True, min_periods=1).median().to_numpy()
    rpm = ep[["wheel_rpm_0", "wheel_rpm_1", "wheel_rpm_2", "wheel_rpm_3"]].abs().mean(axis=1).to_numpy()
    gyro = np.linalg.norm(ep[["gyroscope_x", "gyroscope_y", "gyroscope_z"]].to_numpy(), axis=1)
    shock_acc = np.abs(np.linalg.norm(acc, axis=1) - np.linalg.norm(g))
    t = ep["t"].to_numpy()
    out = []
    for a, b in runs((ep["cmd_lin"].to_numpy() > 0.5) & (rpm < 8), int(STALL_S * FPS)):
        out.append({"kind": "stall", "t": float(t[a]), "dur": float((b - a) / FPS),
                    "detail": f"commanded forward but wheels ~0 rpm for {(b - a) / FPS:.1f}s"})
    for a, b in runs(tilt_s > TILT_DEG, FPS):
        # IMU orientation drifts on these robots; only trust a tilt that starts with a real rotation spike
        if gyro[max(0, a - FPS):a + FPS].max() < SHOCK_GYRO:
            continue
        out.append({"kind": "tilt", "t": float(t[a]), "dur": float((b - a) / FPS),
                    "detail": f"body tilted {tilt_s[a:b].max():.0f} deg from normal for {(b - a) / FPS:.1f}s"})
    for i in np.where((gyro > SHOCK_GYRO) | (shock_acc > 1.0))[0]:
        if not out or abs(out[-1]["t"] - t[i]) > 3:
            out.append({"kind": "shock", "t": float(t[i]), "dur": 0.0,
                        "detail": f"gyro {gyro[i]:.0f} deg/s, accel jump {shock_acc[i]:.2f} g"})
    # behavior: the human driver reacting to the robot doing the wrong thing (near miss, wrong path)
    lin, ang = ep["cmd_lin"].to_numpy(), ep["cmd_ang"].to_numpy()
    k = FPS // 2
    for i in range(k, len(ep)):
        if lin[i - k] >= 0.7 and lin[i] <= 0.1 and rpm[i - k] > 60:
            kind, detail = "brake", f"hard brake: command {lin[i - k]:.1f} -> {lin[i]:.1f} in 0.5s at speed"
        elif abs(ang[i] - ang[i - k]) >= 0.8 and lin[i] > 0.4:
            kind, detail = "swerve", f"sudden swerve: steering {ang[i - k]:+.1f} -> {ang[i]:+.1f} in 0.5s"
        else:
            continue
        if not any(abs(e["t"] - t[i]) < 4 for e in out):
            out.append({"kind": kind, "t": float(t[i]), "dur": 0.0, "detail": detail})
    # jerky driving: stop-go throttle or twitchy steering sustained over a 3 s window
    jerk = pd.Series(np.abs(np.diff(lin, prepend=lin[0])) + 0.5 * np.abs(np.diff(ang, prepend=ang[0])))
    jw = jerk.rolling(3 * FPS, min_periods=FPS).sum().to_numpy()
    for i in np.where(jw > 6.0)[0]:
        if not any(abs(e["t"] - t[i]) < 6 for e in out):
            out.append({"kind": "jerky", "t": float(t[i]), "dur": 3.0,
                        "detail": f"jerky driving: {jw[i]:.1f} units of throttle/steering change in 3s"})
    for e in out:  # severity used for ranking
        e["severity"] = {"tilt": 3.0, "stall": 2.5, "brake": 2.0, "swerve": 1.5, "jerky": 1.5, "shock": 1.0}[e["kind"]] + min(e["dur"], 10) / 5
    out.sort(key=lambda e: -e["severity"])
    phys = [e for e in out if e["kind"] in ("tilt", "stall", "shock")][:2]
    behav = [e for e in out if e["kind"] in ("brake", "swerve", "jerky")][:2]
    return phys + behav  # at most two events per ride so one noisy robot can't flood the list


def cut(video: Path, start: float, dur: float, out: Path):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", str(video), "-t", f"{dur:.2f}",
                    "-vf", "scale=448:256:flags=lanczos", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out)],
                   check=True)


def main(top: int = 40, file_idx: int = 0):
    global OUT
    OUT = Path(__file__).parent / ("real_clips" if file_idx == 0 else f"real_clips_v{file_idx}")
    S, eps = load(file_idx)
    video = RAW / f"videos/observation.images.front/chunk-000/file-{file_idx:03d}.mp4"
    cands = []
    for e_id, ep in S.groupby("episode"):
        for ev in events(ep):
            cands.append({"episode": int(e_id), **ev})
    cands.sort(key=lambda c: -c["severity"])
    # keep a balanced mix: physical failures and behavior mistakes
    phys = [c for c in cands if c["kind"] in ("tilt", "stall", "shock")]
    behav = [c for c in cands if c["kind"] in ("brake", "swerve", "jerky")]
    cands = phys[: top // 2] + behav[: top - min(len(phys), top // 2)]
    print(f"{len(cands)} candidate events in {S.episode.nunique()} rides "
          f"({sum(c['kind'] == 'tilt' for c in cands)} tilt, {sum(c['kind'] == 'stall' for c in cands)} stall, "
          f"{sum(c['kind'] == 'shock' for c in cands)} shock, {sum(c['kind'] == 'brake' for c in cands)} brake, "
          f"{sum(c['kind'] == 'swerve' for c in cands)} swerve, {sum(c['kind'] == 'jerky' for c in cands)} jerky)")
    OUT.mkdir(parents=True, exist_ok=True)
    kept = []
    for c in cands[:top]:
        ep = S[S.episode == c["episode"]]
        t0 = max(0.0, c["t"] - PRE)
        t1 = min(float(ep["t"].max()), c["t"] + POST + c["dur"])
        name = f"ep{c['episode']:05d}_{c['kind']}_{c['t']:.0f}s"
        cut(video, float(eps.loc[c["episode"], "v_from"]) + t0, t1 - t0, OUT / f"{name}.mp4")
        w = ep[(ep["t"] >= t0) & (ep["t"] <= t1)]
        imu = pd.DataFrame({"t": (w["t"] - t0).round(2),
                            "ax": w.accelerometer_x, "ay": w.accelerometer_y, "az": w.accelerometer_z,
                            "gx": w.gyroscope_x, "gy": w.gyroscope_y, "gz": w.gyroscope_z,
                            "wheel_rpm": w[["wheel_rpm_0", "wheel_rpm_1", "wheel_rpm_2", "wheel_rpm_3"]].mean(axis=1).round(1),
                            "cmd_lin": w.cmd_lin.round(2), "cmd_ang": w.cmd_ang.round(2)})
        imu.round(3).to_csv(OUT / f"{name}.imu.csv", index=False)
        kept.append({**c, "clip": f"{name}.mp4", "clip_t_event": round(c["t"] - t0, 2)})
        print(f"  {name}: {c['detail']}")
    (OUT / "candidates.json").write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    import sys
    for i in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else [0]):
        main(file_idx=i)
