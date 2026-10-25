"""FrodoBots-Mini-4K (CC-BY-SA-4.0): find curb falls, tip-overs and collisions in raw rides, cut 1024x576 clips.

Each ride folder has the front camera as HLS (.m3u8 + .ts, uid_s_1000 = front), per-frame timestamps, a clean
50 Hz accelerometer (g), and control/wheel-rpm logs, all on the same clock. Candidates come from physics:

  tip_over   gravity direction moves > 55 deg from the ride's normal and stays there >= 2 s
  curb_fall  brief free-fall (|a| < 0.45 g for >= 60 ms) followed within 0.5 s by an impact (|a| > 1.8 g)
  collision  sharp horizontal deceleration (> 0.9 g perpendicular to gravity) while driving forward

Every candidate is then checked by Argon (curbsense.steps.s0_confirm), which must see it happen in the video.

uv run python -m data.mini4k  ->  data/mini4k_clips/*.mp4 + candidates.json
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
RIDES = ROOT / "raw" / "mini4k" / "rides"
OUT = ROOT / "mini4k_clips"
PRE, POST = 7.0, 5.0


def _ride_id(ride: Path) -> str:
    return ride.name.split("_")[1]


def load_imu(ride: Path) -> pd.DataFrame:
    f = next(ride.glob("imu_data_*.csv"))
    rows = []
    for acc in pd.read_csv(f, usecols=["accelerometer"])["accelerometer"]:
        for ax, ay, az, t in ast.literal_eval(acc):
            rows.append((float(t), float(ax), float(ay), float(az)))
    df = pd.DataFrame(rows, columns=["ts", "ax", "ay", "az"]).drop_duplicates("ts").sort_values("ts")
    return df.reset_index(drop=True)


def load_control(ride: Path) -> pd.DataFrame:
    c = pd.read_csv(next(ride.glob("control_data_*.csv")))
    c["rpm"] = c[["rpm_1", "rpm_2", "rpm_3", "rpm_4"]].abs().mean(axis=1)
    return c.rename(columns={"timestamp": "ts"}).sort_values("ts").reset_index(drop=True)


def front_video(ride: Path) -> tuple[Path, float] | None:
    """Concatenate the front-camera HLS into one mp4; returns (mp4, unix time of its first frame)."""
    m3u8 = next(iter(sorted((ride / "recordings").glob("*uid_s_1000__uid_e_video.m3u8"))), None)
    if m3u8 is None:
        return None
    segs = [l.strip() for l in m3u8.read_text().splitlines() if l.strip().endswith(".ts")]
    segs = [s for s in segs if (ride / "recordings" / s).exists()]
    if not segs:
        return None
    out = ride / "front.mp4"
    if not out.exists():
        lst = ride / "front_concat.txt"
        lst.write_text("".join(f"file '{(ride / 'recordings' / s).resolve()}'\n" for s in segs))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", str(out)], check=True)
    stamp = segs[0].rsplit("_", 1)[1].removesuffix(".ts")  # 20260108143545515 (UTC, ms)
    t0 = pd.Timestamp(f"{stamp[:8]}T{stamp[8:14]}.{stamp[14:]}", tz="UTC").timestamp()
    return out, t0


def duration(mp4: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(mp4)],
                       capture_output=True, text=True)
    return float(r.stdout.strip() or 0)


def runs(mask: np.ndarray, min_len: int):
    out, start = [], None
    for i, m in enumerate(np.append(mask, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out


def detect(imu: pd.DataFrame, ctl: pd.DataFrame) -> list[dict]:
    a = imu[["ax", "ay", "az"]].to_numpy()
    ts = imu["ts"].to_numpy()
    norm = np.linalg.norm(a, axis=1)
    g = np.median(a, axis=0)
    g = g / np.linalg.norm(g)
    cosang = (a @ g) / (norm + 1e-9)
    tilt = pd.Series(np.degrees(np.arccos(np.clip(cosang, -1, 1)))).rolling(25, center=True, min_periods=5).median().to_numpy()
    horiz = np.linalg.norm(a - np.outer(a @ g, g), axis=1)
    lin = np.interp(ts, ctl["ts"], ctl["linear"]) if len(ctl) else np.zeros_like(ts)
    rpm = np.interp(ts, ctl["ts"], ctl["rpm"]) if len(ctl) else np.zeros_like(ts)
    out = []
    for s, e in runs(tilt > 55, 100):  # 50 Hz -> 2 s
        out.append({"kind": "tip_over", "ts": float(ts[s]), "score": float(np.nanmax(tilt[s:e])),
                    "detail": f"gravity tilted {np.nanmax(tilt[s:e]):.0f} deg from normal for {(e - s) / 50:.1f}s"})
    for s, e in runs(norm < 0.45, 3):  # >= 60 ms of free fall
        win = norm[e:e + 25]
        if len(win) and win.max() > 1.8:
            out.append({"kind": "curb_fall", "ts": float(ts[s]), "score": float(win.max()),
                        "detail": f"free fall {(e - s) * 20} ms then impact {win.max():.1f} g"})
    for i in np.where(horiz > 0.9)[0]:
        if lin[i] > 0.15 or rpm[max(0, i - 25):i].mean() > 20:
            out.append({"kind": "collision", "ts": float(ts[i]), "score": float(horiz[i]),
                        "detail": f"horizontal shock {horiz[i]:.2f} g while driving (cmd {lin[i]:.2f}, rpm {rpm[i]:.0f})"})
    out.sort(key=lambda e: -e["score"])
    kept = []
    for e in out:
        if not any(abs(k["ts"] - e["ts"]) < 8 for k in kept):
            kept.append(e)
    return kept


def main(max_per_ride: int = 6):
    OUT.mkdir(parents=True, exist_ok=True)
    cands = []
    for ride in sorted(RIDES.glob("ride_*")):
        try:
            imu, ctl = load_imu(ride), load_control(ride)
            fv = front_video(ride)
        except Exception as ex:  # noqa: BLE001 - partially downloaded rides are expected
            print(f"skip {ride.name}: {ex}")
            continue
        if fv is None:
            print(f"skip {ride.name}: no front video")
            continue
        mp4, t0 = fv
        dur = duration(mp4)
        evs = [e for e in detect(imu, ctl) if PRE <= e["ts"] - t0 <= dur - 1][:max_per_ride]
        print(f"{ride.name}: {dur / 60:.1f} min video, {len(imu) / 50 / 60:.1f} min IMU, {len(evs)} candidates "
              f"({', '.join(e['kind'] for e in evs)})")
        for e in evs:
            t = e["ts"] - t0
            name = f"{_ride_id(ride)}_{e['kind']}_{t:.0f}s"
            clip = OUT / f"{name}.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t - PRE:.2f}", "-i", str(mp4),
                            "-t", f"{PRE + POST:.2f}", "-vf", "scale=1024:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-an", str(clip)], check=True)
            w = imu[(imu.ts >= e["ts"] - PRE) & (imu.ts <= e["ts"] + POST)].iloc[::5]  # 10 Hz for Argon
            pd.DataFrame({"t": (w.ts - (e["ts"] - PRE)).round(2), "ax_g": w.ax, "ay_g": w.ay, "az_g": w.az}).round(3) \
                .to_csv(clip.with_suffix(".imu.csv"), index=False)
            cands.append({"ride": ride.name, **e, "clip_t_event": PRE, "clip": str(clip.relative_to(ROOT.parent))})
    (OUT / "candidates.json").write_text(json.dumps(cands, indent=1))
    print(f"{len(cands)} candidates")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
