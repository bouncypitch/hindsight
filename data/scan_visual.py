"""Find curb falls, tip-overs and collisions from what the robot's camera actually saw.

Per frame (5 fps) we track corner features between consecutive frames and fit a similarity transform, giving the
image's rotation (deg), vertical shift and overall motion. Telemetry glitches cannot fake these.

  collision  camera moving forward, then stops dead within ~0.5 s while the driver still commands forward
  curb_fall  sharp vertical jolt of the whole image together with a gyro spike
  tip_over   image rotates > 30 deg within 2 s and the camera is (nearly) still afterwards

uv run python -m data.scan_visual 0 2 3   ->  data/visual_candidates/<kind>/*.mp4 + candidates.json
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from data import mine_failures as mf

FPS = 5.0
W, H = 224, 128
OUT = Path(__file__).parent / "visual_candidates"


def frame_motion(video: str, start: float, dur: float) -> pd.DataFrame:
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", video, "-t", f"{dur:.2f}",
           "-vf", f"fps={FPS},scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    rows = [{"rot": 0.0, "ty": 0.0, "tx": 0.0, "motion": 0.0, "n": 0}]
    for a, b in zip(frames[:-1], frames[1:]):
        p0 = cv2.goodFeaturesToTrack(a, maxCorners=150, qualityLevel=0.01, minDistance=6)
        r = {"rot": np.nan, "ty": np.nan, "tx": np.nan, "motion": np.nan, "n": 0}
        if p0 is not None and len(p0) >= 12:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p0, None, winSize=(15, 15), maxLevel=2)
            ok = st.ravel() == 1
            if ok.sum() >= 10:
                m, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                d = np.linalg.norm((p1[ok] - p0[ok]).reshape(-1, 2), axis=1)
                if m is not None:
                    r = {"rot": math.degrees(math.atan2(m[1, 0], m[0, 0])), "ty": float(m[1, 2]), "tx": float(m[0, 2]),
                         "motion": float(np.median(d)), "n": int(inl.sum()) if inl is not None else 0}
        rows.append(r)
    df = pd.DataFrame(rows)
    df["t"] = np.arange(len(df)) / FPS
    return df


def detect(df: pd.DataFrame, ep: pd.DataFrame) -> list[dict]:
    t_tel = ep["t"].to_numpy()
    cmd = np.interp(df.t, t_tel, ep["cmd_lin"].to_numpy())
    gyro = np.interp(df.t, t_tel, np.linalg.norm(ep[["gyroscope_x", "gyroscope_y", "gyroscope_z"]].to_numpy(), axis=1))
    mot = df["motion"].fillna(0).to_numpy()
    rot = df["rot"].fillna(0).to_numpy()
    ty = df["ty"].fillna(0).to_numpy()
    k = int(FPS)
    out = []
    for i in range(k, len(df) - 2 * k):
        pre, post = np.median(mot[i - k:i]), np.median(mot[i + 1:i + 1 + int(1.5 * FPS)])
        if pre >= 1.0 and post <= 0.15 and cmd[i + 1:i + 1 + int(1.5 * FPS)].mean() >= 0.4:
            out.append({"kind": "collision", "t": float(df.t[i]), "score": float(pre / (post + 0.05)),
                        "detail": f"camera motion {pre:.2f} -> {post:.2f} px/frame while command stays {cmd[i + 1:i + 8].mean():.2f}"})
        if abs(ty[i]) >= 12 and gyro[max(0, i - 2):i + 3].max() >= 60:
            out.append({"kind": "curb_fall", "t": float(df.t[i]), "score": float(abs(ty[i]) * gyro[max(0, i - 2):i + 3].max() / 60),
                        "detail": f"vertical image jolt {ty[i]:+.0f}px with gyro {gyro[max(0, i - 2):i + 3].max():.0f} deg/s"})
        cum = rot[i:i + 2 * k].sum()
        after = np.median(mot[i + 2 * k:i + 4 * k]) if i + 4 * k <= len(mot) else 9
        if abs(cum) >= 30 and after <= 0.3:
            out.append({"kind": "tip_over", "t": float(df.t[i]), "score": float(abs(cum)),
                        "detail": f"image rotated {cum:+.0f} deg in 2 s, then still ({after:.2f} px/frame)"})
    # keep the strongest event per kind per 6 s window
    out.sort(key=lambda e: -e["score"])
    kept = []
    for e in out:
        if not any(k_["kind"] == e["kind"] and abs(k_["t"] - e["t"]) < 6 for k_ in kept):
            kept.append(e)
    return kept


def scan_episode(args):
    video, e_id, v_from, ep = args
    try:
        df = frame_motion(video, v_from, float(ep["t"].max()) + 0.1)
        return [{"episode": int(e_id), "v_from": v_from, **e} for e in detect(df, ep)]
    except Exception as ex:  # noqa: BLE001
        return [{"episode": int(e_id), "error": str(ex)}]


def main(files: list[int], per_kind: int = 15):
    jobs = []
    for f in files:
        S, eps = mf.load(f)
        video = str(mf.RAW / f"videos/observation.images.front/chunk-000/file-{f:03d}.mp4")
        for e_id, ep in S.groupby("episode"):
            if len(ep) >= 100:
                jobs.append((video, e_id, float(eps.loc[e_id, "v_from"]), ep.reset_index(drop=True)))
    print(f"scanning {len(jobs)} rides", flush=True)
    with Pool(8) as pool:
        res = [e for r in pool.imap_unordered(scan_episode, jobs, chunksize=4) for e in r]
    errs = [r for r in res if "error" in r]
    cands = [r for r in res if "error" not in r]
    print(f"{len(cands)} candidates, {len(errs)} rides failed", flush=True)
    vid_of = {j[1]: j[0] for j in jobs}
    OUT.mkdir(parents=True, exist_ok=True)
    kept = []
    for kind in ("collision", "curb_fall", "tip_over"):
        ks = sorted([c for c in cands if c["kind"] == kind], key=lambda c: -c["score"])[:per_kind]
        print(f"{kind}: {sum(c['kind'] == kind for c in cands)} found, keeping top {len(ks)}")
        (OUT / kind).mkdir(exist_ok=True)
        for c in ks:
            name = f"ep{c['episode']:05d}_{c['t']:.0f}s"
            t0 = max(0.0, c["t"] - 5)
            mf.cut(Path(vid_of[c["episode"]]), c["v_from"] + t0, 10.0, OUT / kind / f"{name}.mp4")
            kept.append({**c, "clip": f"{kind}/{name}.mp4", "clip_t_event": round(c["t"] - t0, 1)})
    (OUT / "candidates.json").write_text(json.dumps(kept, indent=1))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0])
