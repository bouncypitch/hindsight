"""Photoreal twin of a REAL fleet failure (Mini-4K): Argon rebuilds the room/street from the real frames, and the
robot moves as the MuJoCo physics twin says - the old policy (crashes) or the retrained one (avoids).

uv run python -m curbsense.viz.real_twin demo_assets/141102_tip_over_443s
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from curbsense.viz.coco_twin import build_from
from curbsense.viz.shoot import capture

# Earth Rover Mini front camera: low and wide (approximate)
CAMERA = {"hfov_deg": 120.0, "vfov_deg": 88.0, "height_m": 0.12, "aspect": 16 / 9}


def to_twin(sim_traj: dict) -> dict:
    """MuJoCo export (p, q per frame) -> twin trajectory (x, y, yaw), plus the obstacle for the scene."""
    frames = []
    for f in sim_traj["frames"]:
        w, x, y, z = f["q"]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        frames.append({"t": f["t"], "x": f["p"][0], "y": f["p"][1], "yaw": round(yaw, 4), "z": f["p"][2]})
    world = sim_traj["meta"]["world"]
    return {"fps": 30, "camera": CAMERA, "frames": frames, "outcome": sim_traj["meta"]["outcome"],
            "obstacle": world.get("obstacle"), "pedestrian": world.get("pedestrian")}


def build(asset_dir: Path, rounds: int = 2) -> Path:
    old = to_twin(json.loads((asset_dir / "07_traj_old.json").read_text()))
    new = to_twin(json.loads((asset_dir / "07_traj_new.json").read_text()))
    out = asset_dir / "photoreal"
    out.mkdir(parents=True, exist_ok=True)
    (out / "trajectory_new.json").write_text(json.dumps(new))
    diag = json.loads((asset_dir / "02_diagnosis.json").read_text())
    real = asset_dir / "01_real_failure.mp4"
    t_fail = diag.get("t_sec") or 7.0
    real_times = [max(0.2, t_fail - d) for d in (4.0, 2.5, 1.5, 0.6)]
    dur = old["frames"][-1]["t"]
    ev = next((f["t"] for f in old["frames"] if old["obstacle"] and abs(f["x"] - old["obstacle"]["x"]) < 0.5), dur * 0.7)
    twin_times = [max(0.0, ev - d) for d in (1.6, 1.0, 0.6, 0.2)]
    notes = (f"\n\n## This clip\nArgon's diagnosis of the real failure: {diag['cause']}\nThe obstacle the robot hits is at "
             f"{old['obstacle']} in trajectory coordinates (x, y meters, r radius): model it as the real object you see in "
             "the frames (e.g. the barstool / chair: put the leg the robot hits exactly there, the rest of the furniture "
             "around it), plus the room around it. The trajectory may have a `z` per frame (robot height from physics); use it. "
             "The reference frames are from BEFORE the impact (approach), matching the listed twin times.")
    return build_from(real, old, out, real_times, twin_times, notes=notes, rounds=rounds)


def videos(asset_dir: Path) -> None:
    out = asset_dir / "photoreal"
    for tag, tp in [("old", out / "trajectory.json"), ("new", out / "trajectory_new.json")]:
        for cam in ("robot", "chase"):
            capture(out / "twin.html", tp, out / f"twin_{tag}_{cam}.mp4", cam=cam, fps=30, size=(960, 540))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(asset_dir / "01_real_failure.mp4"),
                    "-i", str(out / "twin_old_robot.mp4"), "-filter_complex",
                    "[0]scale=960:540,fps=30[a];[1]scale=960:540,fps=30[b];[a][b]hstack=shortest=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "real_vs_twin.mp4")], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--videos-only", action="store_true")
    a = ap.parse_args()
    d = Path(a.asset_dir).resolve()
    if not a.videos_only:
        print(build(d, a.rounds))
    videos(d)
    print("done")
