"""Render labeled sim ride clips (+ IMU CSV) to test the Argon steps before real data is wired in.

uv run python -m data.make_sim_clips  ->  data/sim_clips/{good,bad}/*.mp4 + *.imu.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from curbsense.scenario import ScenarioSpec
from curbsense.sim.reference import NAIVE, SAFE
from curbsense.sim.rollout import run, summary

OUT = Path(__file__).parent / "sim_clips"

CLIPS = [
    ("good", "safe_a", SAFE, ScenarioSpec(curb_height=0.20, approach_angle_deg=6, name="safe_a")),
    ("good", "safe_b", SAFE, ScenarioSpec(curb_height=0.22, approach_angle_deg=-10, name="safe_b")),
    ("bad", "flip_a", NAIVE, ScenarioSpec(curb_height=0.20, approach_angle_deg=8, name="flip_a")),
    ("bad", "flip_b", NAIVE, ScenarioSpec(curb_height=0.21, approach_angle_deg=-4, start_dist=2.5, name="flip_b")),
    ("bad", "flip_c", NAIVE, ScenarioSpec(curb_height=0.19, approach_angle_deg=14, friction=0.7, name="flip_c")),
]


def write_imu(states: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "ax", "ay", "az", "gx", "gy", "gz", "odom_speed"])
        for s in states:
            w.writerow([s["t"], *[round(a, 3) for a in s["accel"]], *[round(g, 3) for g in s["gyro"]], round(s["speed"], 3)])


if __name__ == "__main__":
    truth = {}
    for label, name, policy, spec in CLIPS:
        d = OUT / label
        d.mkdir(parents=True, exist_ok=True)
        r = run(spec, policy, video=d / f"{name}.mp4")
        write_imu(r["states"], d / f"{name}.imu.csv")
        truth[name] = {"label": label, "spec": json.loads(spec.to_json()), **summary(r)}
        print(name, label, "success" if r["success"] else f"FELL at {r['fail_t']}s")
    (OUT / "ground_truth.json").write_text(json.dumps(truth, indent=1))
