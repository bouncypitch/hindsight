"""Step 0: cheap fleet-scale screening. Flash-Lite reviews every mined clip; Argon only sees the flagged ones."""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from curbsense import argon

TRIAGE_MODEL = "gemini-3.5-flash-lite"

PROMPT = """This is a {dur:.0f}s clip from the front camera of a small sidewalk delivery robot, with its sensor log.
A sensor rule flagged it: {detail} (around t={t:.1f}s in the clip).
Columns: t (s), ax/ay/az accelerometer (g), gx/gy/gz gyro (deg/s), wheel_rpm (mean of 4), cmd_lin/cmd_ang
operator command (0..1, -1..1).

Decide quickly whether something actually went wrong here. Two kinds count:
- PHYSICAL failure: stuck, blocked, fell, tipped, collided, dropped off or failed to climb a curb, wheel trapped.
- BEHAVIOR failure (the robot did the wrong thing even if nothing broke): passed too close to or cut off a
  pedestrian, drove toward a person/obstacle and needed a last-second brake or swerve, entered the roadway or a
  bike lane, wrong side of the path, drove onto grass/off the sidewalk, hesitated or oscillated, jerky stop-go
  or twitchy steering that would look unsafe to people nearby, blocked others.
A brake or swerve is a failure only if the robot's own driving caused the danger; braking for a person who
suddenly steps out is correct behavior. Otherwise call it a false alarm (normal stop, sensor glitch, normal bump)."""


class Triage(BaseModel):
    failure: bool
    category: Literal["curb_drop", "curb_climb", "obstacle_collision", "stuck_terrain", "tipped", "wheel_trap",
                      "pedestrian_conflict", "unsafe_pedestrian_pass", "near_miss", "entered_roadway",
                      "off_path", "hesitation", "jerky_driving", "false_alarm", "other"]
    failure_class: Literal["physical", "behavior", "none"]
    curb_related: bool
    severity: int = Field(ge=0, le=3, description="0 none .. 3 robot cannot continue without help")
    one_line: str


def triage_one(clip_dir: Path, c: dict) -> dict:
    clip = clip_dir / c["clip"]
    imu = clip.with_suffix(".imu.csv").read_text()
    try:
        r: Triage = argon.ask([argon.video(clip, fps=4),
                               argon.text(PROMPT.format(dur=c.get("dur", 0) + 10, detail=c["detail"], t=c["clip_t_event"])
                                          + "\n\nSensor log:\n" + imu[:20000])],
                              Triage, model=TRIAGE_MODEL, thinking="low", tag=f"triage:{c['clip']}")
        return {**c, "triage": r.model_dump()}
    except Exception as e:  # noqa: BLE001 - keep going; one bad clip shouldn't stop the fleet scan
        return {**c, "triage": {"error": str(e)}}


def triage(clip_dir: Path, workers: int = 8) -> list[dict]:
    cands = json.loads((clip_dir / "candidates.json").read_text())
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda c: triage_one(clip_dir, c), cands))
    res.sort(key=lambda r: (-r["triage"].get("severity", -1), not r["triage"].get("curb_related", False)))
    (clip_dir / "triage.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/real_clips")
    a = ap.parse_args()
    for r in triage(Path(a.clips)):
        t = r["triage"]
        flag = "ERR " if "error" in t else (("FAIL" if t["failure_class"] == "physical" else "BHVR") if t["failure"] else "ok  ")
        print(f"{flag} sev{t.get('severity', '-')} {t.get('category', ''):19s} curb={t.get('curb_related', '')!s:5s} "
              f"{r['clip']:28s} {t.get('one_line', t.get('error', ''))[:110]}")
