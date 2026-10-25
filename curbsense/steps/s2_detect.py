"""Step 2: Argon as off-board perception. Watch a ride clip (+ IMU), find the failure, estimate the scene."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from curbsense import argon

SYSTEM = """You are the failure analyst for a fleet of small sidewalk delivery robots
(4-wheel skid-steer, about 0.5 m long, 0.34 m wheel track, tall cargo bay, top speed ~2 m/s).
You review ride clips and sensor logs and explain exactly what went wrong so it can be
rebuilt in a physics simulator. Be quantitative. Never guess silently: if something cannot
be determined from the evidence, put it in `unknowns` and lower `confidence`."""

PROMPT = """Watch this ride clip{imu_note}.
1. Did the robot fail (tip over, roll, nose over, get stuck, hit something)? If yes, give the
   time in seconds where the failure becomes unrecoverable.
2. Explain the cause in one or two sentences, citing what you see (and the IMU if given).
3. Estimate the scene so we can rebuild it in simulation: what kind of situation it is (driving down
   off a curb, up onto a curb, up a slope, hitting a static obstacle such as a pole/furniture leg/wall corner,
   or a moving person/animal), curb height (m) or slope grade (deg) or obstacle distance/offset/size, the robot's approach
   angle relative to the curb/slope (deg, 0 = straight at it, positive = turned left), its speed when it
   reached the edge or slope (m/s), the surface (dry/wet/gravel...), and anything else that mattered.
4. Cross-check video against the sensor log. If they disagree, say which one you trust and why.
Use visual cues for scale: the robot is ~0.5 m long and its wheels are 0.16 m in diameter."""


class SceneEstimate(BaseModel):
    scenario_kind: Literal["drop", "climb", "slope", "obstacle", "pedestrian", "flat"] = Field(
        description="drop = driving down off a curb; climb = driving up onto a curb; slope = uphill grade; "
                    "obstacle = a static thing in the path (pole, chair/stool leg, wall corner, bin); "
                    "pedestrian = a moving person or animal crossing the path; flat = none")
    obstacle_distance_m: float = Field(default=1.5, description="how far ahead the obstacle/person was when first visible")
    obstacle_lateral_m: float = Field(default=0.0, description="obstacle offset from the robot's path (+ = left)")
    obstacle_radius_m: float = Field(default=0.15, description="size of the obstacle (chair leg ~0.03, pole ~0.08, bin ~0.3)")
    mover_speed_mps: float = Field(default=0.0, description="speed of a moving person/animal, 0 if static")
    curb_height_m: float = Field(description="curb step height in meters (0 if no curb)")
    slope_deg: float = Field(default=0.0, description="uphill grade in degrees (0 if flat)")
    approach_angle_deg: float = Field(description="0 = straight off the curb; positive = turned left")
    speed_mps: float = Field(description="robot speed at the curb edge")
    t_edge_sec: float | None = Field(description="clip time when the front wheels cross the curb edge")
    surface: str = Field(description="dry concrete, wet, gravel, ...")
    other_factors: list[str] = Field(default_factory=list)


class FailureReport(BaseModel):
    failed: bool
    t_sec: float | None = Field(description="when the failure became unrecoverable")
    failure_type: Literal["nose_over", "rollover", "high_center", "stuck", "collision", "none", "other"]
    cause: str
    evidence: list[str] = Field(description="specific observations with timestamps")
    scene: SceneEstimate
    confidence: float = Field(ge=0, le=1)
    unknowns: list[str]


def detect(clip: str | Path, imu_csv: str | Path | None = None, fps: float = 8, **kw) -> FailureReport:
    parts = [argon.video(clip, fps=fps)]
    imu_note = ""
    if imu_csv:
        imu_note = " together with the robot's IMU log (CSV below; t in seconds from clip start)"
        parts.append(argon.text("IMU log:\n" + Path(imu_csv).read_text()[:60000]))
    parts.append(argon.text(PROMPT.format(imu_note=imu_note)))
    return argon.ask(parts, FailureReport, system=SYSTEM, thinking=kw.pop("thinking", "high"), tag="detect", **kw)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--imu")
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = detect(a.clip, a.imu)
    js = rep.model_dump_json(indent=2)
    print(js)
    if a.out:
        Path(a.out).write_text(js)
