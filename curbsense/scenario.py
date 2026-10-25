"""Scenario = one curb-drop situation the robot can face. Argon estimates these from video."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class ScenarioSpec:
    kind: str = "drop"               # drop | climb | slope | obstacle (static bollard/pole/bin) | pedestrian (crossing person)
    curb_height: float = 0.20        # m, curb step height (drop / climb)
    slope_deg: float = 0.0           # incline for kind="slope"
    approach_angle_deg: float = 8.0   # robot heading vs curb normal at start (0 = square)
    start_dist: float = 2.0          # m from robot center to curb edge
    lateral_offset: float = 0.0      # m along the curb
    friction: float = 0.9            # tire/ground sliding friction
    observed_speed: float = 1.1      # m/s the robot was driving in the original clip
    obs_x: float = 1.5               # m ahead of the start line: obstacle position / where the person crosses
    obs_y: float = 0.0               # m lateral position of a static obstacle (0 = dead ahead)
    obs_r: float = 0.15              # m obstacle radius (bollard ~0.12, bin ~0.3, pole ~0.08)
    ped_speed: float = 1.3           # m/s walking speed of the crossing person
    ped_start_y: float = -2.5        # m where the person starts (walks across to +|y|)
    ped_t0: float = 0.5              # s when the person starts walking
    name: str = "scenario"

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "ScenarioSpec":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


# Physically sensible bounds; anything Argon proposes is clipped to these.
BOUNDS = {
    "curb_height": (0.0, 0.25),
    "slope_deg": (0.0, 30.0),
    "approach_angle_deg": (-70.0, 70.0),
    "start_dist": (0.8, 4.0),
    "lateral_offset": (-1.0, 1.0),
    "friction": (0.3, 1.5),
    "observed_speed": (0.2, 2.5),
    "obs_x": (0.5, 4.0),
    "obs_y": (-1.0, 1.0),
    "obs_r": (0.05, 0.5),
    "ped_speed": (0.3, 2.5),
    "ped_start_y": (-4.0, 4.0),
    "ped_t0": (0.0, 4.0),
}


def clip(spec: ScenarioSpec) -> ScenarioSpec:
    d = asdict(spec)
    for k, (lo, hi) in BOUNDS.items():
        d[k] = float(np.clip(d[k], lo, hi))
    return ScenarioSpec(**d)


def randomize(spec: ScenarioSpec, ranges: dict[str, tuple[float, float]], n: int, seed: int = 0) -> list[ScenarioSpec]:
    """n variants of `spec`; each key in `ranges` gets a uniform +/- delta (lo, hi) around the base value."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        d = asdict(spec)
        for k, (lo, hi) in ranges.items():
            d[k] = d[k] + rng.uniform(lo, hi)
        d["name"] = f"{spec.name}_v{i:03d}"
        out.append(clip(ScenarioSpec(**d)))
    return out
