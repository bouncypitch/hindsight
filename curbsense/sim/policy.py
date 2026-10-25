"""The robot's driving policy: a small parameterized controller that RL (CMA-ES) tunes.

Perception is idealized: the controller gets distance to the curb edge and heading relative
to the curb normal (on the real robot this would come from onboard perception).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields

import numpy as np


@dataclass
class PolicyParams:
    cruise_speed: float = 1.1       # m/s on open sidewalk
    slow_dist: float = 0.0          # m before the edge where curb behavior kicks in (0 = never)
    slow_speed: float = 1.1         # m/s while crossing the edge
    descent_angle_deg: float = 0.0  # heading to hold while crossing the edge (0 = straight off)
    heading_gain: float = 0.0       # yaw-rate gain toward the descent angle (0 = keep heading)
    align_tol_deg: float = 90.0     # stop and turn in place near the edge if heading error exceeds this
    yield_dist: float = 0.0         # m: stop for a person in or near the path closer than this (0 = never yields)
    avoid_dist: float = 0.0         # m: start steering around things closer than this (0 = never avoids)
    avoid_gain: float = 0.0         # how hard to steer away

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in NAMES])

    @classmethod
    def from_vector(cls, x) -> "PolicyParams":
        x = np.clip(np.asarray(x, float), LO, HI)
        return cls(**dict(zip(NAMES, map(float, x))))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def load(cls, path) -> "PolicyParams":
        return cls(**json.load(open(path)))


NAMES = [f.name for f in fields(PolicyParams)]
LO = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0])
HI = np.array([2.0, 2.0, 2.0, 60.0, 4.0, 90.0, 4.0, 4.0, 4.0])

ROBOT_R = 0.30  # m, matches world.ROBOT_R
NAIVE = PolicyParams()  # the shipped policy: drives straight at cruise speed, ignores the curb


def act(p: PolicyParams, dist_to_edge: float, heading: float, obs: dict | None = None) -> tuple[float, float]:
    """-> (forward speed m/s, yaw rate rad/s). heading in rad, 0 = facing straight off the curb.
    obs (if any): nearest obstacle/person in the robot frame {dx, dy, dist, moving}."""
    v, w = _curb(p, dist_to_edge, heading)
    if obs:
        in_path = obs["dx"] > 0 and abs(obs["dy"]) < (1.2 if obs["moving"] else 0.6)
        if obs["moving"] and in_path and obs["dist"] < p.yield_dist:
            return 0.0, 0.0  # yield to the person
        need = obs.get("r", 0.2) + ROBOT_R + 0.10 + 0.1 * p.avoid_gain  # lateral gap to aim for
        if 0 < obs["dx"] < p.avoid_dist and abs(obs["dy"]) < need:
            # aim at a point beside the obstacle with enough clearance, on the side it already leans to
            side = -1.0 if obs["dy"] >= 0 else 1.0
            aim = math.atan2(obs["dy"] + side * need, max(obs["dx"], 0.3))
            w = float(np.clip(p.heading_gain * aim, -2.0, 2.0))
            v = min(v, max(0.3, p.slow_speed))
        elif obs["dx"] <= 0 and p.avoid_dist > 0:
            w = float(np.clip(-p.heading_gain * 0.5 * heading, -2.0, 2.0))  # passed it: straighten out
    return v, w


def _curb(p: PolicyParams, dist_to_edge: float, heading: float) -> tuple[float, float]:
    near = p.slow_dist > 0 and -0.6 < dist_to_edge < p.slow_dist
    if not near:
        return p.cruise_speed, 0.0
    # hold the descent angle on whichever side the robot is already turned toward
    target = np.copysign(np.radians(p.descent_angle_deg), heading if heading != 0 else 1.0)
    err = heading - target
    w = float(np.clip(-p.heading_gain * err, -2.0, 2.0))
    v = 0.0 if abs(np.degrees(err)) > p.align_tol_deg else p.slow_speed
    return v, w
