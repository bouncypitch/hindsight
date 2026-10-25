"""Run one episode: policy drives the robot off the curb; we record state, reward, failure, video."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import mujoco
import numpy as np

from curbsense.scenario import ScenarioSpec
from curbsense.sim import world
from curbsense.sim.policy import NAIVE, PolicyParams, act

CTRL_DT = 0.05          # 20 Hz control, like a real robot loop
MAX_T = 16.0
TIP_DEG = 75.0
STALL_WINDOW = 2.5      # s without forward progress while commanded forward = stalled (stuck or rolling back)


def _euler(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def obstacle_rel(spec: ScenarioSpec, data, t: float) -> dict | None:
    """Nearest obstacle/person in the robot frame (what onboard perception would report)."""
    pos = world.obstacle_pos(spec, t)
    if pos is None:
        return None
    x, y = data.qpos[:2]
    yaw = _euler(data.qpos[3:7])[2]
    ddx, ddy = pos[0] - x, pos[1] - y
    dx = math.cos(yaw) * ddx + math.sin(yaw) * ddy
    dy = -math.sin(yaw) * ddx + math.cos(yaw) * ddy
    r = spec.obs_r if spec.kind == "obstacle" else 0.22
    dist = math.hypot(ddx, ddy)
    return {"dx": dx, "dy": dy, "dist": dist, "clearance": dist - r - world.ROBOT_R, "r": r,
            "moving": spec.kind == "pedestrian"}


def state(model, data, t: float, cmd=(0.0, 0.0), spec: ScenarioSpec | None = None) -> dict:
    x, y, z = data.qpos[:3]
    roll, pitch, yaw = _euler(data.qpos[3:7])
    vx, vy = data.qvel[:2]
    obs = obstacle_rel(spec, data, t) if spec is not None else None
    flat = spec is not None and spec.kind in ("obstacle", "pedestrian")
    return {
        "t": round(t, 3), "x": float(x), "y": float(y), "z": float(z),
        "roll_deg": math.degrees(roll), "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw),
        "speed": float(math.hypot(vx, vy)), "vx": float(vx),
        "dist_to_edge": 99.0 if flat else float(-x), "heading_err_deg": math.degrees(yaw),
        "obs": obs, "clearance": obs["clearance"] if obs else None,
        "on_road": bool(x > 0.3), "accel": [float(a) for a in data.sensor("acc").data],
        "gyro": [float(g) for g in data.sensor("gyro").data],
        "cmd_v": cmd[0], "cmd_w": cmd[1],
    }


def tipped(s: dict) -> bool:
    return abs(s["roll_deg"]) > TIP_DEG or abs(s["pitch_deg"]) > TIP_DEG


PERSONAL_SPACE = 0.25  # m of clearance a robot must keep from a person
GRAZE = 0.05           # m: passing closer than this to an obstacle counts as hitting it (Argon caught RL side-scraping)


class Monitor:
    """Decides when an episode ends: fell, stalled, collided, near_miss, success (past the goal), or running."""

    def __init__(self, spec: ScenarioSpec):
        self.spec = spec
        self.goal = world.goal_x(spec)
        self.hist: list[tuple[float, float]] = []

    def update(self, s: dict) -> str | None:
        if tipped(s):
            return "fell"
        if s["clearance"] is not None:
            if s["clearance"] < GRAZE:
                return "collided"
            if self.spec.kind == "pedestrian" and s["clearance"] < PERSONAL_SPACE:
                return "near_miss"
        if s["x"] > self.goal:
            return "success"
        self.hist.append((s["t"], s["x"]))
        old = [x for t, x in self.hist if t <= s["t"] - STALL_WINDOW]
        if old and s["cmd_v"] > 0.05 and s["x"] - old[-1] < 0.05:
            return "stalled"
        return None


def run(spec: ScenarioSpec, params: PolicyParams = NAIVE,
        video: str | Path | None = None, view: str = "chase", max_t: float = MAX_T) -> dict:
    """One episode. Returns per-tick states plus episode metrics a reward or a judge can use."""
    model, data = world.build(spec)
    substeps = int(round(CTRL_DT / model.opt.timestep))
    every = int(round(1 / (30 * model.opt.timestep)))  # 30 fps video
    renderer = mujoco.Renderer(model, 360, 640) if video else None
    cam = _camera(model, view) if video else None
    frames, states = [], []
    peak_impact = 0.0
    step = 0
    acc_win = []  # 20 ms window: a filtered IMU reading, not raw rigid-contact spikes

    def advance():
        nonlocal step, peak_impact
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            step += 1
            acc_win.append(float(np.linalg.norm(data.sensor("acc").data)))
            if len(acc_win) > 10:
                acc_win.pop(0)
            peak_impact = max(peak_impact, abs(sum(acc_win) / len(acc_win) - 9.81))
            if renderer is not None and step % every == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())

    t, outcome, mon = 0.0, None, Monitor(spec)
    min_clear = None
    while t < max_t:
        s = state(model, data, t, spec=spec)
        v, w = act(params, s["dist_to_edge"], math.radians(s["heading_err_deg"]), s["obs"])
        world.set_drive(data, v, w)
        world.move_actors(spec, data, t)
        advance()
        t += CTRL_DT
        s = state(model, data, t, (v, w), spec)
        if s["clearance"] is not None:
            min_clear = s["clearance"] if min_clear is None else min(min_clear, s["clearance"])
        states.append(s)
        outcome = mon.update(s)
        if outcome:
            break
    if video:  # keep ~1 s after the outcome so the tumble / landing is visible
        for k in range(int(1.0 / CTRL_DT)):
            world.set_drive(data, 0.0, 0.0)
            world.move_actors(spec, data, t + k * CTRL_DT)
            advance()
        _write(frames, video)
        renderer.close()
    edge = next((s for s in states if s["x"] > 0), states[-1])
    return {
        "spec": spec, "params": params, "states": states,
        "success": outcome == "success", "fell": outcome == "fell", "stalled": outcome == "stalled",
        "collided": outcome == "collided", "near_miss": outcome == "near_miss",
        "min_clearance": None if min_clear is None else round(min_clear, 3),
        "outcome": outcome or "timeout",
        "fail_t": states[-1]["t"] if outcome in ("fell", "stalled", "collided", "near_miss") else None,
        "duration": round(t, 3),
        "edge_speed": edge["speed"],                     # speed when crossing the curb edge / slope start
        "edge_heading_deg": edge["heading_err_deg"],     # 0 = straight off the curb
        "peak_impact": round(peak_impact, 2),            # m/s^2 above 1 g: landing shock to cargo
        "max_tilt_deg": max(max(abs(s["roll_deg"]), abs(s["pitch_deg"])) for s in states),
        "jerk": _jerk(states),                           # mean |change in commands| per second: stop-go / twitchy steering
        "video": str(video) if video else None,
    }


def _jerk(states: list[dict]) -> float:
    if len(states) < 3:
        return 0.0
    dv = [abs(b["cmd_v"] - a["cmd_v"]) + 0.3 * abs(b["cmd_w"] - a["cmd_w"]) for a, b in zip(states, states[1:])]
    return round(sum(dv) / (len(dv) * CTRL_DT), 3)


def summary(r: dict) -> dict:
    """JSON-safe episode outcome (no per-tick states)."""
    keys = ["outcome", "success", "fell", "stalled", "collided", "near_miss", "min_clearance", "fail_t", "duration", "edge_speed", "edge_heading_deg", "peak_impact", "max_tilt_deg", "jerk", "video"]
    return {"scenario": r["spec"].name, **{k: r[k] for k in keys}}


def _camera(model, view: str):
    if view == "front":
        return model.camera("front").id
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("robot").id
    cam.distance, cam.elevation, cam.azimuth = 2.6, -18.0, 215.0
    if view == "top":  # overhead: makes clearance to obstacles unambiguous for a reviewer
        cam.distance, cam.elevation, cam.azimuth = 3.2, -89.0, 90.0
    return cam


def _write(frames, path):
    import imageio.v3 as iio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.stack(frames), fps=30, codec="libx264", macro_block_size=8)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="render naive (falls) vs careful (safe) on the default curb")
    ap.add_argument("--out", default="runs/demo")
    a = ap.parse_args()
    spec = ScenarioSpec(name="demo")
    careful = PolicyParams(cruise_speed=1.4, slow_dist=1.0, slow_speed=1.0, descent_angle_deg=45, heading_gain=2.5, align_tol_deg=30)
    for label, p in [("naive", NAIVE), ("careful", careful)]:
        r = run(spec, p, video=f"{a.out}/{label}.mp4")
        print(label, summary(r))
