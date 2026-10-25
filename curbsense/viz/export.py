"""Export a MuJoCo episode as a 30 fps trajectory JSON that the Three.js street viewer replays."""
from __future__ import annotations

import json
import math
from pathlib import Path

import mujoco

from curbsense.scenario import ScenarioSpec
from curbsense.sim import world
from curbsense.sim.policy import PolicyParams, act
from curbsense.sim.rollout import CTRL_DT, MAX_T, Monitor, state

FPS = 30


def trajectory(spec: ScenarioSpec, params: PolicyParams, tail: float = 1.0) -> dict:
    model, data = world.build(spec)
    substeps = int(round(CTRL_DT / model.opt.timestep))
    every = int(round(1 / (FPS * model.opt.timestep)))
    wheels = [model.joint(f"j_{n}").qposadr[0] for n in ("fl", "fr", "rl", "rr")]
    frames, step, t, outcome, end_t = [], 0, 0.0, "timeout", None
    mon = Monitor(spec)

    def capture():
        ped = world.obstacle_pos(spec, data.time) if spec.kind == "pedestrian" else None
        frames.append({"t": round(data.time, 4), **({"ped": [round(ped[0], 3), round(ped[1], 3)]} if ped else {}),
                       "p": [round(float(v), 4) for v in data.qpos[:3]],
                       "q": [round(float(v), 5) for v in data.qpos[3:7]],  # w x y z
                       "w": [round(float(data.qpos[a]), 4) for a in wheels]})

    capture()
    while t < MAX_T + tail:
        if end_t is None:
            s = state(model, data, t, spec=spec)
            v, w = act(params, s["dist_to_edge"], math.radians(s["heading_err_deg"]), s["obs"])
        else:
            v, w = 0.0, 0.0
        world.set_drive(data, v, w)
        world.move_actors(spec, data, t)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
            step += 1
            if step % every == 0:
                capture()
        t += CTRL_DT
        if end_t is None:
            res = mon.update(state(model, data, t, (v, w), spec))
            if res:
                outcome, end_t = res, t
        elif t > end_t + tail:
            break
    return {
        "meta": {"fps": FPS, "outcome": outcome, "event_t": end_t, "scenario": json.loads(spec.to_json()),
                 "policy": json.loads(params.to_json()),
                 "robot": {"chassis_half": [0.26, 0.15, 0.06], "cargo_half": [0.22, 0.15, 0.12],
                           "cargo_offset": [-0.02, 0.0, world.CARGO_Z], "wheel_radius": world.WHEEL_R,
                           "wheel_half_width": 0.03, "wheel_offsets": {
                               "fl": [world.BASE / 2, world.TRACK / 2, -0.07], "fr": [world.BASE / 2, -world.TRACK / 2, -0.07],
                               "rl": [-world.BASE / 2, world.TRACK / 2, -0.07], "rr": [-world.BASE / 2, -world.TRACK / 2, -0.07]}},
                 "world": {"kind": spec.kind, "curb_x": 0.0, "curb_height": spec.curb_height,
                           "obstacle": ({"x": spec.obs_x, "y": spec.obs_y, "r": spec.obs_r} if spec.kind == "obstacle" else None),
                           "pedestrian": ({"radius": 0.22, "height": 1.7, "note": "frames[i].ped = [x, y] position"}
                                          if spec.kind == "pedestrian" else None),
                           "slope_deg": spec.slope_deg, "slope_len": world.SLOPE_LEN,
                           "note": ("z up; curb edge runs along y at x=0. drop: sidewalk x<0 raised by curb_height, road x>0 at z=0. "
                                    "climb: road x<0 at z=0, sidewalk x>0 raised. slope: flat x<0, ramp rising at slope_deg "
                                    "for slope_len meters from x=0, then a plateau")}},
        "frames": frames,
    }


if __name__ == "__main__":
    import argparse
    from curbsense.sim.reference import BEHAVIORS
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", default="naive", choices=list(BEHAVIORS))
    ap.add_argument("--out", default="runs/viz/naive.json")
    a = ap.parse_args()
    traj = trajectory(ScenarioSpec(name="demo"), BEHAVIORS[a.behavior][0])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(traj))
    print(a.out, traj["meta"]["outcome"], len(traj["frames"]), "frames")
