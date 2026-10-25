"""Step 3: build the digital twin of a failure, prove it reproduces the failure, then multiply it.

Reproduction test: replay the robot's original behavior in the twin. The twin is accepted only if
the robot fails the same way (same failure type, similar time after crossing the edge). If not,
Argon compares the sim render with the real clip and revises the scene (max 3 tries).
Then Argon chooses which parameters to randomize to get N failure variants for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, Field

from curbsense import argon
from curbsense.scenario import BOUNDS, ScenarioSpec, randomize
from curbsense.sim.policy import PolicyParams
from curbsense.sim.rollout import run, summary
from curbsense.steps.s2_detect import SYSTEM, FailureReport, SceneEstimate

TIME_TOL = 0.6  # s, failure time after edge crossing must match within this


class TwinRevision(BaseModel):
    diagnosis: str = Field(description="what differs between the sim replay and the real clip")
    scene: SceneEstimate


class VariationRange(BaseModel):
    param: str = Field(description=f"one of {list(BOUNDS)}")
    minus: float = Field(description="how far below the base value to sample")
    plus: float = Field(description="how far above the base value to sample")
    why: str


class VariationPlan(BaseModel):
    reasoning: str
    ranges: list[VariationRange]


def spec_from(scene: SceneEstimate, name: str) -> ScenarioSpec:
    # put the robot far enough back that it reaches the edge when it did in the clip
    t_edge = scene.t_edge_sec if scene.t_edge_sec and scene.t_edge_sec > 0.3 else 1.8
    start = max(0.8, min(4.0, scene.speed_mps * t_edge + 0.3))
    friction = 0.6 if any(w in scene.surface.lower() for w in ("wet", "gravel", "sand", "ice")) else 0.9
    kind = scene.scenario_kind if scene.scenario_kind != "flat" else "slope"
    if kind == "pedestrian" and scene.mover_speed_mps < 0.4:
        kind = "obstacle"  # a person/animal standing (nearly) still is an obstacle the robot drives into
    if kind in ("obstacle", "pedestrian"):
        # flat floor; the robot starts at the origin, the obstacle / crossing point is ahead of it
        return ScenarioSpec(kind=kind, start_dist=0.0, approach_angle_deg=0.0, friction=friction,
                            observed_speed=max(scene.speed_mps, 0.2), obs_x=max(0.6, scene.obstacle_distance_m),
                            obs_y=scene.obstacle_lateral_m, obs_r=max(0.03, scene.obstacle_radius_m),
                            ped_speed=max(0.3, scene.mover_speed_mps or 1.0),
                            ped_start_y=-2.5, ped_t0=0.0, name=name)
    return ScenarioSpec(kind=kind, curb_height=scene.curb_height_m, slope_deg=scene.slope_deg,
                        approach_angle_deg=scene.approach_angle_deg, start_dist=start, friction=friction,
                        observed_speed=max(scene.speed_mps, 0.2), name=name)


def observed_policy(spec: ScenarioSpec) -> PolicyParams:
    """What the robot was doing in the clip: driving straight at the observed speed."""
    return PolicyParams(cruise_speed=spec.observed_speed, slow_speed=spec.observed_speed)


def failure_type(r: dict) -> str:
    if r.get("collided") or r.get("near_miss"):
        return "collision"
    if r["stalled"]:
        return "stuck"
    if not r["fell"]:
        return "none"
    last = r["states"][-1]
    return "rollover" if abs(last["roll_deg"]) > abs(last["pitch_deg"]) else "nose_over"


# failure types Argon may report -> what the simulator can show
SIM_TYPE = {"nose_over": "nose_over", "rollover": "rollover", "stuck": "stuck", "high_center": "stuck",
            "wheel_trap": "stuck", "collision": "collision", "other": "stuck"}


def after_edge(r: dict, t: float | None) -> float | None:
    edge = next((s["t"] for s in r["states"] if s["x"] > 0), None)
    return None if t is None or edge is None else t - edge


def reproduces(r: dict, rep: FailureReport) -> tuple[bool, str]:
    got = failure_type(r)
    want = SIM_TYPE.get(rep.failure_type, "stuck")
    if rep.scene.scenario_kind in ("obstacle", "pedestrian"):
        want = "collision"  # e.g. "tipped over after clipping a stool leg": the twin must reproduce the contact
    if got != want:
        return False, f"twin outcome '{got}' but the clip shows '{rep.failure_type}'"
    if got == "collision":
        return True, f"twin reproduces the collision (min clearance {r.get('min_clearance')} m)"
    if got == "stuck":  # stall timing depends on the stall window, only the outcome type must match
        return True, f"twin reproduces the stall (robot stuck at x={r['states'][-1]['x']:.2f} m)"
    sim_dt = after_edge(r, r["fail_t"])
    real_dt = (rep.t_sec - rep.scene.t_edge_sec) if rep.t_sec and rep.scene.t_edge_sec else None
    if sim_dt is not None and real_dt is not None and abs(sim_dt - real_dt) > TIME_TOL:
        return False, f"twin fails {sim_dt:.2f}s after the edge, clip fails {real_dt:.2f}s after the edge"
    return True, f"twin reproduces a {got} ({sim_dt if sim_dt is not None else float('nan'):.2f}s after the edge)"


def build_twin(clip: str | Path, rep: FailureReport, out: Path, imu: str | Path | None = None,
               tries: int = 3, log=print) -> tuple[ScenarioSpec, list[dict]]:
    out.mkdir(parents=True, exist_ok=True)
    scene, history = rep.scene, []
    spec = None
    for i in range(1, tries + 1):
        spec = spec_from(scene, f"twin_v{i}")
        vid = out / f"twin_v{i}.mp4"
        r = run(spec, observed_policy(spec), video=vid)
        ok, why = reproduces(r, rep)
        history.append({"try": i, "spec": json.loads(spec.to_json()), "ok": ok, "check": why, **summary(r)})
        log(f"twin_v{i}: {'REPRODUCED' if ok else 'MISMATCH'}  {why}  (h={spec.curb_height:.2f} "
            f"angle={spec.approach_angle_deg:.0f} v={spec.observed_speed:.2f})")
        if ok or i == tries:
            break
        parts = [argon.text("REAL CLIP:"), argon.video(clip, fps=8)]
        if imu:
            parts.append(argon.text("Real IMU log:\n" + Path(imu).read_text()[:40000]))
        parts += [argon.text("SIMULATED TWIN (replaying the robot's original behavior):"), argon.video(vid, fps=8),
                  argon.text(f"Your earlier analysis:\n{rep.model_dump_json(indent=1)}\n\n"
                             f"Twin parameters used: {spec.to_json()}\nSim outcome: {json.dumps(summary(r))}\n"
                             f"Reproduction check failed: {why}.\n"
                             "The real clip may be from the robot's own front camera while the twin is a chase camera; "
                             "compare what happens (curb/slope geometry relative to the wheels, approach angle, speed, "
                             "how the body pitches/rolls or stalls) and return a corrected scene estimate so the twin "
                             "fails the same way the real robot did.")]
        rev: TwinRevision = argon.ask(parts, TwinRevision, system=SYSTEM, thinking="high", tag=f"twin_revise_{i}")
        log(f"  Argon: {rev.diagnosis}")
        scene = rev.scene
    (out / "twin.json").write_text(json.dumps({"spec": json.loads(spec.to_json()), "history": history}, indent=1))
    return spec, history


def plan_variants(spec: ScenarioSpec, rep: FailureReport, n: int, out: Path) -> list[ScenarioSpec]:
    prompt = (f"A robot failure was reproduced in simulation with these parameters:\n{spec.to_json()}\n"
              f"Failure analysis:\n{rep.model_dump_json(indent=1)}\n\n"
              f"We will train the driving policy on {n} randomized variants of this scene so it learns to "
              "handle this failure robustly, not just this exact curb. Choose which parameters to randomize "
              f"and how far. Allowed params and hard bounds: {json.dumps(BOUNDS)}. Center the spread on "
              "the uncertainty in your estimates (see `unknowns`) and on realistic neighbors of this situation; "
              "keep most variants dangerous for the current policy.")
    plan: VariationPlan = argon.ask(prompt, VariationPlan, system=SYSTEM, thinking="medium", tag="variants")
    ranges = {r.param: (-abs(r.minus), abs(r.plus)) for r in plan.ranges if r.param in BOUNDS}
    variants = randomize(spec, ranges, n, seed=1)
    (out / "variants.json").write_text(json.dumps([json.loads(v.to_json()) for v in variants], indent=1))
    (out / "variation_plan.json").write_text(plan.model_dump_json(indent=1))
    return variants


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("report", help="FailureReport json from step 2")
    ap.add_argument("--imu")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="runs/twin")
    a = ap.parse_args()
    rep = FailureReport.model_validate_json(Path(a.report).read_text())
    out = Path(a.out)
    spec, _ = build_twin(a.clip, rep, out, a.imu)
    vs = plan_variants(spec, rep, a.n, out)
    print(f"{len(vs)} variants -> {out / 'variants.json'}")
