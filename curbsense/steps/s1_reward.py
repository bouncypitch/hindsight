"""Step 1: Argon watches good and bad rides and writes the reward function, then we verify it.

Verification loop: the reward must score known-good sim episodes above known-bad ones on the
same curbs. If the code crashes or ranks them wrong, the evidence goes back to Argon (max 3 tries).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

from pydantic import BaseModel, Field

from curbsense import argon
from curbsense.scenario import ScenarioSpec, randomize
from curbsense.sim.reference import BEHAVIORS
from curbsense.sim.rollout import run

CONTRACT = Path(__file__).resolve().parents[1].joinpath("rewards", "v0_baseline.py").read_text().split('"""')[1]

SYSTEM = """You design reward functions for reinforcement learning of a sidewalk delivery robot
(4-wheel skid-steer, tall cargo bay, ~0.5 m long) in a MuJoCo simulation. The policy will be
optimized hard against your reward, so think about how it could be gamed and close those loopholes.
Write plain Python (math and builtins only, no imports besides math)."""

PROMPT = """The first {n_good} clip(s) are GOOD rides: this is how operators want the robot to drive.
The next {n_bad} clip(s) are BAD rides (real fleet failures from the robot's front camera, plus simulated ones).
The robot faces curbs to drive down, curbs to climb, and uphill slopes; the reward must work for all three.

Write `score(ep) -> float` that rewards what the good rides do and penalizes what the bad rides do.
Higher = better. It is evaluated once per episode. The episode dict follows this contract:
{contract}
Units: speeds m/s, angles degrees, peak_impact m/s^2 above 1 g (20 ms filtered IMU), positions m
(x = 0 is the curb edge, x > 1.2 means the robot made it onto the road).

Return the full module source in `code` (it must define `score`), plus your reasoning."""

RETRY = """Your reward failed verification:
{problem}

Scores your reward gave on reference sim episodes (same curbs for every behavior):
{table}
Fix the reward and return the complete new module."""


class RewardProposal(BaseModel):
    rationale: str = Field(description="what the good and bad clips show, and how the reward encodes it")
    encourages: list[str]
    penalizes: list[str]
    loopholes_closed: list[str] = Field(description="ways RL could game a naive reward that this one blocks")
    code: str


def load_score(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score


def reference_episodes(n: int = 6) -> dict[str, list[dict]]:
    """Episodes from known behaviors: curb drops (randomized) and curb climbs."""
    drops = randomize(ScenarioSpec(name="ref"), {"curb_height": (-0.03, 0.03), "approach_angle_deg": (-12, 12)}, n, seed=7)
    climbs = [ScenarioSpec(kind="climb", curb_height=0.06, approach_angle_deg=a, name=f"climb{a}") for a in (-8, 8, 10)]
    actors = [ScenarioSpec(kind="obstacle", obs_x=1.5, obs_y=y, start_dist=0, name=f"obs{y}") for y in (-0.2, 0.1)] + \
             [ScenarioSpec(kind="pedestrian", obs_x=2.0, ped_speed=v, ped_t0=t0, start_dist=0, name=f"ped{v}")
              for v, t0 in ((1.2, 0.3), (1.5, 0.0))]
    out = {}
    for name, (policy, _) in BEHAVIORS.items():
        out[name] = []
        group = climbs if name.startswith("climb") else actors if name in ("avoid", "actor_naive") else drops
        for s in group:
            r = run(s, policy)
            out[name].append({k: r[k] for k in ["outcome", "success", "fell", "stalled", "collided", "near_miss",
                                                "min_clearance", "fail_t", "duration", "edge_speed",
                                                "edge_heading_deg", "peak_impact", "max_tilt_deg", "jerk", "states"]})
    return out


def check(path: Path, refs: dict[str, list[dict]]) -> tuple[bool, str, dict]:
    """Pass = every SAFE episode outscores every episode where NAIVE flipped. Jumper ranking is a warning."""
    try:
        score = load_score(path)
        table = {name: [float(score(ep)) for ep in eps] for name, eps in refs.items()}
    except Exception as e:  # noqa: BLE001 - any failure goes back to Argon
        return False, f"the code raised {type(e).__name__}: {e}", {}
    bad = [v for vs in table.values() for v in vs if not math.isfinite(v)]
    if bad:
        return False, "the reward returned NaN/inf", table
    falls = [v for v, ep in zip(table["naive"], refs["naive"]) if ep["fell"]]
    if min(table["safe"]) <= max(falls):
        return False, ("some GOOD (safe descent) episodes scored at or below BAD (robot flipped) episodes: "
                       f"min good {min(table['safe']):.2f} vs max bad {max(falls):.2f}"), table
    climbs = [v for v, ep in zip(table["climb_ok"], refs["climb_ok"]) if ep["success"]]
    stalls = [v for v, ep in zip(table["climb_slow"], refs["climb_slow"]) if ep["stalled"]]
    if climbs and stalls and min(climbs) <= max(stalls):
        return False, ("some GOOD (made it up the curb) episodes scored at or below BAD (stalled at the curb) episodes: "
                       f"min good {min(climbs):.2f} vs max bad {max(stalls):.2f}"), table
    hits = [v for v, ep in zip(table["actor_naive"], refs["actor_naive"]) if ep["collided"] or ep["near_miss"]]
    safe_pass = [v for v, ep in zip(table["avoid"], refs["avoid"]) if ep["success"]]
    if hits and safe_pass and min(safe_pass) <= max(hits):
        return False, ("some GOOD (passed the obstacle/person safely) episodes scored at or below BAD (hit it or cut "
                       f"within personal space) episodes: min good {min(safe_pass):.2f} vs max bad {max(hits):.2f}"), table
    if max(table["jumper"]) > min(table["safe"]):
        return True, ("ok, but WARNING: launching off the curb at 2 m/s scores above a gentle descent, "
                      "so RL will likely learn to jump curbs"), table
    return True, "ok", table


def fmt(table: dict) -> str:
    return "\n".join(f"  {k:7s} ({BEHAVIORS[k][1]}): " + ", ".join(f"{v:.2f}" for v in vs) for k, vs in table.items())


def write_reward(good: list[str], bad: list[str], out_dir: Path, tries: int = 3, fps: float = 8) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = reference_episodes()
    parts = [argon.video(p, fps=fps) for p in good + bad]
    parts.append(argon.text(PROMPT.format(n_good=len(good), n_bad=len(bad), contract=CONTRACT)))
    history = []
    for i in range(1, tries + 1):
        prop: RewardProposal = argon.ask(parts, RewardProposal, system=SYSTEM, thinking="high", tag=f"reward_v{i}")
        path = out_dir / f"reward_v{i}.py"
        path.write_text(prop.code)
        ok, problem, table = check(path, refs)
        history.append({"version": i, "path": str(path), "ok": ok, "problem": problem, "table": table,
                        **prop.model_dump(exclude={"code"})})
        print(f"reward_v{i}: {'PASS' if ok else 'FAIL'}  {problem}\n{fmt(table) if table else ''}")
        if ok:
            break
        parts = parts + [argon.text(f"Previous attempt:\n```python\n{prop.code}\n```"),
                         argon.text(RETRY.format(problem=problem, table=fmt(table) if table else "(no scores)"))]
    (out_dir / "reward_history.json").write_text(json.dumps(history, indent=1))
    return history[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--good", nargs="+", required=True)
    ap.add_argument("--bad", nargs="+", required=True)
    ap.add_argument("--out", default="runs/reward")
    ap.add_argument("--check", help="only verify an existing reward module")
    a = ap.parse_args()
    if a.check:
        ok, problem, table = check(Path(a.check), reference_episodes())
        print("PASS" if ok else "FAIL", problem, "\n" + fmt(table))
    else:
        res = write_reward(a.good, a.bad, Path(a.out))
        print(json.dumps({k: res[k] for k in ["path", "ok", "problem"]}, indent=1))
