"""Step 4: RL in the digital twin. CMA-ES tunes the policy to maximize the reward over failure variants."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from curbsense.scenario import ScenarioSpec, randomize
from curbsense.sim.policy import HI, LO, NAIVE, PolicyParams
from curbsense.sim.rollout import run

DEFAULT_RANGES = {"curb_height": (-0.03, 0.03), "approach_angle_deg": (-15, 15),
                  "start_dist": (-0.5, 0.5), "friction": (-0.2, 0.2)}


def load_reward(path: str):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score


# worker globals so each process loads the reward once
_SCORE = None
_VARIANTS: list[ScenarioSpec] = []


def _init(reward_path, variants):
    global _SCORE, _VARIANTS
    _SCORE, _VARIANTS = load_reward(reward_path), variants


def _episode(p: PolicyParams, s: ScenarioSpec) -> dict:
    r = run(s, p)
    return {**{k: r[k] for k in ["outcome", "success", "fell", "stalled", "collided", "near_miss", "min_clearance",
                                  "fail_t", "duration", "edge_speed",
                                  "edge_heading_deg", "peak_impact", "max_tilt_deg", "jerk"]}, "states": r["states"]}


def _evaluate(x) -> tuple[float, float, dict]:
    p = PolicyParams.from_vector(LO + np.clip(x, 0, 1) * (HI - LO))
    eps = [_episode(p, s) for s in _VARIANTS]
    return float(np.mean([_SCORE(e) for e in eps])), float(np.mean([e["success"] for e in eps])), _stats(eps)


def _stats(eps) -> dict:
    return {"success_rate": float(np.mean([e["success"] for e in eps])),
            "fall_rate": float(np.mean([e["fell"] for e in eps])),
            "stall_rate": float(np.mean([e["stalled"] for e in eps])),
            "collision_rate": float(np.mean([e["collided"] or e["near_miss"] for e in eps])),
            "mean_edge_speed": float(np.mean([e["edge_speed"] for e in eps])),
            "mean_abs_edge_heading": float(np.mean([abs(e["edge_heading_deg"]) for e in eps])),
            "mean_peak_impact": float(np.mean([e["peak_impact"] for e in eps])),
            **({"min_obstacle_clearance_m": float(min(c)), "mean_obstacle_clearance_m": float(np.mean(c))}
               if (c := [e["min_clearance"] for e in eps if e.get("min_clearance") is not None]) else {})}


def evaluate(p: PolicyParams, variants: list[ScenarioSpec], reward_path: str) -> dict:
    score = load_reward(reward_path)
    eps = [_episode(p, s) for s in variants]
    return {"reward": float(np.mean([score(e) for e in eps])), **_stats(eps)}


def train(variants: list[ScenarioSpec], reward_path: str, gens: int = 20, popsize: int = 10,
          start: PolicyParams = NAIVE, seed: int = 0, log=print) -> tuple[PolicyParams, list[dict]]:
    import cma
    x0 = np.clip((start.vector() - LO) / (HI - LO), 0.0, 1.0)  # observed behavior may sit outside the search box
    es = cma.CMAEvolutionStrategy(x0, 0.3, {"bounds": [0, 1], "popsize": popsize, "seed": seed + 1, "verbose": -9})
    history, best = [], (-1e9, x0)
    with Pool(os.cpu_count(), initializer=_init, initargs=(reward_path, variants)) as pool:
        for g in range(gens):
            xs = es.ask()
            res = pool.map(_evaluate, xs)
            es.tell(xs, [-r[0] for r in res])
            i = int(np.argmax([r[0] for r in res]))
            if res[i][0] > best[0]:
                best = (res[i][0], xs[i])
            history.append({"gen": g, "best_reward": res[i][0], **res[i][2]})
            log(f"gen {g:2d}  reward {res[i][0]:7.2f}  success {res[i][2]['success_rate']:.0%}  "
                f"edge_speed {res[i][2]['mean_edge_speed']:.2f}  edge_heading {res[i][2]['mean_abs_edge_heading']:.0f}°  "
                f"impact {res[i][2]['mean_peak_impact']:.0f}")
    return PolicyParams.from_vector(LO + np.clip(best[1], 0, 1) * (HI - LO)), history


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="curbsense/rewards/v0_baseline.py")
    ap.add_argument("--variants", help="variants.json from step 3 (default: randomize the demo curb)")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--out", default="runs/train")
    a = ap.parse_args()
    if a.variants:
        variants = [ScenarioSpec.from_dict(d) for d in json.load(open(a.variants))]
    else:
        variants = randomize(ScenarioSpec(name="demo"), DEFAULT_RANGES, a.n)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    before = evaluate(NAIVE, variants, a.reward)
    print("before:", json.dumps(before))
    policy, hist = train(variants, a.reward, gens=a.gens)
    after = evaluate(policy, variants, a.reward)
    print("after: ", json.dumps(after))
    print("policy:", policy.to_json(), f"({time.time() - t0:.0f}s)")
    (out / "policy.json").write_text(policy.to_json())
    (out / "report.json").write_text(json.dumps({"reward": a.reward, "before": before, "after": after, "history": hist}, indent=1))
