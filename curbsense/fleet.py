"""Federated fleet loop: robots learn locally from their own failures and send only a tiny policy update.

Each round: every robot runs RL on twins of ITS OWN failures starting from the master policy and uploads the
parameter delta (a few dozen bytes instead of gigabytes of video). Argon reviews every update against a shared
validation set from all cities and rejects harmful ones; the server averages the accepted updates (FedAvg).

uv run python -m curbsense.fleet --reward runs/reward_real/reward_v1.py --rounds 3
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from curbsense import argon
from curbsense.scenario import ScenarioSpec, randomize
from curbsense.sim.policy import HI, LO, NAIVE, NAMES, PolicyParams
from curbsense.steps.s4_train import evaluate, train

ROOT = Path(__file__).resolve().parents[1]
SPEED_ONLY = ROOT / "curbsense" / "rewards" / "v0_baseline.py"  # what the misconfigured robot trains on


@dataclass
class Robot:
    name: str
    city: str
    variants: list[ScenarioSpec]
    video_bytes: int  # what uploading its raw failure footage would cost
    reward_override: Path | None = None


def fleet(seed: int = 0) -> list[Robot]:
    def rnd(base, ranges, n, s):
        return randomize(base, ranges, n, seed=seed + s)
    return [
        Robot("robot-A", "Berkeley (tall curbs)",
              rnd(ScenarioSpec(kind="drop", curb_height=0.20, name="A"), {"curb_height": (-0.03, 0.03), "approach_angle_deg": (-15, 15)}, 10, 1),
              video_bytes=1_900_000_000),
        Robot("robot-B", "San Francisco (busy plaza)",
              rnd(ScenarioSpec(kind="pedestrian", obs_x=2.0, start_dist=0, name="B"), {"ped_speed": (-0.4, 0.4), "ped_t0": (-0.3, 0.8)}, 5, 2)
              + rnd(ScenarioSpec(kind="obstacle", obs_x=1.5, start_dist=0, name="Bo"), {"obs_y": (-0.4, 0.4), "obs_r": (-0.05, 0.15)}, 5, 3),
              video_bytes=2_400_000_000),
        Robot("robot-C", "Hayward (hills, curb climbs)",
              rnd(ScenarioSpec(kind="slope", slope_deg=24, observed_speed=0.4, name="C"), {"slope_deg": (-3, 2), "approach_angle_deg": (-10, 10)}, 5, 4)
              + rnd(ScenarioSpec(kind="climb", curb_height=0.07, name="Cc"), {"curb_height": (-0.02, 0.01), "approach_angle_deg": (-10, 10)}, 5, 5),
              video_bytes=1_600_000_000),
        Robot("robot-D", "Oakland (curbs)",  # misconfigured: trains on a speed-only reward; the gate is not told
              rnd(ScenarioSpec(kind="drop", curb_height=0.18, name="D"), {"curb_height": (-0.03, 0.03)}, 10, 6),
              video_bytes=1_200_000_000, reward_override=SPEED_ONLY),
    ]


class UpdateDecision(BaseModel):
    robot: str
    accept: bool
    reason: str


class GateDecision(BaseModel):
    decisions: list[UpdateDecision]
    summary: str = Field(description="one or two sentences for the fleet operator")


GATE = """You are the gatekeeper for a sidewalk-robot fleet's shared driving policy. Each robot trained locally on
twins of its own real failures and uploaded a policy update. Accept an update only if it improves its own city
WITHOUT making the robot less safe anywhere else (falls, collisions, near-misses with people, stalls). Reject
anything that trades safety for speed. Policy parameters: {names}.

Master policy now: {master}
Per-city validation results of the CURRENT master: {master_eval}

Updates:
{updates}"""


def vec(p: PolicyParams) -> np.ndarray:
    return (p.vector() - LO) / (HI - LO)


def from_vec(x) -> PolicyParams:
    return PolicyParams.from_vector(LO + np.clip(x, 0, 1) * (HI - LO))


def city_eval(p: PolicyParams, robots: list[Robot], reward: Path) -> dict:
    out = {}
    for r in robots:
        if r.reward_override:
            continue  # the misconfigured robot's own scenarios are covered by robot-A's curbs
        e = evaluate(p, r.variants[::2], str(reward))
        out[r.city] = {"success": round(e["success_rate"], 2), "fall": round(e["fall_rate"], 2),
                       "collision_or_near_miss": round(e["collision_rate"], 2), "stall": round(e["stall_rate"], 2)}
    return out


def run_fleet(reward: Path, rounds: int = 3, local_gens: int = 8, out: Path | None = None, log=print) -> dict:
    out = out or ROOT / "demo_assets" / "fleet"
    out.mkdir(parents=True, exist_ok=True)
    robots = fleet()
    master = NAIVE
    history = []
    log(f"round 0 master: {city_eval(master, robots, reward)}")
    for rnd in range(1, rounds + 1):
        master_eval = city_eval(master, robots, reward)
        updates, deltas = [], {}
        for r in robots:
            local, _ = train(r.variants, str(r.reward_override or reward), gens=local_gens, start=master, seed=rnd,
                             log=lambda *a: None)
            delta = vec(local) - vec(master)
            deltas[r.name] = (delta, len(r.variants))
            payload = len(json.dumps({"robot": r.name, "delta": [round(float(d), 4) for d in delta]}).encode())
            cand = from_vec(vec(master) + delta)
            updates.append({"robot": r.name, "city": r.city, "upload_bytes": payload, "video_bytes_avoided": r.video_bytes,
                            "delta": {n: round(float(d), 3) for n, d in zip(NAMES, delta) if abs(d) > 0.01},
                            "local_before": evaluate(master, r.variants, str(r.reward_override or reward))["success_rate"],
                            "local_after": evaluate(cand, r.variants, str(r.reward_override or reward))["success_rate"],
                            "fleet_validation_if_applied": city_eval(cand, robots, reward)})
            log(f"  {r.name}: local success {updates[-1]['local_before']:.0%} -> {updates[-1]['local_after']:.0%}, "
                f"upload {payload} B (vs {r.video_bytes / 1e9:.1f} GB of video)")
        gate: GateDecision = argon.ask(GATE.format(names=NAMES, master=master.to_json(), master_eval=json.dumps(master_eval),
                                                   updates=json.dumps(updates, indent=1)),
                                       GateDecision, thinking="medium", tag=f"fleet_gate_{rnd}")
        accepted = [d.robot for d in gate.decisions if d.accept]
        for d in gate.decisions:
            log(f"  Argon {'ACCEPTS' if d.accept else 'REJECTS'} {d.robot}: {d.reason}")
        if accepted:
            w = np.array([deltas[a][1] for a in accepted], float)
            avg = sum(deltas[a][0] * wi for a, wi in zip(accepted, w)) / w.sum()
            master = from_vec(vec(master) + avg)
        new_eval = city_eval(master, robots, reward)
        log(f"round {rnd} master: {new_eval}")
        history.append({"round": rnd, "updates": updates, "gate": gate.model_dump(), "accepted": accepted,
                        "master": json.loads(master.to_json()), "master_eval": new_eval})
        (out / "fleet_report.json").write_text(json.dumps({"reward": str(reward), "history": history}, indent=1))
    (out / "master_policy.json").write_text(master.to_json())
    return {"master": master, "history": history}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", default="curbsense/rewards/ref_safe.py")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gens", type=int, default=8)
    a = ap.parse_args()
    run_fleet(ROOT / a.reward, a.rounds, a.gens)
