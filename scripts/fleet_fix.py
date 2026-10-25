"""Regression caught -> fixed: San Francisco's rejected update is retrained with Berkeley's scenarios included,
then resubmitted to the Argon fleet gate."""
import json
from pathlib import Path

from curbsense import argon
from curbsense.fleet import GATE, GateDecision, city_eval, fleet, from_vec, vec
from curbsense.sim.policy import NAMES, PolicyParams
from curbsense.steps.s4_train import evaluate, train

ROOT = Path(__file__).resolve().parents[1]


def main():
    rep = json.loads((ROOT / "demo_assets/fleet/fleet_report.json").read_text())
    reward = ROOT / rep["reward"] if not Path(rep["reward"]).is_absolute() else Path(rep["reward"])
    master = PolicyParams(**rep["history"][0]["master"])  # after round 1: Berkeley's fix accepted
    robots = {r.name: r for r in fleet()}
    A, B = robots["robot-A"], robots["robot-B"]
    before = city_eval(master, list(robots.values()), reward)
    # first attempt: San Francisco alone (this is what Argon rejected)
    solo, _ = train(B.variants, str(reward), gens=8, start=master, seed=7, log=lambda *a: None)
    solo_eval = city_eval(solo, list(robots.values()), reward)
    # fix: retrain on San Francisco + Berkeley scenarios so the update cannot break Berkeley
    joint, _ = train(B.variants + A.variants, str(reward), gens=10, start=master, seed=7, log=lambda *a: None)
    joint_eval = city_eval(joint, list(robots.values()), reward)
    updates = [{"robot": "robot-B (first try, San Francisco data only)", "delta": {n: round(float(d), 3) for n, d in zip(NAMES, vec(solo) - vec(master)) if abs(d) > 0.01},
                "fleet_validation_if_applied": solo_eval},
               {"robot": "robot-B (retrained with Berkeley's scenarios)", "delta": {n: round(float(d), 3) for n, d in zip(NAMES, vec(joint) - vec(master)) if abs(d) > 0.01},
                "fleet_validation_if_applied": joint_eval}]
    gate = argon.ask(GATE.format(names=NAMES, master=master.to_json(), master_eval=json.dumps(before), updates=json.dumps(updates, indent=1)),
                     GateDecision, thinking="medium", tag="fleet_fix_gate")
    out = {"master_eval": before, "solo_eval": solo_eval, "joint_eval": joint_eval, "gate": gate.model_dump()}
    (ROOT / "demo_assets/fleet/regression_fix.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
