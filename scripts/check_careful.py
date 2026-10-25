"""Naive vs hand-tuned diagonal-descent policy over a grid of curb variants."""
import sys
from curbsense.scenario import ScenarioSpec
from curbsense.sim.policy import NAIVE, PolicyParams
from curbsense.sim.rollout import run

careful = PolicyParams(cruise_speed=1.1, slow_dist=1.2, slow_speed=0.8, descent_angle_deg=45, heading_gain=3.0, align_tol_deg=20)
for name, p in [("naive", NAIVE), ("diagonal", careful)]:
    res = []
    for h in [0.18, 0.20, 0.23]:
        for ang in [-15, 0, 8, 20]:
            r = run(ScenarioSpec(curb_height=h, approach_angle_deg=ang), p)
            res.append("OK" if r["success"] else ("F" if r["fell"] else "S"))
    print(f"{name:9s} {res.count('OK')}/{len(res)}  {' '.join(res)}")
