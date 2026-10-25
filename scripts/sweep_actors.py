"""Naive vs avoiding/yielding robot on obstacle and pedestrian scenarios."""
from curbsense.scenario import ScenarioSpec
from curbsense.sim.policy import NAIVE, PolicyParams
from curbsense.sim.rollout import run

smart = PolicyParams(cruise_speed=1.1, yield_dist=2.0, avoid_dist=1.8, avoid_gain=2.5)
for name, p in [("naive", NAIVE), ("smart", smart)]:
    row = []
    for y in [-0.2, 0.0, 0.3]:
        r = run(ScenarioSpec(kind="obstacle", obs_x=1.5, obs_y=y, start_dist=0), p)
        row.append(f"obs y={y}:{r['outcome']}({r['min_clearance']})")
    for ps, t0 in [(1.2, 0.3), (1.0, 0.8), (1.5, 0.0)]:
        r = run(ScenarioSpec(kind="pedestrian", obs_x=2.0, ped_speed=ps, ped_t0=t0, start_dist=0), p)
        row.append(f"ped v={ps} t0={t0}:{r['outcome']}({r['min_clearance']})")
    print(name, " | ".join(row))
