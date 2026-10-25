"""Sweep curb height / angle / speed (straight driving, fixed heading) to map where the robot flips."""
from curbsense.scenario import ScenarioSpec
from curbsense.sim.policy import PolicyParams
from curbsense.sim.rollout import run

for h in [0.15, 0.20, 0.23]:
    for ang in [0, 10, 25, 45]:
        row = []
        for v in [0.6, 1.0, 1.4, 1.8]:
            r = run(ScenarioSpec(curb_height=h, approach_angle_deg=ang), PolicyParams(cruise_speed=v, slow_speed=v))
            mx = max(max(abs(s["roll_deg"]), abs(s["pitch_deg"])) for s in r["states"])
            row.append(f"v{v}:{'OK ' if r['success'] else ('FELL' if r['fell'] else 'STK ')}{mx:4.0f}")
        print(f"h={h} ang={ang:2d}  " + "  ".join(row))
