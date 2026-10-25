"""Where does the naive robot fail on climb / slope scenarios?"""
from curbsense.scenario import ScenarioSpec
from curbsense.sim.policy import PolicyParams
from curbsense.sim.rollout import run

for kind, key, vals in [("climb", "curb_height", [0.04, 0.07, 0.10, 0.13]), ("slope", "slope_deg", [8, 12, 16, 20, 25])]:
    for ang in [0, 30]:
        for v in [0.4, 1.1, 1.8]:
            row = []
            for val in vals:
                r = run(ScenarioSpec(kind=kind, approach_angle_deg=ang, **{key: val}), PolicyParams(cruise_speed=v, slow_speed=v))
                row.append(f"{val}:{r['outcome'][:5]}")
            print(f"{kind:5s} ang={ang:2d} v={v}  " + "  ".join(row))
