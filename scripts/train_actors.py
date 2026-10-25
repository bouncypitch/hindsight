"""Can RL learn to avoid obstacles and yield to people? (reference reward, sanity check)"""
import json
from curbsense.scenario import ScenarioSpec, randomize
from curbsense.sim.policy import NAIVE
from curbsense.steps.s4_train import train, evaluate



def main():
    vs = randomize(ScenarioSpec(kind="obstacle", obs_x=1.5, start_dist=0, name="obs"), {"obs_y": (-0.4, 0.4), "obs_r": (-0.05, 0.15)}, 12, seed=3) \
       + randomize(ScenarioSpec(kind="pedestrian", obs_x=2.0, start_dist=0, name="ped"), {"ped_speed": (-0.4, 0.4), "ped_t0": (-0.5, 0.8)}, 12, seed=4)
    R = "curbsense/rewards/ref_actors.py"
    print("before", json.dumps({k: round(v, 2) for k, v in evaluate(NAIVE, vs, R).items()}), flush=True)
    p, _ = train(vs, R, gens=15, log=lambda m: print(m, flush=True))
    print("after ", json.dumps({k: round(v, 2) for k, v in evaluate(p, vs, R).items()}))
    print(p.to_json())


if __name__ == "__main__":
    main()
