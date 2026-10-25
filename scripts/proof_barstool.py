"""Is the barstool twin solvable with Argon's obstacle reward under the stricter (graze) collision check?"""
import json
from pathlib import Path
from curbsense.scenario import ScenarioSpec
from curbsense.sim.rollout import run, summary
from curbsense.steps.s3_twin import observed_policy
from curbsense.steps.s4_train import train, evaluate

D = Path("demo_assets/141102_tip_over_443s")


def main():
    spec = ScenarioSpec.from_dict(json.loads((D / "03_twin/twin.json").read_text())["spec"])
    variants = [ScenarioSpec.from_dict(d) for d in json.loads((D / "03_twin/variants.json").read_text())]
    R = str(sorted(Path("runs/reward_obstacle").glob("reward_v*.py"))[-1])
    old = observed_policy(spec)
    print("reward", R)
    print("old on nominal twin:", summary(run(spec, old)))
    print("before:", {k: round(v, 2) for k, v in evaluate(old, variants, R).items()})
    p, _ = train(variants, R, gens=25, start=old, log=lambda m: None)
    print("after: ", {k: round(v, 2) for k, v in evaluate(p, variants, R).items()})
    print("new on nominal twin:", summary(run(spec, p)))
    print("policy", p.to_json())
    (D / "proof_policy.json").write_text(p.to_json())


if __name__ == "__main__":
    main()
