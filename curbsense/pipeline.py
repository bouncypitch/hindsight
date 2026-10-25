"""End-to-end: one real fleet failure clip -> diagnosis -> twin -> variants -> RL -> verification, saving every
intermediate artifact (videos + JSON) under demo_assets/<name>/ for the demo site and the narrated video.

uv run python -m curbsense.pipeline data/real_clips/ep00057_stall_12s.mp4 --reward runs/reward/reward_v2.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from curbsense import argon
from curbsense.scenario import ScenarioSpec
from curbsense.sim.rollout import run, summary
from curbsense.steps.s2_detect import FailureReport, detect
from curbsense.steps.s3_twin import build_twin, observed_policy, plan_variants
from curbsense.steps.s4_train import evaluate, train
from curbsense.steps.s5_verify import verify
from curbsense.viz.export import trajectory

ROOT = Path(__file__).resolve().parents[1]


def log_to(path: Path):
    def log(msg: str):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with path.open("a") as f:
            f.write(line + "\n")
    return log


def variant_grid(variants: list[ScenarioSpec], policy, out: Path, n: int = 12) -> Path:
    """Tile n variant replays (old policy) into one mosaic video: the 'many copies of the failure' shot."""
    tiles = out.parent / "variant_tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    vids = []
    for i, v in enumerate(variants[:n]):
        p = tiles / f"v{i:02d}.mp4"
        run(v, policy, video=p, max_t=8)
        vids.append(p)
    cols = 4
    inputs = sum([["-i", str(v)] for v in vids], [])
    scaled = "".join(f"[{i}]scale=320:180,tpad=stop_mode=clone:stop_duration=6[s{i}];" for i in range(len(vids)))
    layout = "|".join(f"{(i % cols) * 320}_{(i // cols) * 180}" for i in range(len(vids)))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                    scaled + "".join(f"[s{i}]" for i in range(len(vids))) + f"xstack=inputs={len(vids)}:layout={layout}:shortest=1",
                    "-t", "8", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def checked_reward(code: str, path: Path, log, tries: int = 2) -> Path | None:
    """Every reward Argon rewrites must compile and rank good > bad on reference episodes (same check as step 1)."""
    from curbsense.steps.s1_reward import check, fmt, reference_episodes
    refs = reference_episodes()
    for k in range(tries + 1):
        path.write_text(code)
        ok, problem, table = check(path, refs)
        log(f"  reward check {path.name}: {'PASS' if ok else 'FAIL'} {problem}")
        if ok:
            return path
        if k == tries:
            return None
        code = argon.ask(f"This reward module failed verification:\n{problem}\n{fmt(table) if table else ''}\n\n"
                         f"```python\n{code}\n```\nReturn ONLY the corrected complete module in one ```python block.",
                         thinking="medium", tag="reward_fix")
        if "```" in code:
            code = code.split("```python")[-1].split("```")[0] if "```python" in code else code.split("```")[1]
    return None


def main(clip: Path, reward: Path, name: str | None = None, n_variants: int = 50, gens: int = 20,
         refine: bool = True) -> Path:
    name = name or clip.stem
    out = ROOT / "demo_assets" / name
    out.mkdir(parents=True, exist_ok=True)
    log = log_to(out / "pipeline.log")
    imu = clip.with_suffix(".imu.csv")
    imu = imu if imu.exists() else None
    manifest: dict = {"name": name, "steps": []}

    def step(title: str, **files):
        manifest["steps"].append({"title": title, **{k: str(Path(v).relative_to(out)) if isinstance(v, Path) else v
                                                      for k, v in files.items()}})
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    shutil.copy(clip, out / "01_real_failure.mp4")
    step("Real fleet clip", video=out / "01_real_failure.mp4")

    if (out / "02_diagnosis.json").exists():  # resume
        rep = FailureReport.model_validate_json((out / "02_diagnosis.json").read_text())
        manifest["steps"] = [s_ for s_ in manifest["steps"]][:1]
    else:
        log("Argon: diagnosing the real clip (video + sensors)")
        rep = detect(clip, imu)
        (out / "02_diagnosis.json").write_text(rep.model_dump_json(indent=1))
    log(f"  {rep.failure_type} at t={rep.t_sec}s, confidence {rep.confidence}: {rep.cause}")
    step("Argon diagnosis", json=out / "02_diagnosis.json")

    twin_json = out / "03_twin" / "twin.json"
    if twin_json.exists():  # resume
        spec = ScenarioSpec.from_dict(json.loads(twin_json.read_text())["spec"])
    else:
        log("Argon: building a twin that reproduces the failure")
        spec, hist = build_twin(clip, rep, out / "03_twin", imu, log=log)
    old = observed_policy(spec)
    view = "top" if spec.kind in ("obstacle", "pedestrian") else "chase"
    run(spec, old, video=out / "03_twin_old_policy.mp4", view=view)
    step("Twin reproduces the failure", video=out / "03_twin_old_policy.mp4", json=out / "03_twin" / "twin.json")

    vjson = out / "03_twin" / "variants.json"
    if vjson.exists():  # resume
        variants = [ScenarioSpec.from_dict(d) for d in json.loads(vjson.read_text())]
    else:
        log(f"Argon: planning {n_variants} variants")
        variants = plan_variants(spec, rep, n_variants, out / "03_twin")
    if not (out / "04_variants_grid.mp4").exists():
        variant_grid(variants, old, out / "04_variants_grid.mp4")
    step("Many copies of the failure", video=out / "04_variants_grid.mp4", json=out / "03_twin" / "variation_plan.json")

    rounds = 2 if refine else 1
    reward_path = reward
    for r in range(1, rounds + 1):
        log(f"RL round {r}: CMA-ES on {len(variants)} variants with {reward_path.name}")
        before = evaluate(old, variants, str(reward_path))
        policy, history = train(variants, str(reward_path), gens=gens, start=old, log=log)
        after = evaluate(policy, variants, str(reward_path))
        log(f"  success {before['success_rate']:.0%} -> {after['success_rate']:.0%}")
        (out / f"05_rl_round{r}.json").write_text(json.dumps({"reward": str(reward_path), "before": before, "after": after,
                                                               "policy": json.loads(policy.to_json()), "history": history}, indent=1))
        new_vid = out / f"05_twin_new_policy_round{r}.mp4"
        new_nom = summary(run(spec, policy, video=new_vid, view=view))
        old_nom = summary(run(spec, old))
        step(f"RL round {r}: before vs after", video=new_vid, json=out / f"05_rl_round{r}.json")

        log("Argon: verifying the new policy against the real failure")
        keys = ("outcome", "min_clearance", "peak_impact", "duration")
        v = verify(clip, out / "03_twin_old_policy.mp4", new_vid, before, after, len(variants), reward_path, imu,
                   {k: old_nom[k] for k in keys}, {k: new_nom[k] for k in keys})
        (out / f"06_verdict_round{r}.json").write_text(v.model_dump_json(indent=1))
        log(f"  fixes={v.fixes_real_failure} hacking={v.reward_hacking} ship={v.ship} {v.hack_description}")
        step(f"Argon verdict round {r}", json=out / f"06_verdict_round{r}.json")
        if v.ship or not v.new_reward_code.strip() or r == rounds:
            break
        fixed = checked_reward(v.new_reward_code, out / f"reward_round{r + 1}.py", log)
        if fixed is None:
            log("  Argon's rewritten reward failed verification twice; keeping the round-1 policy")
            break
        reward_path = fixed
        log(f"  Argon rewrote the reward -> {reward_path.name} (verified); retraining")

    (out / "policy.json").write_text(policy.to_json())
    for tag, p in [("old", old), ("new", policy)]:
        (out / f"07_traj_{tag}.json").write_text(json.dumps(trajectory(spec, p)))
    step("Trajectories for the photoreal viewer", old=out / "07_traj_old.json", new=out / "07_traj_new.json")
    log("done")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--reward", default="curbsense/rewards/ref_safe.py")
    ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--no-refine", action="store_true")
    a = ap.parse_args()
    print(main(Path(a.clip), Path(a.reward), a.name, a.n, a.gens, not a.no_refine))
