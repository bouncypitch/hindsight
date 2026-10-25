"""Run the retrain -> verify -> fix-the-reward loop inside a Google-hosted Antigravity sandbox (Managed Agents).

The agent (gemini-4-argon) gets our simulator, policy, CMA-ES trainer, the twin's failure variants and the current
reward as inline files. In the sandbox it installs MuJoCo, trains, inspects the learned behavior for reward hacking,
edits the reward if needed, retrains, and reports a JSON summary. Nothing leaves the sandbox except that report.

uv run python -m curbsense.managed demo_assets/141102_tip_over_443s
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from curbsense import argon

ROOT = Path(__file__).resolve().parents[1]
AGENT = "antigravity-preview-09-2026"
CODE = ["curbsense/__init__.py", "curbsense/scenario.py", "curbsense/sim/__init__.py", "curbsense/sim/world.py",
        "curbsense/sim/policy.py", "curbsense/sim/rollout.py", "curbsense/sim/reference.py",
        "curbsense/steps/__init__.py", "curbsense/steps/s4_train.py", "curbsense/rewards/__init__.py",
        "curbsense/rewards/v0_baseline.py"]

TASK = """You are in a Linux sandbox with a sidewalk-robot RL project in /workspace/curbsense_repo.
A real fleet robot failed (diagnosis in /workspace/case/diagnosis.json). We built a MuJoCo digital twin that
reproduces it and {n} randomized variants (/workspace/case/variants.json). The reward to optimize is
/workspace/case/reward.py (contract: score(ep) -> float, documented in curbsense/rewards/v0_baseline.py).

Do this autonomously, verifying each step with real program output:
1. cd /workspace/curbsense_repo && pip install mujoco numpy cma (headless: export MUJOCO_GL=egl or osmesa if a
   renderer is needed; training does not render).
2. Train: python -m curbsense.steps.s4_train --reward /workspace/case/reward.py --variants /workspace/case/variants.json
   --gens {gens} --out /workspace/out/round1   (prints before/after success, fall, stall, collision rates).
3. Inspect the learned policy (/workspace/out/round1/policy.json) and per-episode metrics yourself (write a small
   script with curbsense.sim.rollout.run over the variants). Look for reward hacking: launching off curbs at max
   speed, crawling/refusing to move, passing obstacles or people with tiny clearance, jerky driving.
4. If you find hacking or the failure is not fixed, write an improved reward to /workspace/out/reward_v2.py (same
   contract), retrain into /workspace/out/round2, and compare.
5. Finish with ONE fenced ```json block: {{"rounds": [{{"reward": ..., "before": {{...}}, "after": {{...}}}}],
   "hacking_found": bool, "hack_description": str, "final_policy": {{...}}, "final_reward_code": str,
   "ship": bool, "notes": str}}"""


def run(asset_dir: Path, gens: int = 12, log=print) -> dict:
    variants = (asset_dir / "03_twin" / "variants.json").read_text()
    rl = sorted(asset_dir.glob("05_rl_round*.json"))[-1]
    reward_path = Path(json.loads(rl.read_text())["reward"])
    sources = [{"type": "inline", "content": (ROOT / f).read_text(), "target": f"/workspace/curbsense_repo/{f}"} for f in CODE]
    sources += [{"type": "inline", "content": variants, "target": "/workspace/case/variants.json"},
                {"type": "inline", "content": (ROOT / reward_path if not reward_path.is_absolute() else reward_path).read_text(),
                 "target": "/workspace/case/reward.py"},
                {"type": "inline", "content": (asset_dir / "02_diagnosis.json").read_text(), "target": "/workspace/case/diagnosis.json"}]
    n = len(json.loads(variants))
    t0 = time.time()
    it = argon.client().interactions.create(
        agent=AGENT, agent_config={"type": "antigravity", "model": "gemini-4-argon"},
        input=TASK.format(n=n, gens=gens), environment={"type": "remote", "sources": sources}, background=True)
    log(f"managed agent started: interaction {it.id}")
    while getattr(it, "status", "completed") in ("in_progress", "queued"):
        time.sleep(10)
        it = argon.client().interactions.get(id=it.id)
    text = it.output_text or ""
    m = re.findall(r"```json\s*(.*?)```", text, re.S)
    report = json.loads(m[-1]) if m else {"raw": text[-4000:]}
    report.update({"interaction_id": it.id, "environment_id": getattr(it, "environment_id", None),
                   "status": getattr(it, "status", None), "minutes": round((time.time() - t0) / 60, 1)})
    (asset_dir / "08_managed_agent.json").write_text(json.dumps(report, indent=1))
    (asset_dir / "08_managed_agent_transcript.md").write_text(text)
    log(f"managed agent {report['status']} in {report['minutes']} min; hacking_found={report.get('hacking_found')} ship={report.get('ship')}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("--gens", type=int, default=12)
    a = ap.parse_args()
    run(Path(a.asset_dir).resolve(), a.gens)
