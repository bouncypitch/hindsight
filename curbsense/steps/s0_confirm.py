"""Strict failure confirmation with Argon: curb falls, tip-overs and collisions only.

Precision first. Argon may only report a failure it can SEE happen in the video, with timestamps and visual
evidence. Telemetry alone is never enough (FrodoBots logs have dropouts, saturated axes and video/log
misalignment), and video artifacts (cuts, freezes, frame skips) must be called out as such.

uv run python -m curbsense.steps.s0_confirm --eval eval/labels.json     # accuracy on the labeled set
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from curbsense import argon

KINDS = ["curb_fall", "tip_over", "collision_pedestrian", "collision_object", "collision_vehicle", "curb_failure",
         "other_failure", "none"]

SYSTEM = """You audit videos of small sidewalk delivery robots for real mobility failures. You are precise and
conservative: a false alarm costs more than a miss. You only report what you can actually see happen."""

PROMPT = """Watch this video{view_note}. Did the delivery robot have one of these failures?
- curb_fall: the robot drops off a curb / ledge / step / path edge in an uncontrolled way (wheel(s) go over the
  edge, body lurches or falls).
- tip_over: the robot ends up on its side, front, back or upside down.
- collision_pedestrian: the robot makes contact with a person (or a person's wheelchair/scooter/stroller).
- collision_object: the robot hits a static object (pole, bollard, wall, glass, fence, parked car, bin).
- collision_vehicle: a moving vehicle hits the robot or the robot drives into a moving vehicle.
- curb_failure: the robot gets stuck on or cannot climb a curb.
- other_failure: any other clear mobility failure you can see (e.g. wheel trapped, stuck under furniture, drove
  into water/hole, had to be picked up or rescued by a person after a mishap). Describe it in other_description.
- none: none of the above is visibly happening.

Rules:
- Report a failure ONLY if you can see it happen (or its unmistakable immediate result) in the frames.
- Bumps, speed bumps, normal curb ramps, braking, slowing, waiting for people, turning: NOT failures.
- Video artifacts are NOT failures: hard cuts to a different scene, frozen frames, frame skips, black frames,
  sudden jumps in viewpoint without continuous motion. If the evidence is an artifact, say so and answer none.
- Telemetry (if provided) can support what you see but can never establish a failure on its own; robot logs here
  are often glitchy or misaligned with the video.
Give the time of the failure and specific visual evidence."""


class Confirmation(BaseModel):
    failure: Literal["curb_fall", "tip_over", "collision_pedestrian", "collision_object", "collision_vehicle",
                     "curb_failure", "other_failure", "none"]
    other_description: str = Field(default="", description="only for other_failure")
    t_sec: float | None = Field(description="when it happens in the video")
    visual_evidence: list[str] = Field(description="what you see, with timestamps")
    artifacts_seen: list[str] = Field(description="video cuts / freezes / skips you noticed, with timestamps")
    confidence: float = Field(ge=0, le=1)


def confirm(source: str, view: str = "robot", imu: str | None = None, model: str | None = None) -> Confirmation:
    if source.startswith("http"):
        part = {"type": "video", "uri": source}
    else:
        part = argon.video(source, fps=5)
    note = (" (from the robot's own front camera, low resolution)" if view == "robot"
            else " (filmed by a bystander or news camera)")
    parts = [part]
    if imu and Path(imu).exists():
        parts.append(argon.text("Robot accelerometer log (g, 10 Hz; t in seconds from clip start):\n" + Path(imu).read_text()[:20000]))
    parts.append(argon.text(PROMPT.format(view_note=note)))
    return argon.ask(parts, Confirmation, system=SYSTEM, thinking="high", model=model, tag=f"confirm:{Path(source).name}")


def evaluate(labels_path: Path, out: Path, workers: int = 6, model: str | None = None) -> dict:
    labels = json.loads(labels_path.read_text())

    def one(l):
        imu = str(Path(l["source"]).with_suffix(".imu.csv")) if not l["source"].startswith("http") else None
        try:
            c = confirm(l["source"], l["view"], imu, model)
            return {**l, "pred": c.failure, "pred_detail": c.model_dump()}
        except Exception as e:  # noqa: BLE001
            return {**l, "pred": "error", "pred_detail": {"error": str(e)[:300]}}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, labels))
    scored = [r for r in res if r["pred"] != "error"]
    tp = sum(r["label"] != "none" and r["pred"] != "none" for r in scored)
    fp = sum(r["label"] == "none" and r["pred"] != "none" for r in scored)
    fn = sum(r["label"] != "none" and r["pred"] == "none" for r in scored)
    tn = sum(r["label"] == "none" and r["pred"] == "none" for r in scored)
    exact = sum(r["label"] != "none" and r["pred"] == r["label"] for r in scored)
    m = {"n": len(scored), "errors": len(res) - len(scored), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
         "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
         "type_correct_on_positives": f"{exact}/{tp + fn}"}
    out.write_text(json.dumps({"metrics": m, "results": res}, indent=1))
    return m


def confirm_candidates(cands_path: Path, workers: int = 6) -> list[dict]:
    """Argon verdict for every mined candidate; writes confirmed.json next to candidates.json."""
    root = Path(__file__).resolve().parents[2]
    cands = json.loads(cands_path.read_text())
    prev_path = cands_path.with_name("confirmed.json")
    prev = {r["clip"]: r for r in json.loads(prev_path.read_text())} if prev_path.exists() else {}

    def one(c):
        if c["clip"] in prev and prev[c["clip"]]["pred"] != "error":
            return {**c, "pred": prev[c["clip"]]["pred"], "pred_detail": prev[c["clip"]]["pred_detail"]}
        clip = root / c["clip"]
        try:
            v = confirm(str(clip), "robot", str(clip.with_suffix(".imu.csv")))
            return {**c, "pred": v.failure, "pred_detail": v.model_dump()}
        except Exception as e:  # noqa: BLE001
            return {**c, "pred": "error", "pred_detail": {"error": str(e)[:300]}}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, cands))
    cands_path.with_name("confirmed.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval/labels.json")
    ap.add_argument("--out", default="eval/results_argon.json")
    ap.add_argument("--model")
    ap.add_argument("--candidates", help="confirm mined candidates (e.g. data/mini4k_clips/candidates.json)")
    a = ap.parse_args()
    if a.candidates:
        for r in confirm_candidates(Path(a.candidates)):
            d = r["pred_detail"]
            print(f"{r['pred']:22s} conf={d.get('confidence', '-')} {Path(r['clip']).name:32s} sensor:{r['kind']:10s} "
                  f"{(d.get('other_description') or (d.get('visual_evidence') or [''])[0] or d.get('error', ''))[:150]}")
        raise SystemExit
    m = evaluate(Path(a.eval), Path(a.out), model=a.model)
    print(json.dumps(m, indent=1))
    for r in json.loads(Path(a.out).read_text())["results"]:
        mark = "OK " if (r["pred"] == "none") == (r["label"] == "none") else "XX "
        print(mark, f"{r['label']:22s} -> {r['pred']:22s} {r['id']}")
