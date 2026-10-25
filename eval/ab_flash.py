"""A/B: the same strict failure confirmation with gemini-3.8-flash vs Argon, on the Mini-4K candidates the user reviewed.

Ground truth = user's review (eval/user_review_mini4k.json): 'ok' means Argon's verdict was right.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from curbsense.steps.s0_confirm import confirm

ROOT = Path(__file__).resolve().parents[1]
rev = json.loads((ROOT / "eval/user_review_mini4k.json").read_text())
res = {Path(r["clip"]).stem: r for r in json.loads((ROOT / "data/mini4k_clips/confirmed.json").read_text())}
# truth: Argon's label where the user said Argon was right
truth = {k: (res[k]["pred"] != "none") for k, v in rev.items() if v == "ok" and k in res}
# add Argon-rejected candidates as extra negatives (not human-verified; reported separately)
extra_neg = [k for k, r in res.items() if r["pred"] == "none" and k not in truth][:14]
for k in extra_neg:
    truth[k] = False


def run(k):
    clip = ROOT / res[k]["clip"]
    try:
        f = confirm(str(clip), "robot", str(clip.with_suffix(".imu.csv")), model="gemini-3.8-flash")
        return k, f.failure != "none", f.failure
    except Exception as e:  # noqa: BLE001
        return k, None, str(e)[:80]


with ThreadPoolExecutor(6) as ex:
    flash = list(ex.map(run, list(truth)))
rows = [(k, truth[k], fp, lab) for k, fp, lab in flash if fp is not None]


def score(pred_of):
    tp = sum(t and p for k, t, p in pred_of); fp = sum((not t) and p for k, t, p in pred_of)
    fn = sum(t and not p for k, t, p in pred_of); tn = sum((not t) and not p for k, t, p in pred_of)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "accuracy": round((tp + tn) / len(pred_of), 3)}


argon_rows = [(k, truth[k], res[k]["pred"] != "none") for k, _, _, _ in rows]
flash_rows = [(k, t, p) for k, t, p, _ in rows]
out = {"n": len(rows), "positives": sum(truth[k] for k, *_ in rows), "argon": score(argon_rows), "flash": score(flash_rows),
       "disagreements": [{"clip": k, "truth": t, "flash": lab, "argon": res[k]["pred"]} for k, t, p, lab in rows if p != (res[k]["pred"] != "none")]}
(ROOT / "eval/ab_flash_vs_argon.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: out[k] for k in ("n", "positives", "argon", "flash")}, indent=1))
for d in out["disagreements"]:
    print(d)
