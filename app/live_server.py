"""Live demo backend: serves the repo and runs Argon steps on demand for the control-room page.

uv run uvicorn app.live_server:app --port 8780
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
app = FastAPI()
JOBS: dict[str, dict] = {}


def _run(job_id: str, kind: str, arg: str):
    job = JOBS[job_id]
    t0 = time.time()
    try:
        if kind == "confirm":
            from curbsense.steps.s0_confirm import confirm
            clip = ROOT / arg
            job["result"] = confirm(str(clip), "robot", str(clip.with_suffix(".imu.csv"))).model_dump()
        elif kind == "diagnose":
            from curbsense.steps.s2_detect import detect
            clip = ROOT / arg
            imu = clip.with_suffix(".imu.csv")
            job["result"] = detect(clip, imu if imu.exists() else None).model_dump()
        elif kind == "gate":
            from curbsense import argon
            from curbsense.fleet import GATE, GateDecision
            from curbsense.sim.policy import NAMES
            rep = json.loads((ROOT / "demo_assets/fleet/fleet_report.json").read_text())["history"][0]
            job["result"] = argon.ask(GATE.format(names=NAMES, master="(round-0 master)", master_eval=json.dumps(
                rep["updates"][0]["fleet_validation_if_applied"]), updates=json.dumps(rep["updates"], indent=1)),
                GateDecision, thinking="medium", tag="live_gate").model_dump()
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e)[:500]
    job["secs"] = round(time.time() - t0, 1)


@app.post("/api/run/{kind}")
def run(kind: str, arg: str = ""):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "kind": kind, "arg": arg, "status": "running", "started": time.time()}
    threading.Thread(target=_run, args=(job_id, kind, arg), daemon=True).start()
    return {"id": job_id}


@app.get("/api/job/{job_id}")
def job(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return JSONResponse({"error": "unknown job"}, 404)
    return {**j, "elapsed": round(time.time() - j["started"], 1)}


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "live.html")


app.mount("/", StaticFiles(directory=ROOT), name="repo")
