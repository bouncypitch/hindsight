"""Photoreal pass: Gemini Omni Flash restyles a twin render to look like the real footage (reference frame), keeping
geometry and motion. Omni edits uploaded videos of <= 10 s.

uv run python -m curbsense.viz.omni TWIN.mp4 REAL_FRAME.png OUT.mp4 [--start 0]
"""
from __future__ import annotations

import argparse
import base64
import subprocess
import time
from pathlib import Path

from curbsense import argon

PROMPT = ("Re-render this video as real camera footage from a small sidewalk delivery robot's low front camera, matching "
          "the look of the reference image exactly: real materials, real night street lighting and glare, sensor noise, "
          "lens character and color grade. Keep EVERYTHING else identical: the camera path and timing, the layout, every "
          "object, vehicle, sign and building in the same place and moving the same way. Do not add or remove anything.")


def upload(path: Path):
    f = argon.client().files.upload(file=str(path))
    while getattr(f.state, "name", f.state) == "PROCESSING":
        time.sleep(5)
        f = argon.client().files.get(name=f.name)
    return f


def restyle(twin: Path, ref: Path, out: Path, start: float = 0.0, dur: float = 9.5, prompt: str = PROMPT) -> Path:
    seg = out.with_name(out.stem + "_input.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(twin), "-t", str(dur),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(seg)], check=True)
    v, r = upload(seg), upload(ref)
    t0 = time.time()
    it = argon.client().interactions.create(
        model="gemini-omni-1.1-flash", background=True,
        input=[{"type": "video", "uri": v.uri}, {"type": "image", "uri": r.uri, "mime_type": r.mime_type},
               {"type": "text", "text": prompt}])
    while getattr(it, "status", "completed") in ("in_progress", "queued"):
        time.sleep(10)
        it = argon.client().interactions.get(id=it.id)
    out.write_bytes(base64.b64decode(it.output_video.data))
    print(f"omni {it.status} in {time.time() - t0:.0f}s -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("twin")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0)
    a = ap.parse_args()
    restyle(Path(a.twin), Path(a.ref), Path(a.out), a.start)
