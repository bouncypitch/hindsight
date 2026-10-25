"""Record the demo film (player.html) in Chrome at 1080p and mux it with the ElevenLabs narration.

uv run python app/video/render.py [--skip whatif]  ->  app/video/curbsense_demo.mp4
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
GAP = 0.9  # seconds of breathing room after each scene's narration (must match player.html)


def main(skip: list[str], script_name="script.json", timing_name="timing.json", out_name="curbsense_demo.mp4"):
    script = json.loads((HERE / script_name).read_text())
    timing = json.loads((HERE / timing_name).read_text())
    scenes = [s for s in script["scenes"] if s["id"] not in skip]
    total = sum(timing[s["id"]] + GAP for s in scenes)
    raw = HERE / "rec"
    raw.mkdir(exist_ok=True)
    for f in raw.glob("*.webm"):
        f.unlink()
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome", headless=True, args=["--autoplay-policy=no-user-gesture-required"])
        ctx = b.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(raw),
                            record_video_size={"width": 1920, "height": 1080})
        page = ctx.new_page()
        t_open = time.time()
        page.goto(f"http://127.0.0.1:8765/app/video/player.html?skip={json.dumps(skip)}&script={script_name}&timing={timing_name}")
        page.wait_for_function("window.__ready === true", timeout=60000)
        page.wait_for_timeout(2500)  # let videos buffer
        lead = time.time() - t_open
        page.evaluate("window.play()")
        page.wait_for_timeout(int(total * 1000) + 500)
        ctx.close()
        b.close()
    webm = next(raw.glob("*.webm"))
    # narration track: each scene's audio followed by GAP seconds of silence
    inputs, filt = [], ""
    for i, s in enumerate(scenes):
        inputs += ["-i", str(HERE / "audio" / f"{s['id']}.mp3")]
        filt += f"[{i}]apad=pad_dur={GAP}[a{i}];"
    filt += "".join(f"[a{i}]" for i in range(len(scenes))) + f"concat=n={len(scenes)}:v=0:a=1[aout]"
    narration = HERE / (Path(out_name).stem + "_narration.m4a")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *inputs, "-filter_complex", filt, "-map", "[aout]", "-c:a", "aac",
                    "-b:a", "192k", str(narration)], check=True)
    out = HERE / out_name
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{lead:.2f}", "-i", str(webm), "-i", str(narration),
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-r", "30", "-c:a", "copy", "-shortest", str(out)], check=True)
    print(out, f"{total:.0f}s, lead trimmed {lead:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--script", default="script.json")
    ap.add_argument("--timing", default="timing.json")
    ap.add_argument("--out", default="curbsense_demo.mp4")
    a = ap.parse_args()
    main(a.skip, a.script, a.timing, a.out)
