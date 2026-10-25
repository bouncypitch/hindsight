"""Headless Chrome helpers: screenshot the viewer at key moments, or capture every frame to mp4."""
from __future__ import annotations

import functools
import http.server
import json
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def serve(root: Path = ROOT):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()


@contextmanager
def page(viewer: Path, traj: Path, cam: str = "chase", size=(1280, 720), scene: Path | None = None, extra: str = ""):
    from playwright.sync_api import sync_playwright
    errors: list[str] = []
    with serve() as base, sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True,
                                     args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        pg = browser.new_page(viewport={"width": size[0], "height": size[1]})
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errors.append(str(e)))
        url = f"{base}/{viewer.relative_to(ROOT)}?traj=/{traj.relative_to(ROOT)}&cam={cam}&capture=1"
        if scene is not None:
            url += f"&scene=/{scene.relative_to(ROOT)}"
        url += extra
        pg.goto(url)
        try:
            pg.wait_for_function("window.__ready === true", timeout=60000)
        except Exception as e:  # noqa: BLE001
            errors.append(f"viewer never set window.__ready: {e}")
        yield pg, errors
        browser.close()


def screenshots(viewer: Path, traj: Path, out: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(traj.read_text())["meta"]
    ev = meta.get("event_t") or 2.0
    moments = [("chase, start", "chase", 0.3), ("chase, at the curb edge", "chase", max(0.3, ev - 0.6)),
               ("chase, outcome", "chase", ev + 0.4), ("side view, at the curb", "side", max(0.3, ev - 0.4)),
               ("street view", "street", ev)]
    shots, errors = [], []
    for cam in ("chase", "side", "street"):
        with page(viewer, traj, cam) as (pg, errs):
            for label, c, t in moments:
                if c != cam:
                    continue
                try:
                    pg.evaluate(f"window.seek({t})")
                    p = out / f"{label.replace(', ', '_').replace(' ', '_')}.png"
                    pg.screenshot(path=str(p))
                    shots.append((label, p))
                except Exception as e:  # noqa: BLE001
                    errs.append(f"seek/screenshot failed at t={t} cam={cam}: {e}")
            errors += errs
    return shots, errors


def shots_at(viewer: Path, traj: Path, times: list[float], out: Path, cam: str = "robot",
             scene: Path | None = None, size=(896, 512), extra: str = "") -> tuple[list[Path], list[str]]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    with page(viewer, traj, cam, size, scene, extra) as (pg, errors):
        for i, t in enumerate(times):
            pg.evaluate(f"window.seek({t})")
            p = out / f"twin_{i}.png"
            pg.screenshot(path=str(p))
            paths.append(p)
    return paths, errors


def capture(viewer: Path, traj: Path, mp4: Path, cam: str = "chase", fps: int = 30,
            scene: Path | None = None, size=(1280, 720), extra: str = "") -> Path:
    frames = json.loads(traj.read_text())["frames"]
    dur = frames[-1]["t"]
    with tempfile.TemporaryDirectory() as tmp, page(viewer, traj, cam, size, scene, extra) as (pg, errors):
        n = int(dur * fps)
        for i in range(n):
            pg.evaluate(f"window.seek({i / fps})")
            pg.screenshot(path=f"{tmp}/f{i:05d}.png")
        mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", f"{tmp}/f%05d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    if errors:
        print("console errors:", *errors[:10], sep="\n  ")
    return mp4


if __name__ == "__main__":
    import sys
    traj, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
    cam = sys.argv[3] if len(sys.argv) > 3 else "chase"
    print(capture(ROOT / "app" / "viewer" / "viewer.html", traj, out, cam))
