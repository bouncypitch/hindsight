"""Argon builds a digital twin of a real Coco fleet clip: the robot drives its exact recorded path through a 3D
replica of the scene, seen from the same camera. Argon refines the twin by comparing real vs twin frames.

uv run python -m curbsense.viz.coco_twin data/coco_samples/09294dbb --rounds 3
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from curbsense import argon
from curbsense.viz.match_scene import pair
from curbsense.viz.shoot import ROOT, capture, shots_at

SYSTEM = """You are a senior graphics engineer who builds photorealistic real-time scenes in Three.js from reference
footage. You write complete, working single-file HTML apps. No placeholders, no TODOs."""

SPEC = """Build `twin.html`: a digital twin of the REAL robot camera clip shown in the reference frames. A Coco
Robotics sidewalk delivery robot recorded this clip from its front camera. The robot's exact recorded path is in
the trajectory JSON. Recreate the scene so that a render from the robot camera at the same moment is as close to
the real frame as possible, ideally indistinguishable.

## What to model (study the frames carefully)
Ground (sidewalk, curb ramp, tactile warning pad, crosswalk paint, asphalt, gutters, manholes), curbs and their
heights, street layout relative to the robot path, buildings and storefronts with lit signage, streetlights and
their glow, traffic signals, parked/moving vehicles (animate moving ones along plausible paths at the times they
appear), trees, poles, and the lighting: it is night - sodium/LED street lighting, light falloff, the robot's own
headlight pool on the ground, bloom on bright lights, dark sky. Match the camera look: wide-angle, slight noise,
exposure and color grade of the real footage.

## Coordinates and camera
World: meters, z up in the trajectory (convert to Three.js y-up: sim (x, y, z) -> three (x, z, -y)). The robot
starts at the origin facing +x. frames[i] = {{t, x, y, yaw}} (yaw in rad about z). Robot camera: at the robot's
position, height {height} m, looking along yaw, pitch adjustable (start at -2 deg), PerspectiveCamera vertical FOV
{vfov:.1f} deg, aspect {aspect:.4f} (pinhole, already undistorted). Also provide `cam=chase` (3 m behind/above the
robot) showing a simple delivery-robot model (white/grey box body ~0.7 x 0.5 x 0.6 m, 6 wheels, orange flag).
The reference frames below were taken at these clip times: {times}.

## Contract (other code depends on this; follow exactly)
- URL params: `traj` (URL of the trajectory JSON to fetch), `cam` = robot|chase (default robot), `capture=1`
  disables autoplay/UI.
- `window.seek(t)` poses everything (robot, camera, moving vehicles) at time t seconds and renders one frame
  synchronously. `window.duration` = last frame time. Set `window.__ready = true` when loaded and first frame rendered.
- Renderer: antialias, preserveDrawingBuffer: true, ACES tone mapping, sRGB; canvas fills the window.
- Three.js 0.170.0 via import map: https://unpkg.com/three@0.170.0/build/three.module.js and addons
  https://unpkg.com/three@0.170.0/examples/jsm/ (EffectComposer + UnrealBloomPass allowed).
- Procedural textures only (canvas), no external images.

Return only the complete HTML file in one ```html code block."""

CRITIQUE = """Each image is a pair: LEFT = real robot camera frame, RIGHT = your twin rendered from the robot camera at
the same moment ({times}). Browser console errors: {errors}

Score how indistinguishable they are (0-10) on the first line as `SCORE: <n>`, then list the most visible
differences (geometry/layout first: where the curb, crosswalk, buildings, lights and vehicles are; then camera
height/pitch/FOV; then materials, lighting, glow, color grade). Then return the complete improved HTML file in one
```html code block, keeping the contract exactly."""


def trajectory(clip: Path) -> dict:
    meta = json.loads((clip / "meta.json").read_text())
    p = np.load(clip / "poses_recorded.npy").astype(float)
    x0, y0, th0 = p[0, 0], p[0, 1], p[0, 2]
    c, s = math.cos(-th0), math.sin(-th0)
    dx, dy = p[:, 0] - x0, p[:, 1] - y0
    xs, ys = c * dx - s * dy, s * dx + c * dy
    yaws = np.unwrap(p[:, 2] - th0)
    fps = meta["fps"]
    cam = meta["camera_params_source"]["pinhole_output"]
    return {"fps": fps, "camera": {"hfov_deg": cam["hfov_deg"], "vfov_deg": cam["vfov_deg"],
                                   "height_m": meta["camera_params_source"]["camera_height_m"],
                                   "aspect": cam["image_size_wh"][0] / cam["image_size_wh"][1]},
            "frames": [{"t": round(i / fps, 3), "x": round(float(x), 3), "y": round(float(y), 3), "yaw": round(float(w), 4),
                        "v": round(float(v), 3)} for i, (x, y, w, v) in enumerate(zip(xs, ys, yaws, p[:, 3]))]}


def extract_html(text: str) -> str:
    m = re.search(r"```html\s*(.*?)```", text, re.S)
    html = (m.group(1) if m else text).strip()
    if "<html" not in html.lower():
        raise ValueError("no HTML in Argon's answer")
    return html


def real_frames(video: Path, times: list[float], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, t in enumerate(times):
        p = out / f"real_{i}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=960:540", str(p)], check=True)
        paths.append(p)
    return paths


def build(clip: Path, rounds: int = 3, log=print) -> Path:
    out = ROOT / "demo_assets" / f"coco_{clip.name}"
    traj = trajectory(clip)
    dur = traj["frames"][-1]["t"]
    times = [round(dur * f, 2) for f in (0.05, 0.22, 0.4, 0.58, 0.76, 0.94)]
    return build_from(clip / "rgb_pinhole.mp4", traj, out, times, times, rounds=rounds, log=log)


def build_from(video: Path, traj: dict, out: Path, real_times: list[float], twin_times: list[float], notes: str = "",
               rounds: int = 3, log=print) -> Path:
    """Twin of a real clip. real_times are clip times of the reference frames; twin_times the matching twin times."""
    out.mkdir(parents=True, exist_ok=True)
    tpath = out / "trajectory.json"
    tpath.write_text(json.dumps(traj))
    times = twin_times
    reals = real_frames(video, real_times, out / "frames")
    cam = traj["camera"]
    spec = SPEC.format(height=cam["height_m"], vfov=cam["vfov_deg"], aspect=cam["aspect"], times=times) + notes
    sample = json.dumps({**traj, "frames": traj["frames"][::20]})
    base = [argon.text(spec + "\n\n## Trajectory (every 20th frame shown)\n```json\n" + sample + "\n```")]
    base += [argon.image(p) for p in reals]
    html = extract_html(argon.ask(base, system=SYSTEM, thinking="high", tag="coco_twin_v1"))
    history = []
    for i in range(1, rounds + 1):
        hp = out / f"twin_v{i}.html"
        hp.write_text(html)
        twins, errors = shots_at(hp, tpath, times, out / f"render_v{i}", cam="robot", size=(960, 540))
        pairs = [pair(r, t, out / f"render_v{i}" / f"pair_{k}.png") for k, (r, t) in enumerate(zip(reals, twins))]
        log(f"twin_v{i}: rendered {len(twins)} frames, {len(errors)} console errors")
        if i == rounds:
            break
        res = argon.ask(base + [argon.text(f"Your current twin:\n```html\n{html}\n```")] + [argon.image(p) for p in pairs]
                        + [argon.text(CRITIQUE.format(times=times, errors="; ".join(errors[:10]) or "none"))],
                        system=SYSTEM, thinking="high", tag=f"coco_twin_v{i + 1}")
        score = re.search(r"SCORE:\s*([\d.]+)", res)
        history.append({"round": i, "score_of_v": i, "score": float(score.group(1)) if score else None,
                        "critique": res.split("```html")[0].strip()[:3000]})
        log(f"  Argon scores twin_v{i}: {history[-1]['score']}/10")
        html = extract_html(res)
    (out / "twin_history.json").write_text(json.dumps(history, indent=1))
    final = out / "twin.html"
    final.write_text(html)
    return final


def videos(clip: Path) -> None:
    out = ROOT / "demo_assets" / f"coco_{clip.name}"
    tw, tp = out / "twin.html", out / "trajectory.json"
    capture(tw, tp, out / "twin_robotcam.mp4", cam="robot", fps=20, size=(960, 540))
    capture(tw, tp, out / "twin_chase.mp4", cam="chase", fps=20, size=(960, 540))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip / "rgb_pinhole.mp4"), "-i", str(out / "twin_robotcam.mp4"),
                    "-filter_complex", "[0]scale=960:540,fps=20[a];[1]scale=960:540,fps=20[b];[a][b]hstack=shortest=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out / "real_vs_twin.mp4")], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--videos-only", action="store_true")
    a = ap.parse_args()
    clip = Path(a.clip).resolve()
    if not a.videos_only:
        print(build(clip, a.rounds))
    videos(clip)
    print("done")
