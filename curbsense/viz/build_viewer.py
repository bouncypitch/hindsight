"""Argon builds a realistic Three.js street that replays MuJoCo trajectories, then improves it
from screenshots (verification loop: render -> Argon critiques against a real street -> fix).

uv run python -m curbsense.viz.build_viewer            # writes app/viewer/viewer.html
uv run python -m curbsense.viz.capture runs/viz/naive.json runs/viz/naive_real.mp4
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from curbsense import argon
from curbsense.viz.shoot import screenshots

ROOT = Path(__file__).resolve().parents[2]
VIEWER = ROOT / "app" / "viewer" / "viewer.html"

SYSTEM = """You are a senior graphics engineer who builds photorealistic real-time scenes in Three.js.
You write complete, working single-file HTML apps. No placeholders, no TODOs."""

SPEC = """Build `viewer.html`: a single self-contained HTML file that replays a sidewalk delivery robot
trajectory (exported from a MuJoCo physics sim) inside a realistic US city street.

## Scene (must look like a real street photo, not a toy)
- A concrete sidewalk with expansion joints, weathering and subtle stains, ending at a granite/concrete
  curb face that drops `meta.world.curb_height` meters to an asphalt road. Curb edge lies along the y axis
  at x = meta.world.curb_x; sidewalk is x < 0 at height curb_height; road is x > 0 at z = 0.
- Asphalt with fine aggregate texture, patches, a painted lane line and a crosswalk a few meters away,
  a gutter line at the curb base, a storm drain.
- Street context: building facades with windows across the sidewalk (x around -6), street trees with tree
  pits, a street lamp, a fire hydrant, parked car silhouettes down the road, distant buildings fading into haze.
- Lighting: late-morning sun with soft shadows (shadow maps, PCFSoft), a physically based sky
  (three/addons Sky), ACES filmic tone mapping, sRGB output, mild fog for depth.
- Generate all textures procedurally on canvas (noise, cracks, aggregate) with normal/roughness variation.
  No external image files.

## Robot (match the physics body exactly; dimensions in meters, half-extents)
Use `meta.robot`: chassis box half-extents chassis_half, cargo bay box half-extents cargo_half at
cargo_offset, 4 wheels radius wheel_radius and half-width wheel_half_width at wheel_offsets (all in the
robot body frame, z up, x forward). Make it look like a modern sidewalk delivery robot: rounded white
shell with a colored lid, a dark sensor strip with camera lenses at the front, small orange safety flag on a
thin pole, black rubber tires with hubcaps. Visual detail may round corners but must not change the overall size.

## Trajectory replay
JSON format (sample below): frames[i] = {t, p:[x,y,z] robot body position, q:[w,x,y,z] body quaternion
(MuJoCo order!), w:[fl,fr,rl,rr] wheel spin angles in rad about the wheel axle (body y axis)}.
MuJoCo is z-up; convert to Three.js (y-up) correctly: map sim (x, y, z) -> three (x, z, -y) and convert the
quaternion consistently (rotate the whole sim world by -90 deg about X). Interpolate between frames.

## Contract (other code depends on this; follow exactly)
- URL params: `traj` (URL of the JSON to fetch), `cam` = chase|side|street (default chase),
  `capture=1` disables autoplay and the UI overlay.
- `window.loadTrajectory(obj)` loads a trajectory object; `window.seek(t)` poses the scene at time t (s)
  and renders one frame synchronously; `window.duration` = last frame time.
- Set `window.__ready = true` once the scene and trajectory are loaded and the first frame is rendered.
- Canvas fills the window; renderer uses `preserveDrawingBuffer: true`.
- Cameras: chase = behind-right of the robot at ~2.5 m, looking at it, smoothed; side = fixed at the curb
  looking along the road side so the drop is visible; street = pedestrian eye height across the road.
- Without capture=1: autoplay loop, a small overlay with time, outcome (meta.outcome) and a camera switcher.
- Load Three.js 0.170.0 via an import map from https://unpkg.com/three@0.170.0/build/three.module.js and
  addons from https://unpkg.com/three@0.170.0/examples/jsm/.

Return only the complete HTML file in one ```html code block."""

CRITIQUE = """Here are screenshots of your viewer at several moments ({labels}).
Browser console errors (if any):
{errors}

Critique them like a VFX supervisor comparing against a real street photograph: what still looks fake or
toy-like (materials, scale, lighting, shadows, composition, robot design, camera framing), and anything broken
(robot floating or sinking into the ground, wrong orientation, wheels not touching the surface, missing
objects, blown-out exposure). Then return the complete improved HTML file in one ```html code block,
keeping the contract exactly."""


UPGRADE = """Upgrade the viewer below (keep everything that works and the existing contract) so a twin can be made
nearly indistinguishable from a REAL sidewalk robot camera clip.

1. New camera `cam=robot`: the robot's own front camera, rigidly attached to the robot body at x=+0.27 m,
   z=+0.10 m above the chassis center, looking forward and ~8 deg down, horizontal FOV from scene.camera.hfov_deg
   (default 110), with optional barrel distortion (scene.camera.distortion 0..0.3) as a post-process pass, plus
   mild sensor noise, vignetting and motion blur so it looks like a cheap wide-angle robot camera.
2. Scene description: URL param `scene` (URL of a JSON file) or `window.loadScene(obj)` BEFORE the trajectory
   renders. Everything about the look comes from it, with sensible defaults for missing fields:
   {
    "ground": {"type": "concrete_slabs|square_pavers|checker_pavers|brick|asphalt|gravel|grass",
               "color": "#hex", "secondary_color": "#hex", "tile_size_m": 0.6, "wear": 0..1, "wet": 0..1},
    "road": {"present": true, "color": "#hex", "lane_markings": true},
    "curb": {"color": "#hex", "style": "concrete|granite|painted_red|painted_yellow"},
    "left": [ {"type": "building|fence|hedge|grass|trees|parked_cars|wall|open_plaza|canopy", "distance_m": 4,
               "height_m": 8, "color": "#hex"} ],
    "right": [ same ],
    "props": [ {"type": "bollard|tree|lamp_post|bench|trash_can|planter|bike|sign|fire_hydrant|traffic_cone",
                "x": 3.0, "y": 1.2} ],
    "sky": {"type": "clear|hazy|overcast", "sun_elevation_deg": 35, "sun_azimuth_deg": 120, "exposure": 1.0},
    "fog": {"density": 0.0..0.05, "color": "#hex"},
    "grade": {"saturation": 0.6..1.3, "contrast": 0.8..1.2, "tint": "#hex", "warmth": -1..1},
    "camera": {"hfov_deg": 110, "distortion": 0.15, "noise": 0.03, "height_m": 0.10, "pitch_deg": -8}
   }
3. Scenario geometry from meta.world.kind (see the sample): drop (as now), climb (road x<0, raised sidewalk
   x>0), slope (ramp rising at slope_deg for slope_len m from x=0, then plateau), obstacle (meta.world.obstacle:
   x, y, r - render as a realistic bollard/pole/bin of that radius), pedestrian (frames[i].ped = [x, y]: render a
   realistic walking person - articulated legs/arms with a walk cycle, clothes, ~1.7 m tall, facing the walking
   direction). Ground for drop/climb/slope/obstacle/pedestrian all use scene.ground.
4. Keep `window.__ready`, `window.seek(t)`, `window.duration`, `cam=chase|side|street`, `capture=1`.

Return only the complete HTML file in one ```html code block."""


def upgrade(sample_traj: Path, src: Path, out: Path) -> Path:
    html = src.read_text()
    parts = [argon.text(UPGRADE + "\n\n## Sample trajectory\n```json\n" + sample(sample_traj) + "\n```\n\n"
                        "## Current viewer\n```html\n" + html + "\n```")]
    res = argon.ask(parts, system=SYSTEM, thinking="high", background=True, max_output_tokens=150000, tag="viewer_upgrade")
    out.write_text(extract_html(res))
    return out


def extract_html(text: str) -> str:
    m = re.search(r"```html\s*(.*?)```", text, re.S)
    html = (m.group(1) if m else text).strip()
    if "<html" not in html.lower():
        raise ValueError("Argon did not return an HTML document")
    return html


def sample(traj_path: Path) -> str:
    t = json.loads(traj_path.read_text())
    return json.dumps({"meta": t["meta"], "frames": t["frames"][:3] + ["..."]}, indent=1)


def build(traj: Path, rounds: int = 2, start: Path | None = None, first: int = 1, log=print) -> Path:
    VIEWER.parent.mkdir(parents=True, exist_ok=True)
    base = [argon.text(SPEC + "\n\n## Sample trajectory\n```json\n" + sample(traj) + "\n```")]
    if start:
        html = start.read_text()
    else:
        out = argon.ask(base, system=SYSTEM, thinking="high", background=True, max_output_tokens=120000, tag="viewer_v1")
        html = extract_html(out)
    for i in range(first, first + rounds):
        path = VIEWER.with_name(f"viewer_v{i}.html")
        path.write_text(html)
        VIEWER.write_text(html)
        shots, errors = screenshots(VIEWER, traj, ROOT / "runs" / "viz" / f"shots_v{i}")
        log(f"viewer_v{i}: {len(shots)} screenshots, {len(errors)} console errors")
        if i == first + rounds - 1:
            break
        parts = base + [argon.text(f"Your current file:\n```html\n{html}\n```")]
        parts += [argon.image(p) for _, p in shots]
        parts.append(argon.text(CRITIQUE.format(labels=", ".join(l for l, _ in shots),
                                                errors="\n".join(errors[:30]) or "(none)")))
        # background mode does not accept image input, so critique rounds run in the foreground
        out = argon.ask(parts, system=SYSTEM, thinking="high", tag=f"viewer_v{i + 1}")
        html = extract_html(out)
    return VIEWER


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="runs/viz/naive.json")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--start", help="existing viewer html to refine instead of generating from scratch")
    ap.add_argument("--first", type=int, default=1, help="version number of the start file")
    ap.add_argument("--upgrade", help="write an upgraded viewer (robot cam, scene JSON, new scenario kinds) to this path")
    a = ap.parse_args()
    if a.upgrade:
        print(upgrade(ROOT / a.traj, VIEWER, Path(a.upgrade)))
        raise SystemExit
    print(build(ROOT / a.traj, a.rounds, Path(a.start) if a.start else None, a.first))
