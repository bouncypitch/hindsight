"""What-if scenarios on the Coco digital twin: Argon adds counterfactual hazards to the twin of a real clip, and the
robot either follows its recorded path (and fails) or the retrained policy (and handles it).

uv run python -m curbsense.viz.whatif data/coco_samples/09294dbb
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from curbsense import argon
from curbsense.viz.coco_twin import SYSTEM, extract_html
from curbsense.viz.shoot import ROOT, capture, shots_at

SCENARIOS = {
    "truck_runs_light": "The red pickup that passes early in the clip comes back (or a second identical pickup) and runs "
                        "the red light through the crosswalk exactly while the robot is crossing, on a path that intersects the robot.",
    "pedestrian_steps_out": "A pedestrian (realistic articulated walking person, ~1.7 m) steps off the curb and walks across "
                            "the robot's path at walking speed, timed so the recorded robot path would hit them.",
    "missing_curb_ramp": "The curb ramp the robot goes down is replaced by a plain vertical 0.20 m curb (no ramp, no tactile pad).",
    "scooter_cuts_across": "An e-scooter rider (person standing on a scooter, ~4 m/s) cuts diagonally across the robot's path.",
}

SPEC = """Extend the digital twin below with WHAT-IF scenarios. Keep the scene, camera and contract exactly as they are
and add two URL params:

`scenario` = baseline | {names}
`policy` = recorded | safe

Scenarios:
{scenarios}

Policies:
- recorded: the robot follows its real recorded path and timing exactly (frames in the trajectory JSON), so it
  ends up in trouble. For the vehicle / pedestrian / scooter scenarios, time the actor so that it reaches the
  robot's path when the robot is there; at the moment of contact the robot is shoved slightly and stops (a short
  jolt, no gore), and the actor stops too (pedestrian stumbles back a step, vehicle brakes).
- safe: the retrained policy. Same path, but the robot's progress along the path is re-timed in JS: it slows and
  yields (stops >= 1.5 m before the conflict point) until the actor has fully cleared, then resumes. No contact.

missing_curb_ramp uses REAL PHYSICS from our MuJoCo simulator instead of animation. Fetch
`whatif/mujoco_drop_recorded.json` (policy=recorded) or `whatif/mujoco_drop_safe.json` (policy=safe), relative to
the twin page. Format: frames[i] = {{t, p:[x,y,z], q:[w,x,y,z] (MuJoCo, z-up)}}; meta.edge_t = time the robot's
center crosses the curb edge (MuJoCo x = 0, robot drives +x, sidewalk top at z = meta.world.curb_height, road at
z = 0). In the twin, find the time t_e when the recorded path reaches the curb edge you are replacing. From
t_e - meta.edge_t onward, drive the robot pose from the MuJoCo frames: position = edge point on the path + MuJoCo
(x, y) rotated by the path heading at t_e, height from MuJoCo z (shift so sidewalk heights match), orientation =
path heading composed with the MuJoCo quaternion (convert z-up to y-up correctly). When the MuJoCo frames end:
recorded -> the robot stays where it fell (it flipped); safe -> resume the recorded path from the matching distance.

Also add `window.scenarioInfo = {{scenario, policy, conflict_t}}` where conflict_t is when the hazard peaks, so the
capture tool can frame it. Return only the complete HTML file in one ```html code block."""


def build(clip: Path) -> Path:
    out = ROOT / "demo_assets" / f"coco_{clip.name}"
    twin = (out / "twin.html").read_text()
    traj = json.loads((out / "trajectory.json").read_text())
    spec = SPEC.format(names=" | ".join(SCENARIOS), scenarios="\n".join(f"- {k}: {v}" for k, v in SCENARIOS.items()))
    parts = [argon.text(spec), argon.text(f"Trajectory (every 20th frame):\n```json\n{json.dumps({**traj, 'frames': traj['frames'][::20]})}\n```"),
             argon.text(f"Current twin:\n```html\n{twin}\n```")]
    # the what-if twin is ~3k lines: needs the long-output mode (background + max_output_tokens > 64k)
    html = extract_html(argon.ask(parts, system=SYSTEM, thinking="high", background=True, max_output_tokens=200000, tag="whatif"))
    p = out / "whatif.html"
    p.write_text(html)
    return p


def render(clip: Path) -> list[Path]:
    out = ROOT / "demo_assets" / f"coco_{clip.name}"
    page, tp = out / "whatif.html", out / "trajectory.json"
    vids = []
    for sc in SCENARIOS:
        for pol in ("recorded", "safe"):
            for cam in ("chase", "robot"):
                mp4 = out / "whatif" / f"{sc}__{pol}__{cam}.mp4"
                capture(page, tp, mp4, cam=cam, fps=20, size=(960, 540), extra=f"&scenario={sc}&policy={pol}")
                vids.append(mp4)
    return vids


def grid(clip: Path) -> Path:
    """2x4 mosaic: each scenario's recorded (top) vs safe (bottom), chase camera."""
    out = ROOT / "demo_assets" / f"coco_{clip.name}" / "whatif"
    ins, tiles = [], []
    for i, sc in enumerate(SCENARIOS):
        for j, pol in enumerate(("recorded", "safe")):
            ins += ["-i", str(out / f"{sc}__{pol}__chase.mp4")]
            tiles.append(f"{i * 480}_{j * 270}")
    n = len(tiles)
    filt = "".join(f"[{k}]scale=480:270,tpad=stop_mode=clone:stop_duration=30[s{k}];" for k in range(n))
    filt += "".join(f"[s{k}]" for k in range(n)) + f"xstack=inputs={n}:layout={'|'.join(tiles)}:shortest=1"
    g = out / "whatif_grid.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex", filt, "-t", "19",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(g)], check=True)
    return g


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()
    clip = Path(a.clip).resolve()
    if not a.render_only:
        print(build(clip))
    out = ROOT / "demo_assets" / f"coco_{clip.name}"
    shots, errs = shots_at(out / "whatif.html", out / "trajectory.json", [3, 8, 12, 16], out / "whatif" / "check", cam="chase", size=(960, 540))
    print("check shots:", [str(s) for s in shots], "errors:", errs[:5])
