"""Make a twin look like its real clip: Argon describes the real scene, the viewer renders it from the robot's
own camera, Argon compares real vs render side by side and revises, until it judges them nearly indistinguishable.

uv run python -m curbsense.viz.match_scene REAL.mp4 TWIN_TRAJ.json --out demo_assets/<name>
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from curbsense import argon
from curbsense.viz.shoot import ROOT, capture, shots_at

VIEWER = ROOT / "app" / "viewer" / "viewer.html"


class Ground(BaseModel):
    type: Literal["concrete_slabs", "square_pavers", "checker_pavers", "brick", "asphalt", "gravel", "grass"]
    color: str
    secondary_color: str
    tile_size_m: float
    wear: float = Field(ge=0, le=1)
    wet: float = Field(ge=0, le=1)


class Side(BaseModel):
    type: Literal["building", "fence", "hedge", "grass", "trees", "parked_cars", "wall", "open_plaza", "canopy"]
    distance_m: float
    height_m: float
    color: str


class Prop(BaseModel):
    type: Literal["bollard", "tree", "lamp_post", "bench", "trash_can", "planter", "bike", "sign", "fire_hydrant", "traffic_cone"]
    x: float
    y: float


class Sky(BaseModel):
    type: Literal["clear", "hazy", "overcast"]
    sun_elevation_deg: float
    sun_azimuth_deg: float
    exposure: float


class Grade(BaseModel):
    saturation: float
    contrast: float
    tint: str
    warmth: float


class Camera(BaseModel):
    hfov_deg: float
    distortion: float
    noise: float
    height_m: float
    pitch_deg: float


class Scene(BaseModel):
    ground: Ground
    road: dict = Field(default_factory=lambda: {"present": True, "color": "#3a3a3c", "lane_markings": True})
    curb: dict = Field(default_factory=lambda: {"color": "#b8b5ae", "style": "concrete"})
    left: list[Side]
    right: list[Side]
    props: list[Prop]
    sky: Sky
    fog: dict = Field(default_factory=lambda: {"density": 0.01, "color": "#c8ccd2"})
    grade: Grade
    camera: Camera


class Comparison(BaseModel):
    similarity: float = Field(ge=0, le=10, description="10 = a viewer could not tell which is real")
    differences: list[str] = Field(description="most visible differences first")
    scene: Scene = Field(description="revised scene description that fixes the differences")


DESCRIBE = """These frames are from a sidewalk delivery robot's own front camera (low, wide-angle, cheap sensor).
Describe the scene precisely enough to rebuild it in a 3D renderer so that a render from the same camera is
nearly indistinguishable from these frames: ground material/pattern/colors (sample real hex colors from the
image), what is on the left and right and how far, props with positions (x forward, y left, meters, robot at
origin at the first frame), sky and sun, haze, color grade, and the camera's field of view, distortion, noise,
mounting height and pitch."""

COMPARE = """Left of each pair: REAL robot camera frame. Right: our 3D twin rendered from the same robot camera
at the same moment. Score how indistinguishable they are (0-10) and list the most visible differences, then
return a revised scene description that fixes them. Current scene description:
{scene}"""


def real_frames(clip: Path, n: int, out: Path) -> tuple[list[Path], list[float]]:
    out.mkdir(parents=True, exist_ok=True)
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(clip)],
                               capture_output=True, text=True).stdout.strip() or 8)
    fracs = [0.15 + 0.7 * i / max(1, n - 1) for i in range(n)]
    paths = []
    for i, f in enumerate(fracs):
        p = out / f"real_{i}.png"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{f * dur:.2f}", "-i", str(clip), "-frames:v", "1",
                        "-vf", "scale=896:512", str(p)], check=True)
        paths.append(p)
    return paths, fracs


def pair(a: Path, b: Path, out: Path) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-i", str(b), "-filter_complex",
                    "[0]scale=896:512[x];[1]scale=896:512[y];[x][y]hstack", str(out)], check=True)
    return out


def match(clip: Path, traj: Path, out: Path, rounds: int = 3, target: float = 8.5, log=print) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    reals, fracs = real_frames(clip, 4, out / "frames")
    dur = json.loads(traj.read_text())["frames"][-1]["t"]
    times = [f * dur for f in fracs]
    scene: Scene = argon.ask([argon.image(p) for p in reals] + [argon.text(DESCRIBE)], Scene, thinking="high", tag="scene_describe")
    history = []
    for i in range(1, rounds + 1):
        sp = out / f"scene_v{i}.json"
        sp.write_text(scene.model_dump_json(indent=1))
        twins, errors = shots_at(VIEWER, traj, times, out / f"render_v{i}", scene=sp)
        pairs = [pair(r, t, out / f"render_v{i}" / f"pair_{k}.png") for k, (r, t) in enumerate(zip(reals, twins))]
        cmp_: Comparison = argon.ask([argon.image(p) for p in pairs] +
                                     [argon.text(COMPARE.format(scene=scene.model_dump_json()) +
                                                 (f"\nViewer console errors: {errors[:5]}" if errors else ""))],
                                     Comparison, thinking="high", tag=f"scene_compare_{i}")
        history.append({"round": i, "similarity": cmp_.similarity, "differences": cmp_.differences})
        log(f"scene_v{i}: similarity {cmp_.similarity}/10; top differences: {cmp_.differences[:3]}")
        if cmp_.similarity >= target:
            break
        scene = cmp_.scene
    final = out / "scene.json"
    final.write_text(scene.model_dump_json(indent=1))
    (out / "match_history.json").write_text(json.dumps(history, indent=1))
    return final


def side_by_side(clip: Path, traj: Path, scene: Path, out_mp4: Path) -> Path:
    """Real clip (left) next to the scene-matched twin from the robot camera (right), same size and length."""
    twin = out_mp4.with_name(out_mp4.stem + "_twin_robotcam.mp4")
    capture(VIEWER, traj, twin, cam="robot", scene=scene, size=(896, 512))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(clip), "-i", str(twin), "-filter_complex",
                    "[0]scale=896:512,fps=30[a];[1]scale=896:512,fps=30[b];[a][b]hstack=shortest=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)], check=True)
    return out_mp4


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("traj")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    scene = match(Path(a.clip), Path(a.traj).resolve(), out, a.rounds)
    print(side_by_side(Path(a.clip), Path(a.traj).resolve(), scene, out / "real_vs_twin.mp4"))
