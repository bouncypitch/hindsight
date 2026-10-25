"""All 40 twin variants of the barstool failure, before (original behavior) and after (approved policy), overhead view."""
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from curbsense.scenario import ScenarioSpec
from curbsense.sim.policy import PolicyParams
from curbsense.sim.rollout import run
from curbsense.steps.s3_twin import observed_policy

D = Path("demo_assets/141102_tip_over_443s")
OUT = Path("runs/grid40")


def one(args):
    i, vd, tag, pol = args
    v = ScenarioSpec.from_dict(vd)
    p = OUT / f"{tag}_{i:02d}.mp4"
    r = run(v, PolicyParams(**pol), video=p, view="scene")
    return tag, i, r["outcome"]


def mosaic(tag, color):
    ins, filt = [], ""
    for i in range(10):
        ins += ["-i", str(OUT / f"{tag}_{i:02d}.mp4")]
        filt += f"[{i}]setpts=PTS/1.6,scale=384:216,tpad=stop_mode=clone:stop_duration=8,drawbox=x=0:y=0:w=384:h=216:color={color}@1:t=4[s{i}];"
    layout = "|".join(f"{(i % 5) * 384}_{(i // 5) * 216}" for i in range(10))
    filt += "".join(f"[s{i}]" for i in range(10)) + f"xstack=inputs=10:layout={layout},pad=1920:1080:0:324:color=black,format=yuv420p[v]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex", filt, "-map", "[v]", "-t", "6", "-c:v", "libx264",
                    "-crf", "22", "-preset", "veryfast", str(OUT / f"grid_{tag}.mp4")], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = ScenarioSpec.from_dict(json.loads((D / "03_twin/twin.json").read_text())["spec"])
    variants = json.loads((D / "03_twin/variants.json").read_text())
    old = json.loads(observed_policy(spec).to_json())
    new = json.loads((D / "policy.json").read_text())
    variants = variants[:10]
    jobs = [(i, v, "before", old) for i, v in enumerate(variants)] + [(i, v, "after", new) for i, v in enumerate(variants)]
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, jobs))
    for tag in ("before", "after"):
        outs = [o for t, _, o in res if t == tag]
        print(tag, {o: outs.count(o) for o in set(outs)})
    mosaic("before", "0xff6b6b")
    mosaic("after", "0x2ee6d6")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(OUT / "grid_before.mp4"), "-i", str(OUT / "grid_after.mp4"), "-filter_complex",
                    "[0][1]xfade=transition=fade:duration=0.8:offset=4.5,format=yuv420p[v]", "-map", "[v]", "-c:v", "libx264", "-crf", "22",
                    "demo_assets/rl_10_variants.mp4"], check=True)


if __name__ == "__main__":
    main()
