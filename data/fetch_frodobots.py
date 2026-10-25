"""Download a slice of the public Berkeley FrodoBots LeRobot dataset (Apache-2.0).

Dataset: https://huggingface.co/datasets/BitRobot/berkeley-frodobots-lerobot-7k
uv run python data/fetch_frodobots.py --videos 0 2 3 --data 0 1 2 3 4
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

from huggingface_hub import hf_hub_download

REPO = "BitRobot/berkeley-frodobots-lerobot-7k"
OUT = "data/raw/frodobots"

ap = argparse.ArgumentParser()
ap.add_argument("--videos", type=int, nargs="+", default=[0])
ap.add_argument("--data", type=int, nargs="+", default=[0])
a = ap.parse_args()
files = ["meta/info.json", "meta/stats.json"] + [f"meta/episodes/chunk-000/file-{i:03d}.parquet" for i in range(5)]
files += [f"data/chunk-000/file-{i:03d}.parquet" for i in a.data]
files += [f"videos/observation.images.front/chunk-000/file-{i:03d}.mp4" for i in a.videos]


def get(f):
    return hf_hub_download(REPO, f, repo_type="dataset", local_dir=OUT)


with ThreadPoolExecutor(4) as ex:
    for p in ex.map(get, files):
        print("ok", p, flush=True)
