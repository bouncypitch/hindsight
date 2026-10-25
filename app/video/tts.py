"""Narration for each scene of the demo video with ElevenLabs; writes audio/<id>.mp3 and durations into timing.json."""
import json
import os
import subprocess
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
load_dotenv(HERE.parents[1] / ".env")
VOICE = os.environ.get("ELEVEN_VOICE", "nPczCjzI2devNBz1zQrb")  # Brian


def dur(p: Path) -> float:
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                                capture_output=True, text=True).stdout)


def main(script_name="script.json", timing_name="timing.json"):
    script = json.loads((HERE / script_name).read_text())
    (HERE / "audio").mkdir(exist_ok=True)
    timing = {}
    for sc in script["scenes"]:
        out = HERE / "audio" / f"{sc['id']}.mp3"
        if not out.exists():
            r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE}?output_format=mp3_44100_128",
                              headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                              json={"text": sc["say"], "model_id": "eleven_multilingual_v2",
                                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.15}},
                              timeout=120)
            r.raise_for_status()
            out.write_bytes(r.content)
        timing[sc["id"]] = round(dur(out), 2)
        print(sc["id"], timing[sc["id"]], "s")
    (HERE / timing_name).write_text(json.dumps(timing, indent=1))
    print("total", round(sum(timing.values()), 1), "s")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])
