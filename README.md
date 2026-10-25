# Federated Robot Learning - Enabled by Argon
**Demo site:** https://bouncypitch.github.io/hindsight/ · **1-minute video:** https://youtu.be/ZgMhqIWLpNs

https://github.com/user-attachments/assets/f1342774-0385-4251-81a7-e0a83d48397b

**Hindsight.** Every robot failure makes the fleet smarter. Federated robot learning, enabled by Gemini 4 Argon.

**Robots that learn from their own failures.** Hindsight watches real robot footage (indoors and outdoors), finds the moments a robot went wrong, rebuilds each one as a physics digital twin that reproduces the failure, retrains the robot with reinforcement learning, and only ships the fix once **Gemini 4 Argon** verifies it. Robots then share what they learned as tiny policy updates instead of hours of video.


Built in one day at the Gemini 4 Argon SF Hackathon (Google DeepMind x Cerebral Valley, 2026-10-25).

## The proof
Two real failures from public FrodoBots-Mini-4K footage, one indoor and one outdoor, both fixed and approved by Argon:
- **Indoor:** a robot clipped a barstool leg and flipped onto its side.
- **Outdoor:** a robot drove into another robot parked among bikes on a crowded sidewalk.

For the barstool case, Hindsight:

1. **Confirmed** the failure from video + IMU (Argon, with visual evidence and timestamps).
2. **Rebuilt** it as a MuJoCo twin that reproduces the crash (original behavior: 3.5 cm clearance, collision).
3. **Retrained** the policy with an Argon-written reward on 40 Argon-planned variants: **success 0% → 100%, collisions 100% → 0%, at least 25 cm clearance in every variant.**
4. **Verified:** Argon reviewed the real clip and the old and new twins, then approved: *"smoothly steers left to give the obstacle a wide berth… passes with 0.307 m clearance… realigns without oscillation, stalling, or collision."*

The sidewalk case went the same way: 0% → 100% success on 40 variants, 0.27 m clearance in the original scenario, approved by Argon.

Before approving, Argon blocked 7+ retrained policies that gamed the reward (freezing then creeping, side-scraping, speeding). Along the way, Argon and the managed agent exposed three bugs in our own simulator.

## Results
| | |
|---|---|
| Public data downloaded and scanned | 8.1 GB · 788 rides · ~50 h of robot video from 6 countries (FrodoBots-7K + Mini-4K) |
| Failure mining on real public video | 130 physics candidates → 44 confirmed by Argon → **15/15 correct on blind human review** (a naive sensor+cheap-model approach scored 0/13 when hand-checked) |
| Argon vs Gemini 3.8 Flash (same strict prompt) | on the 15 human-verified failures, Flash missed 3 (80% recall); Argon missed none (caveat: these positives came from Argon's own detections) |
| End-to-end fix, indoor (barstool) | 0% → 100% success on 40 twin variants, ≥ 0.25 m clearance; Argon: ship |
| End-to-end fix, outdoor (sidewalk) | 0% → 100% success on 40 twin variants, 0.27 m clearance; Argon: ship |
| Regression caught, then fixed | Argon blocked an update that fixed San Francisco but raised Berkeley's stalls 0% → 20%; retrained with Berkeley's scenarios, it passed both (SF 0% → 100%, Berkeley 80% → 100%) and Argon approved it |
| Managed agent (Antigravity sandbox) | 42 min autonomous run: collisions 97.5% → 2.5%, found a spawn-height bug that corrupted every impact reading |
| Data uploaded (measured on Mini-4K) | everything: 3.57 GB (7.7 h) → only failures: 68 MB (53× less) → only policy updates: ~4.6 KB (**~770,000× less**) |
| Federated fleet (4 simulated cities) | Argon accepted the update that took one city's curb falls 60% → 0%, and blocked updates that would hurt other cities |
| Simulator bugs surfaced by Argon / the managed agent | collision check missed side-scrapes · chase camera hid clearance · spawn drop faked a 3 g impact in every episode (all fixed) |

## How it works
```
fleet video ─► physics triggers ─► Argon: confirm ─► Argon: diagnose ─► Argon: build twin ⟲ (revise until it reproduces)
                                                                                │
   Argon: variants ─► Argon: reward ─► [ RL (CMA-ES) ─► Argon: verify ⟲ (hack found → rewrite reward) ─► ship / block ]
                                        └──────────── runs autonomously in an Antigravity managed agent ────────────┘
                                                                                │
   robots upload ~100 B updates ─► Argon: fleet gate ─► master policy        Argon: 3D world ─► Gemini Omni (photoreal)
```

## Agents
- **Gemini 4 Argon** (`gemini-4-argon`), the lead agent: failure auditor, diagnostician, twin builder, scenario planner, reward engineer, safety verifier, fleet gatekeeper, 3D world builder.
- **Antigravity managed agent** (`antigravity-preview-09-2026`, model Argon): autonomous train → audit → fix-reward loop in a hosted sandbox.
- **Gemini Omni Flash** (`gemini-omni-1.1-flash`): photoreal restyle of twin renders (geometry and motion come from the twin).
- **Gemini 3.5 Flash-Lite**: cheap first-pass screening of fleet clips.
- **MuJoCo + CMA-ES**: physics and RL (not LLMs; Argon checks their output).

## Repo map
| Path | What |
|---|---|
| `curbsense/argon.py` | Interactions API wrapper (video/image input, structured output, thinking level, background mode, call log) |
| `data/mini4k.py` | Mini-4K ride loader, physics failure triggers, 1024×576 clip cutting |
| `curbsense/steps/s0_confirm.py` | strict Argon failure confirmation + labeled evaluation |
| `curbsense/steps/s1_reward.py` | Argon writes the reward from good/bad clips; verified against reference behaviors |
| `curbsense/steps/s2_detect.py` | Argon diagnosis (video + IMU → cause, time, scene, unknowns) |
| `curbsense/steps/s3_twin.py` | twin builder with reproduction test and Argon revision loop; variant planning |
| `curbsense/steps/s4_train.py` | CMA-ES policy search on twin variants |
| `curbsense/steps/s5_verify.py` | Argon ship/no-ship review with reward rewriting |
| `curbsense/pipeline.py` | end-to-end run for one real failure; writes `demo_assets/<case>/` |
| `curbsense/managed.py` | the same loop inside an Antigravity managed agent |
| `curbsense/fleet.py` | federated rounds with the Argon fleet gate |
| `curbsense/sim/` | MuJoCo world (curb drop/climb, slope, obstacle, pedestrian), policy, rollout |
| `curbsense/viz/` | Argon-built Three.js twins, real-clip scene matching, what-ifs, Omni restyle |
| `docs/` | the demo website (GitHub Pages) |
| `app/live.html` + `app/live_server.py` | live control room that runs Argon on a chosen failure |
| `app/video/` | the 1-minute film: `script_site.json`, `player_launch.html`, ElevenLabs narration (`tts.py`), recorder (`render.py`) |
| `scripts/` | proofs and visuals: `proof_barstool.py`, `variant_grid40.py`, `fleet_fix.py`, sweeps |

Note: the Python package is still named `curbsense` (the project's working name).

## Run
```bash
uv sync && echo "GEMINI_API_KEY=..." > .env
uv run python data/fetch_frodobots.py ...                 # or download Mini-4K shards (see data/mini4k.py)
uv run python -m data.mini4k                             # mine failure candidates
uv run python -m curbsense.steps.s0_confirm --candidates data/mini4k_clips/candidates.json
uv run python -m curbsense.pipeline data/mini4k_clips/<clip>.mp4 --reward <reward.py>
uv run uvicorn app.live_server:app --port 8780            # live demo
```

## Honest limits
- Small samples; recall of failure mining is not measured.
- The sim robot and the tunable policy are stand-ins for a production robot and neural policy.
- Many real failures in the public data come from indoor rides and a walled robot arena.
- Data savings use front-camera video only.

## Data and credits
- **FrodoBots-Mini-4K** (CC-BY-SA-4.0) and **BitRobot/berkeley-frodobots-lerobot-7k** (Apache-2.0), via Hugging Face.
- **Coco Robotics sample clips** from [VAIL-UCLA/MIMIC](https://github.com/VAIL-UCLA/MIMIC), used with credit and not redistributed here.
- Physics: MuJoCo. Narration: ElevenLabs.
- No dataset media is included in this repository.
