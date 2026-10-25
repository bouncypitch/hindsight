### Summary of Execution & Findings

1. **Environment & Dependencies**:
   - Installed `mujoco`, `numpy`, and `cma` in the sandbox environment.

2. **Round 1 Training (`reward.py` $\rightarrow$ `round1`)**:
   - Ran `python3 -m curbsense.steps.s4_train --reward /workspace/case/reward.py --variants /workspace/case/variants.json --gens 12 --out /workspace/out/round1`.
   - **Before (`NAIVE`)**: `reward = -447.77`, `success_rate = 0.0`, `fall_rate = 0.0`, `stall_rate = 0.025`, `collision_rate = 0.975`.
   - **After (`round1/policy.json`)**: `reward = -209.01`, `success_rate = 0.625` (25/40), `fall_rate = 0.0`, `stall_rate = 0.0`, `collision_rate = 0.025` (1/40).

3. **Policy & Per-Episode Inspection (`inspect_rounds.py`)**:
   - Evaluated `policy.json` across all 40 obstacle variants using `curbsense.sim.rollout.run` and uncovered both **reward design bugs** and **reward hacking**:
     - **Spawn-Drop `peak_impact` Bug**: `world.py` spawns the robot $2\text{ cm}$ above wheel equilibrium (`z0 = WHEEL_R + 0.07 + 0.02`), producing a $\sim 32.6\text{ m/s}^2$ accelerometer spike during the first $0.06\text{ s}$ as the robot settles onto the floor. Because `reward.py` defined `hard_contact = collided or fell or (peak_impact > 6.0)`, `hard_contact` evaluated to `True` and `clean_success` evaluated to `False` on **100% of episodes**, applying a constant $-180$ impact penalty and zeroing out the $+120$ clean-success bonus, time bonus, and clearance bonus.
     - **Hack 1 — Crawling / Stall-Detector Evasion on High-Friction Variants (14/40 Timeouts)**: When `cmd_w` saturates at $2.0\text{ rad/s}$ ($\omega \cdot \text{TRACK}/2 = 0.34\text{ m/s}$) with `cruise_speed` $\approx 0.38\text{ m/s}$, the inner wheels are commanded near zero while the outer wheels saturate the $2.5\text{ N}\cdot\text{m}$ hub motor torque cap against lateral tire scrub on high-friction variants ($\mu \ge 0.95$). The robot crawls forward at $\sim 0.08\text{ m/s}$—just above `Monitor`'s stall detector threshold ($0.05\text{ m}$ over $2.5\text{ s} = 0.02\text{ m/s}$)—reporting `stall_rate = 0.0` while timing out at $t = 16.0\text{ s}$ (`x = 0.46–0.93 m`) before ever reaching the obstacle (`obs_x = 1.02–1.75 m`).
     - **Hack 2 — Fake Upstream Clearance & Tiny-Clearance Grazing**: Because `reward.py` penalized `min_clearance < 0.45 m` without checking whether the robot reached the obstacle, stopping short on the 14 timeouts yielded a large apparent `min_clearance` ($0.21\text{–}0.70\text{ m}$) and zero clearance penalty while collecting upstream `raw_disp` progress. To squeeze out extra upstream progress on those 14 timeouts, CMA-ES pushed `cruise_speed` to $0.380\text{ m/s}$ and reduced `avoid_gain` to $3.611$, causing the robot to shave the obstacle on successful passes with **razor-thin clearance** ($0.005\text{ m} = 5\text{ mm}$ on `v28`, $0.080\text{ m}$ on `v23`, $0.081\text{ m}$ on `v11`, and $0.138\text{ m}$ on `v27`).

4. **Round 2 Reward Improvement & Retraining (`reward_v2.py` $\rightarrow$ `round2`)**:
   - Wrote `reward_v2.py` to: (a) compute a 3-tick smoothed post-spawn impact for $t > 0.30\text{ s}$ so smooth passes achieve `clean_success`, (b) only credit `min_clearance` when the robot actually reaches/passes `obs_x` and penalize stopping short, and (c) place a steep penalty barrier on tiny obstacle clearances ($< 0.04\text{ m}$ and $< 0.12\text{ m}$).
   - Retrained into `round2`: `reward` improved from `-484.67` (`before`) to `+35.55` (`after`), maintaining the $62.5\%$ (25/40) success rate, $0\%$ fall rate, $0\%$ stall rate, and $2.5\%$ (1/40) collision rate while **increasing minimum successful clearance by $9\times$** (from $0.005\text{ m}$ to $0.045\text{ m}$ on `v28`), increasing 10th-percentile clearance from $0.104\text{ m}$ to $0.139\text{ m}$ ($+34\%$), and reducing passes with $<0.10\text{ m}$ clearance from 3 to 1 (`v27` improved from $0.138\text{ m}$ to $0.181\text{ m}$, `v23` from $0.080\text{ m}$ to $0.120\text{ m}$, and `v11` from $0.081\text{ m}$ to $0.110\text{ m}$).
   - **Ship Decision (`ship: false`)**: Because `curbsense/sim/policy.py` fixes `dist_to_edge = 99.0` for `kind == "obstacle"` (disabling speed reduction) and commands constant `v = cruise_speed` while `set_drive` subtracts $\omega \cdot \text{TRACK}/2 = 0.34\text{ m/s}$ from the inside wheels under a $2.5\text{ N}\cdot\text{m}$ motor torque cap, **no parameter vector in `PolicyParams` can exceed $62.5\%$ (25/40) success** across the 40 variants: on the 14 high-friction ($\mu \ge 0.95$) variants, low `cruise_speed` ($\le 0.385\text{ m/s}$) crawls and times out, whereas higher `cruise_speed` ($\ge 0.45\text{ m/s}$) understeers into the obstacle and spikes collisions to $17.5\text{–}30\%$. Shipping requires allowing the controller to modulate speed or pivot before obstacle avoidance.

```json
{
  "rounds": [
    {
      "reward": "/workspace/case/reward.py",
      "before": {
        "reward": -447.7728163993737,
        "success_rate": 0.0,
        "fall_rate": 0.0,
        "stall_rate": 0.025,
        "collision_rate": 0.975,
        "mean_edge_speed": 1.0901634608222355,
        "mean_abs_edge_heading": 0.002136262651159339,
        "mean_peak_impact": 34.19325
      },
      "after": {
        "reward": -209.0107895464263,
        "success_rate": 0.625,
        "fall_rate": 0.0,
        "stall_rate": 0.0,
        "collision_rate": 0.025,
        "mean_edge_speed": 0.23360729392634397,
        "mean_abs_edge_heading": 10.437050950729759,
        "mean_peak_impact": 32.64
      }
    },
    {
      "reward": "/workspace/out/reward_v2.py",
      "before": {
        "reward": -484.674446456876,
        "success_rate": 0.0,
        "fall_rate": 0.0,
        "stall_rate": 0.025,
        "collision_rate": 0.975,
        "mean_edge_speed": 1.0901634608222355,
        "mean_abs_edge_heading": 0.002136262651159339,
        "mean_peak_impact": 34.19325
      },
      "after": {
        "reward": 35.55213236965188,
        "success_rate": 0.625,
        "fall_rate": 0.0,
        "stall_rate": 0.0,
        "collision_rate": 0.025,
        "mean_edge_speed": 0.20189231864845128,
        "mean_abs_edge_heading": 11.930382480157755,
        "mean_peak_impact": 32.64725
      }
    }
  ],
  "hacking_found": true,
  "hack_description": "In Round 1, two reward hacks and a reward bug interacted: (1) world.py spawns the robot 2 cm above wheel equilibrium, creating a ~32.65 m/s^2 accelerometer spike at t < 0.1s; reward.py treated peak_impact > 6.0 as hard_contact=True on 100% of episodes, disabling clean_success and its +120 bonus on every episode. (2) On the 14 high-friction variants (mu >= 0.95), commanding cruise_speed=0.38 m/s with saturated yaw rate cmd_w=2.0 rad/s (w*TRACK/2 = 0.34 m/s) under the 2.5 Nm wheel torque limit caused the robot to crawl at ~0.08 m/s—just above Monitor's 0.05 m / 2.5 s (0.02 m/s) stall threshold—evading the stall penalty (stall_rate=0.0) while timing out at x=0.46-0.93 m before reaching the obstacle (obs_x=1.02-1.75 m) and collecting fake high min_clearance (0.21-0.70 m). (3) To maximize upstream progress on those 14 timeouts, Round 1 traded off steering gain (avoid_gain=3.611, cruise_speed=0.380), causing the 25 successful episodes to shave obstacles with tiny clearance down to 0.005 m (5 mm on v28) and 3 passes below 0.10 m.",
  "final_policy": {
    "cruise_speed": 0.3280998371441812,
    "slow_dist": 0.2458890002942722,
    "slow_speed": 1.7490809771857065,
    "descent_angle_deg": 59.26958774681794,
    "heading_gain": 0.2148512538735674,
    "align_tol_deg": 77.12248487768814,
    "yield_dist": 2.0101646301548524,
    "avoid_dist": 3.9959611154262658,
    "avoid_gain": 3.130760105906441
  },
  "final_reward_code": "import math\n\n\ndef _num(d, key, default=0.0):\n    if not isinstance(d, dict):\n        return float(default)\n    v = d.get(key, default)\n    if v is None:\n        return float(default)\n    try:\n        f = float(v)\n    except (TypeError, ValueError):\n        return float(default)\n    if math.isnan(f) or math.isinf(f):\n        return float(default)\n    return f\n\n\ndef _clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n\n\ndef _smoothed_post_spawn_impact(states: list[dict]) -> float:\n    \"\"\"Compute 3-tick smoothed acceleration deviation from 1g after t > 0.30s to ignore the t=0 spawn drop and 1-tick chatter.\"\"\"\n    post = [s for s in states if _num(s, \"t\", 0.0) > 0.30]\n    if len(post) < 3:\n        return 0.0\n    mags = []\n    for s in post:\n        acc = s.get(\"accel\")\n        if isinstance(acc, (list, tuple)) and len(acc) == 3:\n            norm_a = math.sqrt(sum(float(c) * float(c) for c in acc))\n            mags.append(abs(norm_a - 9.81))\n    if len(mags) < 3:\n        return 0.0\n    peak = 0.0\n    for i in range(len(mags) - 2):\n        win_avg = (mags[i] + mags[i + 1] + mags[i + 2]) / 3.0\n        peak = max(peak, win_avg)\n    return peak\n\n\ndef score(ep: dict) -> float:\n    outcome = ep.get(\"outcome\", \"\")\n    success = bool(ep.get(\"success\", False) or outcome == \"success\")\n    fell = bool(ep.get(\"fell\", False) or outcome == \"fell\")\n    collided = bool(ep.get(\"collided\", False) or outcome == \"collided\")\n    near_miss = bool(ep.get(\"near_miss\", False) or outcome == \"near_miss\")\n    stalled = bool(ep.get(\"stalled\", False) or outcome == \"stalled\")\n    timeout = bool(\n        outcome == \"timeout\"\n        or (not success and not fell and not collided and not near_miss and not stalled)\n    )\n\n    duration = max(0.0, _num(ep, \"duration\", 0.0))\n    max_tilt_deg = max(0.0, _num(ep, \"max_tilt_deg\", 0.0))\n    jerk = max(0.0, _num(ep, \"jerk\", 0.0))\n    min_clearance = _num(ep, \"min_clearance\", 0.0) if ep.get(\"min_clearance\") is not None else None\n\n    states = ep.get(\"states\") or []\n    n = len(states)\n\n    # Ignore the t < 0.30s 2cm spawn-drop artifact (~32.6 m/s^2) when evaluating driving shock\n    drive_impact = _smoothed_post_spawn_impact(states)\n    hard_contact = collided or fell or (max_tilt_deg > 15.0) or (drive_impact > 12.0)\n    clean_success = success and not hard_contact and not near_miss and not stalled\n\n    # Estimate obstacle x and goal x from states (obs dict reports dx in robot frame)\n    start_x = _num(states[0], \"x\", -0.8) if n > 0 else -0.8\n    final_x = _num(states[-1], \"x\", start_x) if n > 0 else (3.0 if success else start_x)\n    max_x = max((_num(s, \"x\", start_x) for s in states), default=final_x)\n\n    obs_x_est = 1.5\n    if n > 0 and isinstance(states[0].get(\"obs\"), dict):\n        obs0 = states[0][\"obs\"]\n        obs_x_est = start_x + _num(obs0, \"dx\", 2.3)\n    goal_x_est = obs_x_est + 1.5\n    target_span = max(1.0, goal_x_est - start_x)\n\n    progress_frac = _clamp((final_x - start_x) / target_span, -0.2, 1.0)\n    passed_obstacle = final_x > (obs_x_est + 0.15)\n\n    total = 0.0\n\n    # 1. Terminal outcome rewards and penalties\n    if clean_success:\n        total += 160.0\n        if min_clearance is not None and min_clearance >= 0.15:\n            total += _clamp(14.0 - 1.4 * max(0.0, duration - 9.5), 0.0, 14.0)\n    elif success:\n        total += 60.0\n\n    if fell:\n        total -= 220.0\n    if collided:\n        total -= 220.0\n    if near_miss:\n        total -= 140.0\n    if stalled:\n        total -= 210.0\n    if timeout and not success:\n        total -= 130.0\n\n    # 2. Progress shaping\n    if success:\n        total += 40.0\n    elif passed_obstacle and not (collided or fell):\n        total += 20.0 * progress_frac\n    else:\n        shortfall = _clamp(1.0 - progress_frac, 0.0, 1.2)\n        total -= 45.0 * shortfall\n\n    # Penalize backtracking\n    backtrack_m = max(0.0, max_x - final_x)\n    total -= 25.0 * _clamp(backtrack_m, 0.0, 1.5)\n\n    # 3. Obstacle clearance: only credit clearance if the robot actually reached/passed the obstacle\n    if min_clearance is not None:\n        if passed_obstacle or collided or near_miss or (max_x >= obs_x_est - 0.35):\n            if min_clearance < 0.04:\n                total -= 140.0 + 1800.0 * (0.04 - max(0.0, min_clearance))\n            elif min_clearance < 0.12:\n                deficit = (0.12 - min_clearance) / 0.08\n                total -= 35.0 + 95.0 * (deficit ** 2)\n            elif min_clearance < 0.30:\n                deficit = (0.30 - min_clearance) / 0.18\n                total -= 35.0 * (deficit ** 2)\n            elif clean_success:\n                total += 50.0 * _clamp((min_clearance - 0.30) / 0.35, 0.0, 1.0)\n        else:\n            # Never reached the obstacle (crawled/parked upstream): no fake clearance bonus\n            total -= 35.0\n\n    # 4. Dynamic impact, tilt, and smoothness\n    if drive_impact > 4.0:\n        imp_ex = drive_impact - 4.0\n        total -= _clamp(4.0 * imp_ex + 0.3 * (imp_ex ** 2), 0.0, 100.0)\n\n    if max_tilt_deg > 8.0:\n        tilt_ex = max_tilt_deg - 8.0\n        total -= _clamp(2.5 * tilt_ex + 0.2 * (tilt_ex ** 2), 0.0, 80.0)\n\n    if n > 0:\n        crawl_ticks = 0\n        reverse_pen = 0.0\n        roll_pen = 0.0\n        for s in states:\n            spd = _num(s, \"speed\", 0.0)\n            vx = _num(s, \"vx\", spd)\n            cmd_v = _num(s, \"cmd_v\", 0.0)\n            roll = abs(_num(s, \"roll_deg\", 0.0))\n            if spd < 0.16 or vx < 0.12:\n                crawl_ticks += 1\n            if cmd_v < -0.02 or vx < -0.02:\n                reverse_pen += abs(min(cmd_v, vx))\n            if roll > 3.0:\n                roll_pen += (roll - 3.0)\n\n        crawl_ratio = crawl_ticks / float(n)\n        if not success:\n            total -= 55.0 * crawl_ratio\n        else:\n            total -= 15.0 * crawl_ratio\n        total -= _clamp(10.0 * reverse_pen / float(n), 0.0, 25.0)\n        total -= _clamp(4.0 * roll_pen / float(n), 0.0, 25.0)\n\n    if jerk > 0.4:\n        j_ex = jerk - 0.4\n        total -= _clamp(10.0 * j_ex + 3.0 * (j_ex ** 2), 0.0, 35.0)\n    elif clean_success:\n        total += 6.0 * ((0.4 - jerk) / 0.4)\n\n    return float(total)\n",
  "ship": false,
  "notes": "Round 2 fixed the t=0 spawn-drop peak_impact false positive and eliminated fake clearance credit upstream of the obstacle, increasing minimum obstacle clearance across the 25 successful variants by 9x (from 0.005 m to 0.045 m), improving 10th-percentile clearance from 0.104 m to 0.139 m, and reducing passes below 0.10 m clearance from 3 to 1 while maintaining 0% falls, 0% stalls, and 2.5% collisions (down from 97.5% on NAIVE). However, ship=false because PolicyParams.act() holds forward speed constant at cruise_speed during obstacle avoidance (since dist_to_edge=99.0 for kind='obstacle') while skid-steering subtracts w*TRACK/2=0.34 m/s from the inner wheel under a 2.5 Nm per-wheel torque cap; consequently, 62.5% (25/40) is the physical ceiling of PolicyParams on these 40 variants (with 14/40 high-friction mu>=0.95 variants timing out at low speed or colliding/stalling at higher speed). Safe deployment requires extending the controller to slow/pivot before close obstacles and re-align heading after passing."
}
```