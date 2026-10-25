import math


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _mag(val) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return abs(float(val))
    if isinstance(val, (list, tuple)):
        return math.sqrt(sum(float(v) * float(v) for v in val))
    return 0.0


def score(ep: dict) -> float:
    outcome = ep.get("outcome", "")
    success = bool(ep.get("success", outcome == "success"))
    fell = bool(ep.get("fell", outcome == "fell"))
    stalled = bool(ep.get("stalled", outcome == "stalled"))
    collided = bool(ep.get("collided", outcome == "collided"))
    near_miss = bool(ep.get("near_miss", outcome == "near_miss"))

    states = ep.get("states") or []
    n = len(states)

    # 1. Forward progress toward x = 1.2
    if n > 0:
        x0 = float(states[0].get("x", 0.0))
        x_end = float(states[-1].get("x", x0))
        target_dist = max(0.5, 1.2 - x0)
        raw_progress = _clamp((x_end - x0) / target_dist, -0.3, 1.1)
    else:
        raw_progress = 1.0 if success else 0.0

    # Discount progress if the robot crashed or flipped
    progress_mult = 0.2 if (fell or collided) else 1.0
    progress_reward = 75.0 * max(0.0, raw_progress) * progress_mult
    if raw_progress < 0.0:
        progress_reward += 25.0 * raw_progress

    # Penalize lack of forward progress when not succeeding (blocks parking/freezing)
    if not success and raw_progress < 0.75:
        progress_reward -= 45.0 * (0.75 - max(0.0, raw_progress))

    # 2. Terminal outcome rewards and penalties
    terminal_reward = 0.0
    if success:
        terminal_reward += 95.0
        duration = float(ep.get("duration") or 0.0)
        if duration > 0.0:
            terminal_reward += 15.0 * _clamp((8.0 - duration) / 5.5, 0.0, 1.0)

    if fell:
        terminal_reward -= 90.0
    if collided:
        terminal_reward -= 80.0
    if near_miss:
        # Lighter near-miss penalty when succeeding so close obstacle avoidance isn't discouraged
        terminal_reward -= 8.0 if success else 18.0
    if stalled:
        terminal_reward -= 65.0
    if not success and not (fell or collided or stalled):
        # Timeout / freezing in place must be penalized as heavily as stalling
        terminal_reward -= 65.0

    # 3. Obstacle clearance (allow tighter avoidance around close initial obstacles)
    min_clearance = ep.get("min_clearance", None)
    clearance_term = 0.0
    if min_clearance is not None:
        mc = float(min_clearance)
        safe_dist = 0.28
        if mc < safe_dist:
            shortfall = (safe_dist - max(0.0, mc)) / safe_dist
            clearance_term -= 20.0 * (shortfall * shortfall)
        elif raw_progress > 0.5 and not (fell or collided):
            clearance_term += 8.0 * _clamp((mc - safe_dist) / 0.45, 0.0, 1.0)

    # 4. Curb/goal approach speed and alignment (only evaluated when actually reaching the goal region)
    edge_speed = ep.get("edge_speed", None)
    edge_heading_deg = ep.get("edge_heading_deg", None)
    edge_term = 0.0

    reached_goal_region = (n > 0 and max(float(s.get("x", -1.0)) for s in states) >= 0.6) or success
    if edge_heading_deg is not None and reached_goal_region:
        eh = abs(float(edge_heading_deg))
        edge_term -= 8.0 * _clamp((eh / 35.0) ** 2, 0.0, 1.5)
        if not (fell or collided):
            edge_term += 5.0 * math.exp(-((eh / 20.0) ** 2))

    if edge_speed is not None and reached_goal_region:
        es = abs(float(edge_speed))
        if es > 1.1:
            edge_term -= 16.0 * _clamp(((es - 1.1) / 0.7) ** 2, 0.0, 1.5)
        elif 0.2 <= es <= 1.1 and not (fell or collided):
            edge_term += 5.0 * math.exp(-(((es - 0.6) / 0.35) ** 2))

    # 5. Impact & Tilt stability
    peak_impact = float(ep.get("peak_impact") or 0.0)
    excess_impact = max(0.0, peak_impact - 4.0)
    impact_penalty = -22.0 * _clamp((excess_impact / 14.0) ** 2, 0.0, 1.5)

    max_tilt_deg = float(ep.get("max_tilt_deg") or 0.0)
    excess_tilt = max(0.0, max_tilt_deg - 20.0)
    tilt_penalty = -25.0 * _clamp((excess_tilt / 22.0) ** 2, 0.0, 1.5)

    # 6. Command smoothness & per-tick trajectory quality (without punishing steering around obstacles)
    jerk = float(ep.get("jerk") or 0.0)
    jerk_penalty = -10.0 * _clamp((jerk / 3.5) ** 2, 0.0, 1.5)

    traj_term = 0.0
    if n > 0:
        sum_roll_sq = 0.0
        sum_pitch_excess_sq = 0.0
        sum_gyro_sq = 0.0
        sum_cmd_w_excess_sq = 0.0
        idle_ticks = 0
        reverse_ticks = 0
        cmd_v_flips = 0
        prev_v_sign = 0

        for s in states:
            roll = float(s.get("roll_deg", 0.0))
            pitch = float(s.get("pitch_deg", 0.0))
            x_pos = float(s.get("x", 0.0))
            cmd_v = float(s.get("cmd_v", 0.0))
            cmd_w = float(s.get("cmd_w", 0.0))
            gyro_mag = _mag(s.get("gyro", 0.0))

            sum_roll_sq += roll * roll
            p_ex = max(0.0, abs(pitch) - 20.0)
            sum_pitch_excess_sq += p_ex * p_ex
            sum_gyro_sq += gyro_mag * gyro_mag

            # Allow normal steering up to 1.2 rad/s for obstacle avoidance without penalty
            w_ex = max(0.0, abs(cmd_w) - 1.2)
            sum_cmd_w_excess_sq += w_ex * w_ex

            # Detect idling/parking before reaching goal
            if x_pos < 1.0 and abs(cmd_v) < 0.05 and abs(cmd_w) < 0.1:
                idle_ticks += 1

            if cmd_v < -0.05:
                reverse_ticks += 1

            v_sign = 1 if cmd_v > 0.05 else (-1 if cmd_v < -0.05 else 0)
            if v_sign != 0 and prev_v_sign != 0 and v_sign != prev_v_sign:
                cmd_v_flips += 1
            if v_sign != 0:
                prev_v_sign = v_sign

        rms_roll = math.sqrt(sum_roll_sq / n)
        rms_pitch_ex = math.sqrt(sum_pitch_excess_sq / n)
        rms_gyro = math.sqrt(sum_gyro_sq / n)
        rms_cmd_w_ex = math.sqrt(sum_cmd_w_excess_sq / n)
        idle_frac = idle_ticks / n
        rev_frac = reverse_ticks / n

        traj_term -= 18.0 * _clamp((rms_roll / 10.0) ** 2, 0.0, 1.5)
        traj_term -= 10.0 * _clamp((rms_pitch_ex / 12.0) ** 2, 0.0, 1.5)
        traj_term -= 5.0 * _clamp((rms_gyro / 100.0) ** 2, 0.0, 1.5)
        traj_term -= 5.0 * _clamp((rms_cmd_w_ex / 1.5) ** 2, 0.0, 1.5)

        # Explicitly penalize standing still / freezing when not at goal
        if not success:
            traj_term -= 35.0 * (idle_frac ** 1.5)
        elif idle_frac > 0.25:
            traj_term -= 15.0 * (idle_frac - 0.25)

        # Allow brief backing up (< 20% of episode) to unstick/clear a close pole, but penalize prolonged reverse or oscillation
        if rev_frac > 0.2:
            traj_term -= 15.0 * (rev_frac - 0.2)
        if cmd_v_flips > 2:
            traj_term -= 4.0 * min(5, cmd_v_flips - 2)

        # Reward finishing in a stable, upright, aligned posture near the centerline
        if success and not (fell or collided):
            last = states[-1]
            end_roll = abs(float(last.get("roll_deg", 0.0)))
            end_pitch = abs(float(last.get("pitch_deg", 0.0)))
            end_herr = abs(float(last.get("heading_err_deg", 0.0)))
            end_y = abs(float(last.get("y", 0.0)))
            if end_roll < 8.0 and end_pitch < 20.0:
                traj_term += 6.0 * (1.0 - end_roll / 10.0)
            traj_term += 6.0 * math.exp(-((end_herr / 20.0) ** 2)) * math.exp(-((end_y / 0.6) ** 2))

    total = (
        progress_reward
        + terminal_reward
        + clearance_term
        + edge_term
        + impact_penalty
        + tilt_penalty
        + jerk_penalty
        + traj_term
    )
    return float(total)
