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

    peak_impact = float(ep.get("peak_impact") or 0.0)
    hard_contact = collided or (peak_impact > 10.0)

    # 1. Forward progress toward x = 1.2
    if n > 0:
        x0 = float(states[0].get("x", 0.0))
        x_end = float(states[-1].get("x", x0))
        target_dist = max(0.5, 1.2 - x0)
        raw_progress = _clamp((x_end - x0) / target_dist, -0.3, 1.1)
    else:
        raw_progress = 1.0 if success else 0.0

    # Prevent farming progress by creeping up to the obstacle and timing out
    if fell or hard_contact:
        progress_mult = 0.0
    elif not success:
        progress_mult = 0.3
    else:
        progress_mult = 1.0

    progress_reward = 45.0 * max(0.0, raw_progress) * progress_mult
    if raw_progress < 0.0:
        progress_reward += 25.0 * raw_progress

    # 2. Terminal outcome rewards and penalties
    terminal_reward = 0.0
    if success and not hard_contact:
        terminal_reward += 80.0
        duration = float(ep.get("duration") or 0.0)
        if duration > 0.0:
            terminal_reward += 15.0 * _clamp((7.5 - duration) / 5.0, 0.0, 1.0)

    if fell:
        terminal_reward -= 90.0
    if collided:
        terminal_reward -= 85.0
    if near_miss:
        terminal_reward -= 30.0
    if stalled:
        terminal_reward -= 60.0
    if not success and not (fell or collided or stalled):
        # Close the timeout loophole so parking in front of the obstacle is heavily penalized
        terminal_reward -= 60.0

    # 3. Obstacle clearance (encourage steering around the obstacle with a safe margin)
    min_clearance = ep.get("min_clearance", None)
    clearance_term = 0.0
    if min_clearance is not None:
        mc = float(min_clearance)
        safe_dist = 0.5
        if mc < safe_dist:
            shortfall = (safe_dist - max(0.0, mc)) / safe_dist
            clearance_term -= 50.0 * (shortfall * shortfall)
        elif success and not hard_contact:
            clearance_term += 12.0 * _clamp((mc - safe_dist) / 0.5, 0.0, 1.0)

    # 4. Impact & Tilt stability (heavily penalize unflagged bumps/scrapes)
    excess_impact = max(0.0, peak_impact - 2.5)
    impact_penalty = -65.0 * _clamp((excess_impact / 15.0) ** 1.5, 0.0, 1.5)

    max_tilt_deg = float(ep.get("max_tilt_deg") or 0.0)
    excess_tilt = max(0.0, max_tilt_deg - 20.0)
    tilt_penalty = -25.0 * _clamp((excess_tilt / 22.0) ** 2, 0.0, 1.5)

    # 5. Command smoothness & trajectory quality (do NOT penalize lateral y offset needed for avoidance)
    jerk = float(ep.get("jerk") or 0.0)
    jerk_penalty = -14.0 * _clamp((jerk / 3.0) ** 2, 0.0, 1.5)

    traj_term = 0.0
    if n > 0:
        sum_roll_sq = 0.0
        sum_pitch_excess_sq = 0.0
        sum_gyro_sq = 0.0
        reverse_ticks = 0
        crawl_ticks = 0

        for s in states:
            roll = float(s.get("roll_deg", 0.0))
            pitch = float(s.get("pitch_deg", 0.0))
            cmd_v = float(s.get("cmd_v", 0.0))
            vx = float(s.get("vx", cmd_v))
            gyro_mag = _mag(s.get("gyro", 0.0))

            sum_roll_sq += roll * roll
            p_ex = max(0.0, abs(pitch) - 20.0)
            sum_pitch_excess_sq += p_ex * p_ex
            sum_gyro_sq += gyro_mag * gyro_mag
            if cmd_v < -0.05:
                reverse_ticks += 1
            if abs(vx) < 0.15 and cmd_v < 0.2:
                crawl_ticks += 1

        rms_roll = math.sqrt(sum_roll_sq / n)
        rms_pitch_ex = math.sqrt(sum_pitch_excess_sq / n)
        rms_gyro = math.sqrt(sum_gyro_sq / n)
        rev_frac = reverse_ticks / n
        crawl_frac = crawl_ticks / n

        traj_term -= 18.0 * _clamp((rms_roll / 10.0) ** 2, 0.0, 1.5)
        traj_term -= 10.0 * _clamp((rms_pitch_ex / 12.0) ** 2, 0.0, 1.5)
        traj_term -= 6.0 * _clamp((rms_gyro / 110.0) ** 2, 0.0, 1.5)
        traj_term -= 20.0 * rev_frac
        traj_term -= 25.0 * max(0.0, crawl_frac - 0.15)

        if success and not (fell or hard_contact):
            last = states[-1]
            end_roll = abs(float(last.get("roll_deg", 0.0)))
            end_pitch = abs(float(last.get("pitch_deg", 0.0)))
            if end_roll < 8.0 and end_pitch < 20.0:
                traj_term += 8.0 * (1.0 - end_roll / 10.0)

    total = (
        progress_reward
        + terminal_reward
        + clearance_term
        + impact_penalty
        + tilt_penalty
        + jerk_penalty
        + traj_term
    )
    return float(total)
