import math


def _num(d, key, default=0.0):
    if not isinstance(d, dict):
        return float(default)
    v = d.get(key, default)
    if v is None:
        return float(default)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    if math.isnan(f) or math.isinf(f):
        return float(default)
    return f


def _mag(v):
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        f = float(v)
        return 0.0 if (math.isnan(f) or math.isinf(f)) else abs(f)
    if isinstance(v, (list, tuple)):
        vals = [_mag(c) for c in v]
        return math.sqrt(sum(c * c for c in vals))
    if isinstance(v, dict):
        vals = [_mag(c) for c in v.values()]
        return math.sqrt(sum(c * c for c in vals))
    return 0.0


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def score(ep: dict) -> float:
    outcome = ep.get("outcome", "")
    success = bool(ep.get("success", False) or outcome == "success")
    fell = bool(ep.get("fell", False) or outcome == "fell")
    collided = bool(ep.get("collided", False) or outcome == "collided")
    near_miss = bool(ep.get("near_miss", False) or outcome == "near_miss")
    stalled = bool(ep.get("stalled", False) or outcome == "stalled")
    timeout = bool(outcome == "timeout" or (not success and not fell and not collided and not stalled))

    duration = max(0.0, _num(ep, "duration", 0.0))
    peak_impact = max(0.0, _num(ep, "peak_impact", 0.0))
    max_tilt_deg = max(0.0, _num(ep, "max_tilt_deg", 0.0))
    jerk = max(0.0, _num(ep, "jerk", 0.0))

    edge_speed_raw = ep.get("edge_speed", None)
    edge_heading_raw = ep.get("edge_heading_deg", None)
    min_clearance_raw = ep.get("min_clearance", None)

    states = ep.get("states") or []
    n = len(states)

    # Treat severe impact or zero clearance as an effective collision
    min_clearance = _num(ep, "min_clearance", 99.0) if min_clearance_raw is not None else 99.0
    effective_collision = collided or (peak_impact > 18.0) or (min_clearance_raw is not None and min_clearance < 0.08)
    clean_success = success and not fell and not effective_collision and not stalled and not near_miss

    # 1. Forward progress & trajectory analysis
    if n > 0:
        start_x = _num(states[0], "x", -1.0)
        final_x = _num(states[-1], "x", start_x)
        max_x = max(_num(s, "x", start_x) for s in states)
        target_span = max(0.5, 1.25 - start_x)
        raw_disp = _clamp((final_x - start_x) / target_span, -0.5, 1.2)
        max_disp = _clamp((max_x - start_x) / target_span, 0.0, 1.2)
        backtrack_m = max(0.0, max_x - final_x)
    else:
        raw_disp = 1.0 if success else 0.0
        max_disp = 1.0 if success else 0.0
        backtrack_m = 0.0

    # Discount progress shaping if the robot stalls, collides, or fails to clear the obstacle
    clearance_factor = 1.0
    if min_clearance_raw is not None and min_clearance < 0.6:
        clearance_factor = _clamp((min_clearance - 0.1) / 0.5, 0.0, 1.0)

    total = 0.0
    if clean_success:
        total += 35.0 * raw_disp + 10.0 * max_disp
    else:
        # Modest shaping when exploring so creeping up to an obstacle cannot offset failure penalties
        total += clearance_factor * (12.0 * raw_disp + 4.0 * max_disp)
    total -= 15.0 * _clamp(backtrack_m, 0.0, 1.5)

    # 2. Strict terminal outcome hierarchy (stalling/timing out cannot be used to dodge collision penalty)
    if clean_success:
        total += 120.0
        if duration > 0.0:
            time_bonus = _clamp(16.0 - 1.6 * max(0.0, duration - 2.5), 0.0, 16.0)
            total += time_bonus
    elif success:
        total += 40.0

    if fell:
        total -= 150.0
    if effective_collision:
        total -= 140.0
    if near_miss:
        total -= 75.0
    if stalled:
        total -= 85.0
    if timeout and not success:
        total -= 75.0

    # 3. Obstacle clearance (critical for steering around cylinders/barriers/cones)
    if min_clearance_raw is not None:
        safe_dist = 0.75
        if min_clearance < safe_dist:
            deficit = (safe_dist - min_clearance) / safe_dist
            total -= 55.0 * (deficit ** 2)
            if min_clearance < 0.35:
                total -= 50.0 * ((0.35 - min_clearance) / 0.35)
        elif clean_success:
            total += 12.0 * _clamp((min_clearance - safe_dist) / 0.5, 0.0, 1.0)

    # 4. Curb / edge speed & heading (only enforce strict heading when not executing an obstacle bypass)
    has_close_obstacle = (min_clearance_raw is not None and min_clearance < 1.2)
    if edge_heading_raw is not None:
        edge_heading = abs(_num(ep, "edge_heading_deg", 0.0))
        allowed_hdg = 28.0 if has_close_obstacle else 12.0
        if edge_heading > allowed_hdg:
            excess_h = edge_heading - allowed_hdg
            total -= 0.4 * excess_h + 0.02 * (excess_h ** 2)
        elif clean_success:
            total += 6.0 * ((allowed_hdg - edge_heading) / allowed_hdg) ** 2

    if edge_speed_raw is not None:
        edge_speed = max(0.0, _num(ep, "edge_speed", 0.0))
        if edge_speed > 0.85:
            excess_v = edge_speed - 0.85
            total -= 20.0 * excess_v + 25.0 * (excess_v ** 2)
        elif edge_speed < 0.25 and not clean_success:
            total -= 15.0 * (0.25 - edge_speed) / 0.25
        if 0.25 <= edge_speed <= 0.85 and clean_success:
            center_err = abs(edge_speed - 0.55) / 0.30
            total += 8.0 * max(0.0, 1.0 - center_err ** 2)

    # 5. Impact & tilt stability (all positive bonuses gated on clean_success)
    if peak_impact > 2.5:
        imp_excess = peak_impact - 2.5
        total -= _clamp(2.8 * imp_excess + 0.25 * (imp_excess ** 2), 0.0, 90.0)
    elif clean_success:
        total += 8.0 * ((2.5 - peak_impact) / 2.5) ** 2

    if max_tilt_deg > 15.0:
        tilt_excess = max_tilt_deg - 15.0
        total -= _clamp(1.2 * tilt_excess + 0.08 * (tilt_excess ** 2), 0.0, 45.0)
    elif clean_success:
        total += 5.0 * ((15.0 - max_tilt_deg) / 15.0)

    # 6. Command smoothness (bonus gated on clean_success so sitting still doesn't farm points)
    if jerk > 0.35:
        jerk_excess = jerk - 0.35
        total -= _clamp(9.0 * jerk_excess + 2.5 * (jerk_excess ** 2), 0.0, 40.0)
    if jerk < 1.5 and clean_success:
        total += 8.0 * ((1.5 - jerk) / 1.5) ** 2

    # 7. Per-tick trajectory analysis
    if n > 0:
        roll_pen_sum = 0.0
        pitch_pen_sum = 0.0
        gyro_pen_sum = 0.0
        reverse_pen_sum = 0.0
        cmd_osc_sum = 0.0
        surging_pen_sum = 0.0
        idle_pen_sum = 0.0
        obs_linger_pen_sum = 0.0

        for i, s in enumerate(states):
            roll = abs(_num(s, "roll_deg", 0.0))
            pitch = abs(_num(s, "pitch_deg", 0.0))
            spd = _num(s, "speed", 0.0)
            cmd_v = _num(s, "cmd_v", spd)
            cmd_w = _num(s, "cmd_w", 0.0)
            gyro_m = _mag(s.get("gyro", 0.0))
            s_clr = _num(s, "clearance", _num(s, "min_clearance", min_clearance))

            if roll > 3.0:
                roll_pen_sum += 0.35 * (roll - 3.0) + 0.04 * ((roll - 3.0) ** 2)
            if pitch > 15.0:
                pitch_pen_sum += 0.25 * (pitch - 15.0) + 0.03 * ((pitch - 15.0) ** 2)
            if gyro_m > 1.2:
                gyro_pen_sum += 1.5 * (gyro_m - 1.2)
            if cmd_v < -0.03:
                reverse_pen_sum += 4.0 * abs(cmd_v)

            # Penalize hesitation / creeping / stalling in place
            if spd < 0.12 and not clean_success:
                idle_pen_sum += 1.5 * ((0.12 - max(0.0, spd)) / 0.12)

            # Penalize lingering close to an obstacle without turning to clear it
            if s_clr < 0.6:
                obs_linger_pen_sum += 2.0 * ((0.6 - max(0.0, s_clr)) / 0.6)
                if spd < 0.15 and abs(cmd_w) < 0.2:
                    obs_linger_pen_sum += 2.5

            if i > 0:
                prev = states[i - 1]
                dt = max(1e-3, _num(s, "t", i * 0.05) - _num(prev, "t", (i - 1) * 0.05))
                prev_cv = _num(prev, "cmd_v", _num(prev, "speed", 0.0))
                prev_cw = _num(prev, "cmd_w", 0.0)
                prev_spd = _num(prev, "speed", 0.0)
                prev_x = _num(prev, "x", 0.0)
                cur_x = _num(s, "x", 0.0)

                d_cv = abs(cmd_v - prev_cv) / dt
                d_cw = abs(cmd_w - prev_cw) / dt
                d_spd = abs(spd - prev_spd) / dt

                if d_cv > 0.8:
                    cmd_osc_sum += 0.8 * (d_cv - 0.8)
                if d_cw > 1.5:
                    cmd_osc_sum += 0.4 * (d_cw - 1.5)
                if d_spd > 1.5:
                    surging_pen_sum += 0.7 * (d_spd - 1.5)
                if (cur_x - prev_x) < -0.003:
                    reverse_pen_sum += 15.0 * abs(cur_x - prev_x)

        inv_n = 1.0 / float(n)
        total -= _clamp(roll_pen_sum * inv_n * 6.0, 0.0, 25.0)
        total -= _clamp(pitch_pen_sum * inv_n * 5.0, 0.0, 20.0)
        total -= _clamp(gyro_pen_sum * inv_n * 4.0, 0.0, 15.0)
        total -= _clamp(reverse_pen_sum * inv_n * 6.0, 0.0, 20.0)
        total -= _clamp(cmd_osc_sum * inv_n * 3.5, 0.0, 20.0)
        total -= _clamp(surging_pen_sum * inv_n * 4.0, 0.0, 20.0)
        total -= _clamp(idle_pen_sum * inv_n * 18.0, 0.0, 25.0)
        total -= _clamp(obs_linger_pen_sum * inv_n * 12.0, 0.0, 30.0)

        mean_pen = (roll_pen_sum + cmd_osc_sum + surging_pen_sum + reverse_pen_sum + idle_pen_sum) * inv_n
        if mean_pen < 0.8 and clean_success:
            total += 10.0 * ((0.8 - mean_pen) / 0.8)

    return float(total)
