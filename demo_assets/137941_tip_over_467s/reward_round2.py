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
    timeout = bool(outcome == "timeout")

    duration = max(0.0, _num(ep, "duration", 0.0))
    peak_impact = max(0.0, _num(ep, "peak_impact", 0.0))
    max_tilt_deg = max(0.0, _num(ep, "max_tilt_deg", 0.0))
    jerk = max(0.0, _num(ep, "jerk", 0.0))

    edge_speed_raw = ep.get("edge_speed", None)
    edge_heading_raw = ep.get("edge_heading_deg", None)
    min_clearance_raw = ep.get("min_clearance", None)

    states = ep.get("states") or []
    n = len(states)

    # 1. Forward progress, curb clearance, and approach momentum analysis
    start_x = -1.0
    final_x = start_x
    max_x = start_x
    max_spd = 0.0
    approach_spd = 0.0
    curb_drive_cmd = 0.0

    if n > 0:
        start_x = _num(states[0], "x", -1.0)
        final_x = _num(states[-1], "x", start_x)
        max_x = max(_num(s, "x", start_x) for s in states)
        target_span = max(0.5, 1.25 - start_x)
        raw_disp = _clamp((final_x - start_x) / target_span, -0.5, 1.2)
        max_disp = _clamp((max_x - start_x) / target_span, 0.0, 1.2)
        backtrack_m = max(0.0, max_x - final_x)

        max_spd = max(max(0.0, _num(s, "speed", 0.0)) for s in states)
        t0 = _num(states[0], "t", 0.0)
        t_reach = t0
        for s in states:
            if _num(s, "x", start_x) >= max_x - 0.02:
                t_reach = _num(s, "t", t0)
                break
        dt_reach = max(0.2, t_reach - t0)
        approach_spd = max(0.0, (max_x - start_x) / dt_reach)

        curb_cmds = [
            max(0.0, _num(s, "cmd_v", _num(s, "speed", 0.0)))
            for s in states
            if abs(_num(s, "dist_to_edge", 99.0)) < 0.45
        ]
        if curb_cmds:
            curb_drive_cmd = sum(curb_cmds) / float(len(curb_cmds))
    else:
        raw_disp = 1.0 if success else 0.0
        max_disp = 1.0 if success else 0.0
        backtrack_m = 0.0

    # Only grant comfort/smoothness bonuses when actually succeeding (never on stalls/falls)
    bonus_gate = 1.0 if success else 0.0

    total = 0.0
    # Dense progress shaping + strong bonus for getting center of mass past the curb (x > 0)
    total += 40.0 * raw_disp + 20.0 * max_disp
    if max_x > 0.02:
        total += 30.0 * _clamp((max_x - 0.02) / 0.6, 0.0, 1.0)
    if final_x > 0.05:
        total += 20.0 * _clamp((final_x - 0.05) / 0.6, 0.0, 1.0)
    total -= 12.0 * _clamp(backtrack_m, 0.0, 1.5)

    # Break the curb-stall local optimum: reward building approach speed and drive power
    if not success and not fell and not collided:
        total += 18.0 * _clamp(approach_spd / 0.80, 0.0, 1.0)
        total += 12.0 * _clamp(max_spd / 0.95, 0.0, 1.0)
        total += 10.0 * _clamp(curb_drive_cmd / 0.85, 0.0, 1.0)
        if stalled:
            # Higher contact impulse on a stalled curb attempt indicates stronger climb momentum
            total += 15.0 * _clamp(peak_impact / 75.0, 0.0, 1.0)

    # 2. Terminal outcome hierarchy
    if success:
        total += 130.0
        if duration > 0.0:
            time_bonus = _clamp(16.0 - 1.6 * max(0.0, duration - 2.2), 0.0, 16.0)
            total += time_bonus

    if fell:
        total -= 120.0
    if collided:
        total -= 120.0
    if near_miss:
        total -= 50.0
    if stalled:
        total -= 65.0
    if timeout and not success:
        total -= 40.0

    # 3. Curb edge alignment & transition speed control
    if edge_heading_raw is not None:
        edge_heading = abs(_num(ep, "edge_heading_deg", 0.0))
        total -= 0.45 * edge_heading + 0.025 * (edge_heading ** 2)
        if edge_heading < 12.0 and not fell:
            total += bonus_gate * 10.0 * ((12.0 - edge_heading) / 12.0) ** 2

    if edge_speed_raw is not None:
        edge_speed = max(0.0, _num(ep, "edge_speed", 0.0))
        # Allow up to 1.05 m/s so up-curb climbs have sufficient momentum
        if edge_speed > 1.05:
            excess_v = edge_speed - 1.05
            total -= 16.0 * excess_v + 18.0 * (excess_v ** 2)
        elif edge_speed < 0.35 and not success:
            total -= 15.0 * (0.35 - edge_speed) / 0.35
        if 0.35 <= edge_speed <= 1.05 and not fell:
            center_err = abs(edge_speed - 0.70) / 0.35
            total += bonus_gate * 10.0 * max(0.0, 1.0 - center_err ** 2)

    # 4. Impact & tilt stability (calibrated to sim contact scale where gentle curb touch is ~45-55)
    if not stalled and peak_impact > 75.0:
        imp_excess = peak_impact - 75.0
        total -= _clamp(0.6 * imp_excess + 0.015 * (imp_excess ** 2), 0.0, 40.0)
    if success and not fell and peak_impact < 80.0:
        total += 8.0 * ((80.0 - peak_impact) / 80.0)

    # Short-wheelbase curb climbs geometrically require ~20-28 deg of pitch
    if max_tilt_deg > 30.0:
        tilt_excess = max_tilt_deg - 30.0
        total -= _clamp(1.2 * tilt_excess + 0.08 * (tilt_excess ** 2), 0.0, 45.0)
    if success and not fell and max_tilt_deg < 28.0:
        total += 6.0 * ((28.0 - max_tilt_deg) / 28.0)

    # 5. Command smoothness (relaxed deadband so step-climb contact doesn't get penalized)
    if not stalled and jerk > 2.5:
        jerk_excess = jerk - 2.5
        total -= _clamp(4.0 * jerk_excess + 0.8 * (jerk_excess ** 2), 0.0, 30.0)
    if success and jerk < 2.5:
        total += 8.0 * ((2.5 - jerk) / 2.5) ** 2

    # 6. Obstacle & pedestrian clearance
    if min_clearance_raw is not None:
        clearance = max(0.0, _num(ep, "min_clearance", 0.0))
        safe_dist = 0.85
        if clearance < safe_dist:
            deficit = (safe_dist - clearance) / safe_dist
            total -= 30.0 * (deficit ** 2)
            if clearance < 0.35:
                total -= 25.0 * ((0.35 - clearance) / 0.35)
        else:
            total += bonus_gate * 8.0 * _clamp((clearance - safe_dist) / 0.6, 0.0, 1.0)

    # 7. Per-tick trajectory analysis
    if n > 0:
        roll_pen_sum = 0.0
        pitch_pen_sum = 0.0
        heading_pen_sum = 0.0
        edge_zone_pen_sum = 0.0
        edge_zone_count = 0
        gyro_pen_sum = 0.0
        reverse_pen_sum = 0.0
        cmd_osc_sum = 0.0
        surging_pen_sum = 0.0
        lat_drift_sum = 0.0

        for i, s in enumerate(states):
            roll = abs(_num(s, "roll_deg", 0.0))
            pitch = abs(_num(s, "pitch_deg", 0.0))
            hdg_err = abs(_num(s, "heading_err_deg", 0.0))
            dist_edge = abs(_num(s, "dist_to_edge", 99.0))
            spd = _num(s, "speed", 0.0)
            cmd_v = _num(s, "cmd_v", spd)
            cmd_w = _num(s, "cmd_w", 0.0)
            y_pos = abs(_num(s, "y", 0.0))
            gyro_m = _mag(s.get("gyro", 0.0))

            # Sideways roll must stay small to prevent diagonal curb flips
            if roll > 5.0:
                roll_pen_sum += 0.4 * (roll - 5.0) + 0.05 * ((roll - 5.0) ** 2)

            # Allow up to 30 deg pitch near the curb step and 20 deg elsewhere
            pitch_limit = 30.0 if dist_edge < 0.5 else 20.0
            if pitch > pitch_limit:
                pitch_pen_sum += 0.25 * (pitch - pitch_limit) + 0.03 * ((pitch - pitch_limit) ** 2)

            if hdg_err > 6.0:
                heading_pen_sum += 0.12 * (hdg_err - 6.0)
            if y_pos > 0.25:
                lat_drift_sum += 2.0 * (y_pos - 0.25)

            if dist_edge < 0.4:
                edge_zone_count += 1
                edge_zone_pen_sum += 0.25 * hdg_err + 0.45 * roll + 2.0 * abs(cmd_w)
                if spd > 1.15:
                    edge_zone_pen_sum += 6.0 * (spd - 1.15)

            # Allow brief pitch rate during step climb; penalize excessive tumbling
            gyro_limit = 3.0 if dist_edge < 0.45 else 1.8
            if gyro_m > gyro_limit:
                gyro_pen_sum += 1.2 * (gyro_m - gyro_limit)

            if cmd_v < -0.05:
                reverse_pen_sum += 4.0 * abs(cmd_v)

            # Only evaluate command oscillation / surging away from the curb contact zone
            if i > 0 and dist_edge >= 0.35:
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

                if d_cv > 1.5:
                    cmd_osc_sum += 0.6 * (d_cv - 1.5)
                if d_cw > 1.8:
                    cmd_osc_sum += 0.4 * (d_cw - 1.8)
                if d_spd > 2.5:
                    surging_pen_sum += 0.5 * (d_spd - 2.5)
                if (cur_x - prev_x) < -0.005:
                    reverse_pen_sum += 12.0 * abs(cur_x - prev_x)

        inv_n = 1.0 / float(n)
        total -= _clamp(roll_pen_sum * inv_n * 6.0, 0.0, 25.0)
        total -= _clamp(pitch_pen_sum * inv_n * 5.0, 0.0, 20.0)
        total -= _clamp(heading_pen_sum * inv_n * 4.0, 0.0, 15.0)
        total -= _clamp(lat_drift_sum * inv_n * 4.0, 0.0, 12.0)
        total -= _clamp(gyro_pen_sum * inv_n * 4.0, 0.0, 15.0)
        total -= _clamp(reverse_pen_sum * inv_n * 6.0, 0.0, 20.0)
        total -= _clamp(cmd_osc_sum * inv_n * 3.5, 0.0, 20.0)
        total -= _clamp(surging_pen_sum * inv_n * 4.0, 0.0, 20.0)

        if edge_zone_count > 0:
            total -= _clamp((edge_zone_pen_sum / float(edge_zone_count)) * 2.5, 0.0, 20.0)

        mean_pen = (roll_pen_sum + cmd_osc_sum + surging_pen_sum + reverse_pen_sum) * inv_n
        if success and mean_pen < 0.8 and not fell and not collided:
            total += 8.0 * ((0.8 - mean_pen) / 0.8)

    return float(total)
