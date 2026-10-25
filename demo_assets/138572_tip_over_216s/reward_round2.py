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

    # 1. Forward progress & trajectory analysis
    if n > 0:
        start_x = _num(states[0], "x", -1.0)
        final_x = _num(states[-1], "x", start_x)
        max_x = max(_num(s, "x", start_x) for s in states)
        target_span = max(0.5, 1.25 - start_x)
        raw_disp = _clamp((final_x - start_x) / target_span, -0.5, 1.2)
        max_disp = _clamp((max_x - start_x) / target_span, 0.0, 1.2)
        backtrack_m = max(0.0, max_x - final_x)
        max_state_speed = max(abs(_num(s, "speed", 0.0)) for s in states)
    else:
        raw_disp = 1.0 if success else 0.0
        max_disp = 1.0 if success else 0.0
        backtrack_m = 0.0
        max_state_speed = 0.0

    edge_speed_val = max(0.0, _num(ep, "edge_speed", 0.0)) if edge_speed_raw is not None else 0.0
    peak_speed = max(edge_speed_val, max_state_speed)

    # Gate positive shaping on safe execution (no crash/fall, no reckless sprinting)
    safe_outcome = not fell and not collided and not near_miss
    speed_Factor = _clamp(1.0 - max(0.0, peak_speed - 0.85) / 0.35, 0.0, 1.0)
    progress_gate = (
        speed_Factor
        * (1.0 if success else _clamp(0.6 * raw_disp + 0.4 * max_disp, 0.0, 1.0))
        if safe_outcome
        else 0.0
    )

    total = 0.0
    # Discount progress heavily if the robot crashes or falls so charging into obstacles is never rewarded
    prog_scale = (0.3 + 0.7 * speed_Factor) if (not fell and not collided) else 0.1
    total += prog_scale * (22.0 * raw_disp + 8.0 * max_disp)
    total -= 15.0 * _clamp(backtrack_m, 0.0, 1.5)

    # 2. Terminal outcome hierarchy
    if success and not fell and not collided:
        # Scale success bonus down if the robot achieved it by sprinting recklessly (> 0.90 m/s)
        total += 100.0 * (0.35 + 0.65 * speed_Factor)
        # Only reward calm, reasonable completion times (4.0s - 8.0s); never incentivize sprinting < 3.5s
        if 3.8 <= duration <= 8.5:
            total += 6.0 * speed_Factor

    if fell:
        total -= 180.0
    if collided:
        total -= 180.0
    if near_miss:
        total -= 85.0
    if stalled:
        total -= 40.0
    if timeout and not success:
        total -= 30.0

    # 3. Speed discipline & curb edge alignment (prevents drag-racing hack and curb slams)
    if peak_speed > 0.80:
        excess_v = peak_speed - 0.80
        total -= 55.0 * excess_v + 95.0 * (excess_v ** 2)

    if edge_speed_raw is not None:
        if edge_speed_val < 0.2 and not success and not collided:
            total -= 8.0 * (0.2 - edge_speed_val) / 0.2
        if 0.25 <= edge_speed_val <= 0.78 and safe_outcome:
            center_err = abs(edge_speed_val - 0.50) / 0.28
            total += progress_gate * 8.0 * max(0.0, 1.0 - center_err ** 2)

    # Only enforce strict heading alignment when an actual curb/tilt transition occurs,
    # or keep penalty mild on flat obstacle-avoidance twins so the robot can steer around dogs/pedestrians.
    has_curb_or_slope = (max_tilt_deg > 3.0) or any(
        abs(_num(s, "dist_to_edge", 99.0)) < 0.5 for s in states
    )
    if edge_heading_raw is not None:
        edge_heading = abs(_num(ep, "edge_heading_deg", 0.0))
        if has_curb_or_slope:
            total -= 0.45 * edge_heading + 0.025 * (edge_heading ** 2)
            if edge_heading < 12.0 and safe_outcome:
                total += progress_gate * 8.0 * ((12.0 - edge_heading) / 12.0) ** 2
        else:
            # Allow up to 25 deg of evasive heading angle without penalty on flat ground
            if edge_heading > 25.0:
                total -= 0.2 * (edge_heading - 25.0)

    # 4. Impact & tilt stability
    if peak_impact > 2.5:
        imp_excess = peak_impact - 2.5
        total -= _clamp(3.0 * imp_excess + 0.28 * (imp_excess ** 2), 0.0, 95.0)
    if safe_outcome and peak_impact < 5.0:
        total += progress_gate * 8.0 * ((5.0 - peak_impact) / 5.0) ** 2

    if max_tilt_deg > 15.0:
        tilt_excess = max_tilt_deg - 15.0
        total -= _clamp(1.5 * tilt_excess + 0.10 * (tilt_excess ** 2), 0.0, 55.0)
    if safe_outcome and max_tilt_deg < 16.0:
        total += progress_gate * 5.0 * ((16.0 - max_tilt_deg) / 16.0)

    # 5. Command smoothness & anti-surging
    if jerk > 0.35:
        jerk_excess = jerk - 0.35
        total -= _clamp(10.0 * jerk_excess + 3.0 * (jerk_excess ** 2), 0.0, 45.0)
    if safe_outcome and jerk < 1.2:
        total += progress_gate * 8.0 * ((1.2 - jerk) / 1.2) ** 2

    # 6. Obstacle & pedestrian clearance (coupled with speed so passing fast & close is heavily punished)
    if min_clearance_raw is not None:
        clearance = max(0.0, _num(ep, "min_clearance", 0.0))
        safe_dist = 0.90
        if clearance < safe_dist:
            deficit = (safe_dist - clearance) / safe_dist
            speed_mult = 1.0 + 1.8 * max(0.0, peak_speed - 0.45)
            total -= speed_mult * (55.0 * (deficit ** 2))
            if clearance < 0.45:
                total -= speed_mult * (45.0 * ((0.45 - clearance) / 0.45))
        elif safe_outcome:
            total += progress_gate * 12.0 * _clamp((clearance - safe_dist) / 0.5, 0.0, 1.0)

    # 7. Per-tick trajectory analysis
    if n > 0:
        roll_pen_sum = 0.0
        pitch_pen_sum = 0.0
        edge_zone_pen_sum = 0.0
        edge_zone_count = 0
        gyro_pen_sum = 0.0
        reverse_pen_sum = 0.0
        cmd_osc_sum = 0.0
        surging_pen_sum = 0.0
        speed_pen_sum = 0.0
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

            obs_d = _num(s, "clearance", _num(s, "obs_dist", _num(s, "dist_to_obstacle", 99.0)))

            # Penalize overspeeding on every tick
            if abs(spd) > 0.78:
                ov = abs(spd) - 0.78
                speed_pen_sum += 8.0 * ov + 18.0 * (ov ** 2)

            # Extra penalty for moving fast when near an obstacle/dog
            if obs_d < 1.0 and abs(spd) > 0.45:
                speed_pen_sum += 12.0 * ((1.0 - obs_d) / 1.0) * (abs(spd) - 0.45)

            if roll > 3.0:
                roll_pen_sum += 0.35 * (roll - 3.0) + 0.04 * ((roll - 3.0) ** 2)
            if pitch > 15.0:
                pitch_pen_sum += 0.25 * (pitch - 15.0) + 0.03 * ((pitch - 15.0) ** 2)

            # Allow generous lateral room (up to 0.75 m) to steer around obstacles;
            # only penalize extreme corridor departure.
            if y_pos > 0.75:
                lat_drift_sum += 1.5 * (y_pos - 0.75)

            if dist_edge < 0.4:
                edge_zone_count += 1
                edge_zone_pen_sum += 0.25 * hdg_err + 0.4 * roll + 2.0 * abs(cmd_w)
                if abs(spd) > 0.80:
                    edge_zone_pen_sum += 10.0 * (abs(spd) - 0.80)

            if gyro_m > 1.2:
                gyro_pen_sum += 1.5 * (gyro_m - 1.2)

            # Strongly penalize reversing (in the real clip, reversing into the dog caused the flip)
            if cmd_v < -0.02 or spd < -0.02:
                rev_v = max(abs(min(0.0, cmd_v)), abs(min(0.0, spd)))
                reverse_pen_sum += 8.0 * rev_v

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
                if d_cw > 1.2:
                    cmd_osc_sum += 0.5 * (d_cw - 1.2)
                if d_spd > 1.5:
                    surging_pen_sum += 0.8 * (d_spd - 1.5)
                if (cur_x - prev_x) < -0.003:
                    reverse_pen_sum += 20.0 * abs(cur_x - prev_x)

        inv_n = 1.0 / float(n)
        total -= _clamp(speed_pen_sum * inv_n * 8.0, 0.0, 45.0)
        total -= _clamp(roll_pen_sum * inv_n * 6.0, 0.0, 25.0)
        total -= _clamp(pitch_pen_sum * inv_n * 5.0, 0.0, 20.0)
        total -= _clamp(lat_drift_sum * inv_n * 4.0, 0.0, 12.0)
        total -= _clamp(gyro_pen_sum * inv_n * 4.0, 0.0, 15.0)
        total -= _clamp(reverse_pen_sum * inv_n * 8.0, 0.0, 30.0)
        total -= _clamp(cmd_osc_sum * inv_n * 3.5, 0.0, 20.0)
        total -= _clamp(surging_pen_sum * inv_n * 4.0, 0.0, 20.0)

        if edge_zone_count > 0:
            total -= _clamp((edge_zone_pen_sum / float(edge_zone_count)) * 2.5, 0.0, 20.0)

        mean_pen = (roll_pen_sum + cmd_osc_sum + surging_pen_sum + reverse_pen_sum + speed_pen_sum) * inv_n
        if mean_pen < 0.6 and safe_outcome:
            total += progress_gate * 8.0 * ((0.6 - mean_pen) / 0.6)

    return float(total)
