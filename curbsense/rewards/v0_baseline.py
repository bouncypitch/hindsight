"""v0: the reward a busy engineer writes first. Get across fast, don't flip.

Contract every reward module follows (Argon writes new ones in the same shape):
    score(ep: dict) -> float
ep has: outcome ("success" | "fell" | "stalled" | "collided" | "near_miss" | "timeout"), success, fell,
        stalled, collided, near_miss, min_clearance (m to the obstacle/person, None if none), fail_t,
        duration, edge_speed, edge_heading_deg, peak_impact, max_tilt_deg, jerk (mean |command change|/s,
        high = stop-go or twitchy steering), and states (list of per-tick dicts: t, x, y, z, roll_deg, pitch_deg,
        yaw_deg, speed, dist_to_edge, heading_err_deg, accel, gyro, cmd_v, cmd_w).
"""


def score(ep: dict) -> float:
    progress = min(ep["states"][-1]["x"] + 2.0, 3.5)
    return progress + 10.0 * ep["success"] - 10.0 * ep["fell"] - 0.5 * ep["duration"]


# Scenarios: "drop" (down off a curb), "climb" (up onto a curb), "slope" (uphill grade), "obstacle" (bollard /
# pole / bin in the path), "pedestrian" (a person crossing the path). x = 0 is the curb edge / slope start; the
# robot starts at x < 0 and succeeds once it is past the curb, slope or obstacle without hitting anything.
