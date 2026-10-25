"""Hand-written reference for obstacle/pedestrian twins (sanity check only; Argon writes the real reward)."""


def score(ep: dict) -> float:
    progress = min(ep["states"][-1]["x"] + 2.0, 5.0)
    clear = ep["min_clearance"] if ep["min_clearance"] is not None else 1.0
    return (progress + 10.0 * ep["success"] - 15.0 * ep["collided"] - 10.0 * ep["near_miss"]
            - 5.0 * ep["stalled"] + 2.0 * min(clear, 0.6) - 0.2 * ep["duration"])
