"""Hand-written reference for a 'safe descent' reward. Sanity check / fallback only;
in the demo Argon writes this kind of reward from the good/bad videos."""


def score(ep: dict) -> float:
    progress = min(ep["states"][-1]["x"] + 2.0, 3.5)
    impact_pen = max(0.0, ep["peak_impact"] - 25.0) / 10.0     # cargo shock above ~2.5 g
    tilt_pen = max(0.0, ep["max_tilt_deg"] - 30.0) / 10.0
    edge_pen = max(0.0, ep["edge_speed"] - 0.8) * 3.0           # don't launch off curbs
    return progress + 10.0 * ep["success"] - 15.0 * ep["fell"] - impact_pen - tilt_pen - edge_pen - 0.1 * ep["duration"]
