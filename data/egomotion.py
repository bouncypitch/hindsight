"""Visual ego-motion from the robot's own camera: how much the camera actually moved, frame to frame.

A telemetry glitch cannot fake this: if the operator commands forward and the camera image does not move for
seconds, the robot is really stuck or blocked. Median optical-flow magnitude is robust to people walking by.
"""
from __future__ import annotations

import subprocess

import cv2
import numpy as np

W, H = 112, 64  # tiny frames are plenty for global motion


def flow_series(video: str, start: float, dur: float, fps: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """-> (t, motion) where motion = median optical-flow magnitude in pixels/frame at W x H."""
    cmd = ["ffmpeg", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", video, "-t", f"{dur:.2f}",
           "-vf", f"fps={fps},scale={W}:{H},format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W)
    if len(frames) < 2:
        return np.zeros(0), np.zeros(0)
    mags = [0.0]
    for a, b in zip(frames[:-1], frames[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 2, 9, 2, 5, 1.1, 0)
        mags.append(float(np.median(np.linalg.norm(f, axis=2))))
    t = np.arange(len(frames)) / fps
    return t, np.array(mags)
