"""Correct a docking error using steps from the observed loaded walking leg."""

import numpy as np
from scipy.spatial.transform import Rotation


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def dock_reference(q, route, goal_xy, goal_yaw):
    xy = q[:2]
    delta = np.asarray(goal_xy) - xy
    error = float(np.linalg.norm(delta))
    heading = Rotation.from_quat(q[3:7], scalar_first=True).as_euler("ZYX")[0]
    # Find a steady forward portion of the actual G1 gait. Avoid its turn, grasp
    # and release. Reverse the joint motion for a correction behind the robot.
    span = min(62, len(route) - 1)
    starts = np.arange(len(route) - span)
    vecs = route[starts + span, :2] - route[starts, :2]
    rotations = Rotation.from_quat(route[starts, 3:7], scalar_first=True).as_euler("ZYX")[:, 0]
    forward = np.column_stack([np.cos(rotations), np.sin(rotations)])
    cost = np.sum(vecs * forward, axis=1) - 0.60 * np.abs(
        vecs[:, 0] * forward[:, 1] - vecs[:, 1] * forward[:, 0]
    )
    start = int(np.argmax(cost))
    seq = route[start : start + span + 1].copy()
    walk_heading = np.arctan2(vecs[start, 1], vecs[start, 0])
    requested = np.arctan2(delta[1], delta[0]) if error > 0.06 else goal_yaw
    if abs(wrap(requested - heading)) > np.pi / 2:
        seq = seq[::-1].copy()
        requested = wrap(requested + np.pi)
    target = heading + np.clip(wrap(requested - heading), -0.68, 0.68)
    if error < 0.095:
        target = heading + np.clip(wrap(goal_yaw - heading), -0.60, 0.60)
    strength = np.clip(error / max(0.15, np.linalg.norm(vecs[start])), 0.37, 1.12)
    seq[:, 7:19] = q[7:19] + strength * (seq[:, 7:19] - q[7:19])
    seq[:, 19:] = q[19:]
    seq[:, 3:7] = (
        Rotation.from_euler("z", wrap(target - walk_heading))
        * Rotation.from_quat(seq[:, 3:7], scalar_first=True)
    ).as_quat(scalar_first=True)
    pre = np.linspace(q, seq[0], 18)
    if q[3:7] @ seq[0, 3:7] < 0:
        pre[:, 3:7] = np.linspace(-q[3:7], seq[0, 3:7], 18)
    pre[:, 3:7] /= np.linalg.norm(pre[:, 3:7], axis=1)[:, None]
    # Settle onto the current standing leg configuration after a complete step pair.
    stop = seq[-1].copy()
    stop[7:19] = q[7:19]
    return np.vstack([pre, seq, np.linspace(seq[-1], stop, 22), np.tile(stop, (18, 1))])
