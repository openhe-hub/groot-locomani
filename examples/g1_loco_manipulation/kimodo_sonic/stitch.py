"""Join generated G1 clips in heading-aligned coordinates with smooth overlaps."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def join(previous, following, overlap):
    """Align at overlap onset, then crossfade translation, joints and root rotation."""
    tail = previous[-overlap:]
    next_clip = following.copy()
    rotations = Rotation.from_quat(next_clip[:, 3:7], scalar_first=True)
    yaw = Rotation.from_quat(tail[0, 3:7], scalar_first=True).as_euler("ZYX")[0]
    align = Rotation.from_euler("z", yaw - rotations[0].as_euler("ZYX")[0])
    next_clip[:, :2] -= next_clip[0, :2].copy()
    next_clip[:, :3] = align.apply(next_clip[:, :3])
    next_clip[:, :2] += tail[0, :2]
    next_clip[:, 3:7] = (align * rotations).as_quat(scalar_first=True)
    # Continue forward travel as the outgoing walk slows into the stationary clip.
    advance = 0.5 * (tail[-1, :2] - tail[0, :2])
    next_clip[:, :2] += advance
    t = np.linspace(0, 1, overlap)
    weights = t**3 * (10 - 15 * t + 6 * t * t)
    blend = (1 - weights[:, None]) * tail + weights[:, None] * next_clip[:overlap]
    for i, w in enumerate(weights):
        pair = np.stack([tail[i, 3:7], next_clip[i, 3:7]])
        blend[i, 3:7] = Slerp([0, 1], Rotation.from_quat(pair, scalar_first=True))(w).as_quat(
            scalar_first=True
        )
    return np.concatenate((previous[:-overlap], blend, next_clip[overlap:]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--sequence", choices=["sequence", "greeting", "mobility"], default="sequence")
    args = p.parse_args()
    fps, overlap = 30, 18
    phases = []
    result = None
    # Use complete gait cycles in the middle of the walking clip.
    clips = [("stand", 0, 60), ("walk", 9, 114), ("squat", 0, 150), ("stand", 0, 60)]
    if args.sequence == "greeting":
        clips = [("walk", 6, 88), ("wave", 0, 120), ("turn", 0, 120), ("stand", 0, 55)]
    elif args.sequence == "mobility":
        clips = [("sidestep", 6, 90), ("reach", 0, 120), ("backward", 6, 90), ("clap", 0, 120)]
    for name, start, end in clips:
        folder = args.root / name
        cur = np.loadtxt(folder / "reference.csv", delimiter=",")[start:end]
        prompt = json.loads((folder / "prompt.json").read_text())["prompt_phases"][0][0]
        if result is None:
            result = cur
        else:
            result = join(result, cur, overlap)
        phases.append([prompt, (end - start) / fps])
    # The displayed phase switches at the midpoint of each blend.
    for i in range(len(phases) - 1):
        phases[i][1] -= overlap / (2 * fps)
        phases[i + 1][1] -= overlap / (2 * fps)
    folder = args.root / args.sequence
    folder.mkdir(exist_ok=True)
    np.savetxt(folder / "reference.csv", result, delimiter=",", fmt="%.9f")
    meta = dict(
        model="Kimodo-G1-RP-v1",
        fps=fps,
        prompt_phases=phases,
        source_clips=clips,
        overlap_frames=overlap,
        method="SE(2) alignment, quintic crossfade and root quaternion SLERP",
    )
    (folder / "prompt.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
