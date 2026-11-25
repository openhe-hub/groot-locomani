"""Run the matched soap trials and summarize physical outcomes for every generated seed."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from kitchen_task import prepare
import numpy as np
from scipy.spatial.transform import Rotation
from track import rollout


# World XY of the pelvis when the arm control starts. Calibrated from the previously
# released successful demonstration at 2.65 s, before generating this paired study.
DOCK_XY = np.array([1.94, 0.72])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("--stand", type=Path, required=True)
    p.add_argument("--zero", type=Path, default=Path("zero"))
    p.add_argument(
        "--robot",
        type=Path,
        default=Path("GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"),
    )
    p.add_argument(
        "--reference-reset",
        action="store_true",
        help="Initialize the joints from each generated reference, as in the historical demo",
    )
    args = p.parse_args()
    # Use the original soap rollout's starting joint configuration, base orientation
    # and XY reset for every arm and seed, before applying the shared room offset.
    reset_pose = np.loadtxt(args.stand.parent.parent / "walk/reference.csv", delimiter=",")[
        4
    ].copy()
    reset_pose[:2] = 0
    rot = Rotation.from_quat(reset_pose[3:7], scalar_first=True)
    reset_pose[3:7] = (Rotation.from_euler("z", -rot.as_euler("ZYX")[0]) * rot).as_quat(
        scalar_first=True
    )
    results = []
    for seed in range(42, 50):
        constraint = json.loads(
            (args.study / str(seed) / "pelvis/motions/walk/constraints.json").read_text()
        )[0]
        frames = constraint["frame_indices"]
        goal = np.asarray(constraint["smooth_root_2d"])[:, ::-1]
        for arm in ("text", "pelvis"):
            root = args.study / str(seed) / arm
            ref = root / "motions/stand/reference.csv"
            ref.parent.mkdir(parents=True, exist_ok=True)
            if not ref.exists():
                ref.symlink_to(args.stand.resolve())
            prepare(root, "soap")
            task = root / "soap"
            rollout(
                SimpleNamespace(
                    folder=task,
                    output=task,
                    zero=args.zero.resolve(),
                    robot=args.robot.resolve(),
                    kitchen="soap",
                    drawer=None,
                    start_offset=0.39,
                    approach_only=False,
                ),
                use_cuda=True,
                reset_pose=None if args.reference_reset else reset_pose,
            )
            metrics = json.loads((task / "metrics.json").read_text())
            with np.load(task / "rollout.npz") as data:
                xy = data["actual"][:, :2]
                at = xy[132]
            walk = np.loadtxt(root / "motions/walk/reference.csv", delimiter=",")
            ref_errors = np.linalg.norm((walk[frames, :2] - walk[0, :2]) - goal, axis=1)[1:]
            results.append(
                dict(
                    seed=seed,
                    arm=arm,
                    success=metrics["task_success"],
                    upright=metrics["upright"],
                    strokes=metrics["completed_strokes"],
                    dock_xy_m=at.tolist(),
                    dock_error_m=float(np.linalg.norm(at - DOCK_XY)),
                    generated_waypoint_mae_m=float(ref_errors.mean()),
                    max_tilt_deg=float(
                        np.degrees(np.arccos(np.clip(metrics["minimum_pelvis_up_z"], -1, 1)))
                    ),
                )
            )
    (args.study / "results.json").write_text(
        json.dumps(
            dict(
                dock_target_xy_m=DOCK_XY.tolist(),
                reset_qpos=None if args.reference_reset else reset_pose.tolist(),
                pairs=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
