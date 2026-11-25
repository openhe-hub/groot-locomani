"""Generate paired root2d trials for the soap pump and three waypoint-conditioned kitchen walks."""

import argparse
import json
from pathlib import Path

from kimodo import load_model
from kimodo.constraints import load_constraints_lst
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.tools import seed_everything
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch


WALK = "Walk forward at a comfortable steady pace."
# Each leg is specified in Isaac kitchen world XY (m), from the station layout and
# the earlier successful run. Only these sparse points go into the motion generator.
LEGS = {
    "loaded": (
        "Walk slowly forward while carrying a basket with both hands and stop upright.",
        3.13,
        [(0, 0.34, 1.62), (50, 0.07, 1.68), (105, -0.48, 1.60), (169, -1.19, 1.35)],
    ),
    "return": (
        "Turn around, walk forward slowly with relaxed arms, and stop.",
        -2.82,
        [
            (0, -1.11, 1.40),
            (48, -1.10, 1.40),
            (105, -0.54, 1.58),
            (171, 0.05, 1.66),
            (210, 0.32, 1.70),
        ],
    ),
    "escort": (
        "Turn around, walk forward slowly with relaxed arms, and stop.",
        0.0,
        [
            (0, 0.32, 1.70),
            (48, 0.30, 1.70),
            (110, -0.20, 1.66),
            (165, -0.72, 1.52),
            (210, -1.09, 1.39),
        ],
    ),
}


def generate(model, converter, directory, prompt, frames, points, seed, headings=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "constraints.json"
    constraints = []
    if points:
        constraints.append(
            dict(
                type="root2d",
                frame_indices=[t for t, _, _ in points],
                smooth_root_2d=[[y, x] for _, x, y in points],
            )
        )
    if headings is not None:
        constraints[0]["global_root_heading"] = np.column_stack(
            [np.cos(headings), np.sin(headings)]
        ).tolist()
    path.write_text(json.dumps(constraints, indent=2) + "\n")
    spec = dict(
        model="Kimodo-G1-RP-v1",
        seed=seed,
        fps=model.fps,
        prompt_phases=[[prompt, frames / model.fps]],
        frame_count=frames,
        num_denoising_steps=100,
        constraint_file=path.name,
    )
    (directory / "prompt.json").write_text(json.dumps(spec, indent=2) + "\n")
    seed_everything(seed)
    with torch.inference_mode():
        motion = model(
            [prompt],
            [frames],
            constraint_lst=load_constraints_lst(str(path), model.skeleton),
            multi_prompt=True,
            num_samples=1,
            num_transition_frames=8,
            num_denoising_steps=100,
            post_processing=False,
            return_numpy=True,
        )
        q = converter.dict_to_qpos(motion, "cuda")
    converter.save_csv(q, str(directory / "reference.csv"))
    return np.loadtxt(directory / "reference.csv", delimiter=",")


def kitchen(model, converter, output, seed, station_headings, sink_offset):
    results = {}
    for leg, (prompt, heading, keys) in LEGS.items():
        if leg == "return" and sink_offset:
            # Account for the measured stopping shortfall at the draining board.
            end_frame, x, y = keys[-1]
            keys = [*keys[:-1], (end_frame, x + sink_offset, y + 0.4 * sink_offset)]
        angles = None
        if station_headings:
            world_angles = {
                "loaded": [140, 174, 188, 200],
                "return": [-160, -70, 0, 6, 3],
                "escort": [3, 95, 175, 194, 200],
            }[leg]
            heading = np.deg2rad(world_angles[0])
            angles = np.deg2rad(world_angles + [world_angles[-1]]) - heading
            keys = [*keys, (keys[-1][0] + 21, *keys[-1][1:])]
        spec = np.array(keys)
        xy = spec[:, 1:] - spec[0, 1:]
        c, s = np.cos(heading), np.sin(heading)
        rotation = np.array([[c, -s], [s, c]])
        local = xy @ rotation
        points = [(int(k[0]), *p) for k, p in zip(keys, local)]
        q = generate(model, converter, output / leg, prompt, keys[-1][0] + 1, points, seed, angles)
        q[:, :2] = (q[:, :2] - q[0, :2]) @ rotation.T + spec[0, 1:]
        rotations = Rotation.from_euler("z", heading) * Rotation.from_quat(
            q[:, 3:7], scalar_first=True
        )
        t = np.arange(len(q)) / model.fps
        new_t = np.arange(int(t[-1] * 50) + 1) / 50
        result = np.stack([np.interp(new_t, t, v) for v in q.T], axis=1)
        quat = Slerp(t, rotations)(new_t).as_quat(scalar_first=True)
        for i in range(1, len(quat)):
            if np.dot(quat[i - 1], quat[i]) < 0:
                quat[i] *= -1
        result[:, 3:7] = quat
        results[leg] = result
        results[leg + "_points"] = spec
    np.savez_compressed(output / "pelvis-routes.npz", **results, seed=seed, control_hz=50)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--study", choices=["soap", "kitchen", "both"], default="both")
    p.add_argument(
        "--station-headings",
        action="store_true",
        help="Also condition the kitchen turns and final docking heading",
    )
    p.add_argument(
        "--sink-offset",
        type=float,
        default=0.0,
        help="Additional approach distance in metres at the worktop (kitchen only)",
    )
    args = p.parse_args()
    model = load_model("Kimodo-G1-RP-v1", device="cuda")
    converter = MujocoQposConverter(model.skeleton)
    if args.study in ("both", "kitchen"):
        kitchen(
            model,
            converter,
            args.output / "kitchen",
            42,
            args.station_headings,
            args.sink_offset,
        )
    if args.study in ("both", "soap"):
        # Authored in the forward (+X) / left (+Y) coordinates of the walking clip.
        points = [
            (0, 0.0, 0.0),
            (19, 0.63, 0.0),
            (34, 1.13, 0.02),
            (49, 1.63, 0.04),
            (63, 2.10, 0.06),
        ]
        for seed in range(42, 50):
            for name, keys in (("text", []), ("pelvis", points)):
                path = args.output / "soap" / str(seed) / name / "motions/walk"
                generate(model, converter, path, WALK, 120, keys, seed)
                print(seed, name, flush=True)


if __name__ == "__main__":
    main()
