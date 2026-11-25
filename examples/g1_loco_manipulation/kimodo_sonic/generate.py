"""Generate G1 reference trajectories for the SONIC demonstration set."""

import argparse
import json
from pathlib import Path

from kimodo import load_model
from kimodo.exports.mujoco import MujocoQposConverter
from kimodo.tools import seed_everything
import numpy as np
import torch


MOTIONS = {
    "stand": [("Stand still upright, arms relaxed by your sides.", 4)],
    "walk": [("Walk forward at a comfortable steady pace.", 4)],
    "run": [("Run straight forward at a brisk jogging pace.", 4)],
    "squat": [("Bend both knees to squat low, then stand back up.", 5)],
    "wave": [("Stand in place and wave hello with your right hand above shoulder height.", 4)],
    "sidestep": [("Take several sideways steps to your left while facing forward.", 4)],
    "backward": [("Walk backwards in a straight line.", 4)],
    "reach": [("Stand on both feet and reach up high with both hands.", 4)],
    "turn": [("Turn around on the spot to face the opposite direction.", 4)],
    "march": [("March in place, raising alternate knees high.", 4)],
    "clap": [("Stand still and clap your hands together in front of your chest.", 4)],
    "drawer": [
        ("Walk straight forward toward the workbench.", 2),
        (
            "Stand still and push the open drawer shut with your right hand, extending your arm forward.",
            3,
        ),
        ("Pull your right hand back and stand still.", 2),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("motions", nargs="*", help="Motion names (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    device = "cuda"
    model = load_model("Kimodo-G1-RP-v1", device=device)
    converter = MujocoQposConverter(model.skeleton)
    for name in args.motions or MOTIONS:
        phases = MOTIONS[name]
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        seed_everything(args.seed)
        with torch.inference_mode():
            motion = model(
                [prompt for prompt, _ in phases],
                [round(seconds * model.fps) for _, seconds in phases],
                multi_prompt=True,
                num_samples=1,
                num_transition_frames=8,
                num_denoising_steps=100,
                post_processing=False,
                return_numpy=True,
            )
            qpos = converter.dict_to_qpos(motion, device)
        converter.save_csv(qpos, str(folder / "reference.csv"))
        values = np.loadtxt(folder / "reference.csv", delimiter=",")
        assert values.shape == (round(sum(t for _, t in phases) * model.fps), 36)
        assert np.isfinite(values).all()
        metadata = dict(
            model="Kimodo-G1-RP-v1",
            seed=args.seed,
            fps=model.fps,
            prompt_phases=phases,
            transition_frames=8,
            denoising_steps=100,
        )
        (folder / "prompt.json").write_text(json.dumps(metadata, indent=2) + "\n")
        print(name, values.shape, flush=True)


if __name__ == "__main__":
    main()
