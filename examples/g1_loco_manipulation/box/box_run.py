import io
import json
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
import numpy as np
import torch
import zmq


parser = get_isaaclab_arena_cli_parser()
args, _ = parser.parse_known_args()
with SimulationAppContext(args), torch.inference_mode():
    from isaaclab_arena.examples.policy_runner_cli import setup_policy_argument_parser
    from isaaclab_arena.metrics.metrics import compute_metrics

    args = setup_policy_argument_parser(parser).parse_args()
    builder = get_arena_builder_from_cli(args)
    env_name, cfg = builder.build_registered()
    print("viewer", cfg.viewer, flush=True)
    cfg.viewer.resolution = (960, 640)
    env = gym.make(env_name, cfg=cfg, render_mode="rgb_array").unwrapped
    sock = zmq.Context().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 150000)
    sock.connect("ipc:///demo/box.sock")
    output = Path("/demo/box-runs-v2")
    output.mkdir(exist_ok=True)
    for episode in range(6):
        obs, _ = env.reset(seed=42 + episode)
        sock.send(b"reset")
        assert sock.recv() == b"ok"
        log = []
        start = env.scene["robot"].data.root_pos_w[0].cpu().numpy().copy()
        fps = round(1 / env.step_dt)
        with imageio.get_writer(
            output / f"box-{episode}.mp4", fps=fps, codec="libx264", macro_block_size=1
        ) as writer:
            for step in range(env.max_episode_length + 1):
                if step % 50 == 0:
                    buffer = io.BytesIO()
                    np.savez(
                        buffer,
                        rgb=obs["camera_obs"]["robot_head_cam_rgb"].cpu().numpy(),
                        joints=obs["policy"]["robot_joint_pos"].cpu().numpy(),
                    )
                    sock.send(buffer.getvalue())
                    chunk = np.load(io.BytesIO(sock.recv()), allow_pickle=False)
                    assert chunk.shape[1] >= 50
                frame = env.render()[:, :, :3].copy()
                if step in (1, 100, 300):
                    imageio.imwrite(output / f"view-{episode}-{step}.jpg", frame)
                    imageio.imwrite(
                        output / f"ego-{episode}-{step}.jpg",
                        obs["camera_obs"]["robot_head_cam_rgb"][0].cpu().numpy()[:, :, :3],
                    )
                writer.append_data(frame)
                with torch.inference_mode():
                    obs, _, term, trunc, _ = env.step(
                        torch.as_tensor(chunk[:, step % 50], device=env.device, dtype=torch.float32)
                    )
                if not (term.any() or trunc.any()):
                    robot = env.scene["robot"].data.root_pos_w[0].cpu().tolist()
                    box = env.scene["brown_box"].data.root_pos_w[0].cpu().tolist()
                    log.append([step, robot, box])
                else:
                    break
        metrics = compute_metrics(env)
        record = dict(
            episode=episode,
            seed=42 + episode,
            steps=step + 1,
            terminated=bool(term.any()),
            truncated=bool(trunc.any()),
            metrics={k: float(v) for k, v in metrics.items()},
            trace=log,
        )
        (output / f"box-{episode}.json").write_text(json.dumps(record))
        print("RESULT", {k: v for k, v in record.items() if k != "trace"}, flush=True)
        if (
            record["terminated"]
            and not record["truncated"]
            and record["metrics"].get("success_rate", 0) > 0
        ):
            break
    env.close()
