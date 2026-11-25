"""Run a Zero-HLM task with its native checkpoints, adding synchronized state traces."""

import argparse
from dataclasses import replace
from importlib import import_module
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def record(
    task,
    output,
    mission=None,
    assets=None,
    pilot=False,
    state_only=False,
    gate_receipt=False,
):
    module = import_module(f"evaluate_{task}")
    if task == "task3":
        module._prepare_task3_color_matcher()
    if mission:
        from missions.workflow import run

        return run(module, mission, assets, output, pilot, state_only, gate_receipt)
    adapter = module.make_task_adapter()
    robot_states = []

    def sample(runtime, slot):
        # The native evaluator accumulates object and door states in this callback.
        metrics = adapter.update_success_state(runtime, slot)
        robot = runtime["robot"].data
        robot_states.append(robot.root_state_w[slot.env_id].detach().cpu().numpy().copy())
        return metrics

    def save(args, runtime, slot, row):
        folder = output / "traces"
        folder.mkdir(exist_ok=True)
        streams = {
            f"{asset.strip('_')}_{key}": np.asarray(values)
            for asset, history in slot.success_state_history.items()
            for key, values in history.items()
        }
        streams["robot_root_state"] = np.asarray(robot_states)
        np.savez_compressed(folder / (row["scenario_id"] + ".npz"), **streams)
        robot_states.clear()
        if adapter.after_episode:
            adapter.after_episode(args, runtime, slot, row)

    def setup(*args, **kwargs):
        try:
            return adapter.setup_runtime(*args, **kwargs)
        except Exception:
            # Kit may terminate the interpreter during shutdown.
            traceback.print_exc()
            raise

    wrapped = replace(adapter, setup_runtime=setup, update_success_state=sample, after_episode=save)
    module.run_evaluation(module.shared.parse_args(wrapped), wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero", type=Path, required=True)
    parser.add_argument("--task", choices=["task2", "task3"], required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--row-start", required=True, type=int)
    parser.add_argument("--row-count", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mission", choices=("home", "factory"))
    parser.add_argument("--station-assets", type=Path)
    parser.add_argument(
        "--delivery-only",
        action="store_true",
        help="Check the first transport leg in the furnished mission scene",
    )
    parser.add_argument(
        "--state-only",
        action="store_true",
        help="Record every physics state for offline camera rendering; the VLA still receives live RGB",
    )
    parser.add_argument(
        "--gate-receipt",
        action="store_true",
        help="Read the cart transponder at the fixed outbound gate while G1 attends the handoff",
    )
    args = parser.parse_args()
    if args.gate_receipt and args.mission != "factory":
        parser.error("--gate-receipt requires the factory mission")
    if args.mission:
        expected = "task3" if args.mission == "home" else "task2"
        if args.task != expected or args.row_count != 1 or args.station_assets is None:
            parser.error(f"{args.mission} requires {expected}, one scene and --station-assets")
        args.station_assets = args.station_assets.resolve()
    root = args.zero.resolve()
    args.output = args.output.resolve()
    sys.path.insert(0, str(root / "src"))
    from zerohlm_bench.launch import build_client, validate_external_assets

    args.python, args.device, args.host = sys.executable, "cuda:0", "127.0.0.1"
    launch = build_client(root, args)
    command = launch.command.copy()
    command[command.index("--diagnostic_video_interval") + 1] = "1"
    if args.mission:
        command[command.index("--max_control_steps") + 1] = "5500"
    # Resolve the standard 4.5 ground plane to the verified local copy.
    assets = args.output / "isaac-assets"
    grid = root / "assets/task1/assets/usd/assets/Isaac/Grid"
    manifest = json.loads((root / "configs/external_assets.json").read_text())
    validate_external_assets(root, [f["path"] for f in manifest["files"]])
    envs = assets / "Isaac/Environments"
    envs.mkdir(parents=True)
    (envs / "Grid").symlink_to(grid, target_is_directory=True)
    command.append(f"--kit_args=--/persistent/isaac/asset_root/cloud={assets}")
    launch.provenance.update(command=command, recorder=Path(__file__).name)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "launch.json").write_text(json.dumps(launch.provenance, indent=2) + "\n")
    sys.path[:0] = [str(Path(command[2]).parent), *launch.env["PYTHONPATH"].split(os.pathsep)]
    os.environ.update(launch.env)
    os.chdir(launch.cwd)
    if args.mission:
        info = json.loads((args.output / "launch.json").read_text())
        info.update(
            mission=args.mission,
            station_assets=str(args.station_assets),
            delivery_only=args.delivery_only,
            state_only=args.state_only,
            gate_receipt=args.gate_receipt,
            observation="factory RTX head RGB"
            if args.mission == "factory"
            else "task3 live GS / mesh head RGB",
        )
        (args.output / "launch.json").write_text(json.dumps(info, indent=2) + "\n")
    sys.argv = command[2:]
    record(
        args.task,
        args.output,
        args.mission,
        args.station_assets,
        args.delivery_only,
        args.state_only,
        args.gate_receipt,
    )


if __name__ == "__main__":
    main()
