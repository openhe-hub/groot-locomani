"""Check a complete mixed-operation mission and render its synchronized review movie."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from presentation import FOLDERS, movie
from stations import STATIONS


def verify(run, checkpoint):
    launch = json.loads((run / "launch.json").read_text())
    report = run / "mission.json"
    recovered = not report.is_file()
    if recovered:
        report = run / "verified-terminal.json"
    metrics = json.loads(report.read_text())
    events = json.loads((run / "events.json").read_text())
    name = metrics["mission"]
    trace = dict(np.load(run / "rollout.npz"))
    work = dict(np.load(run / "station-trace.npz"))
    if (
        launch.get("delivery_only")
        or not metrics["success"]
        or not all(metrics["final_checks"].values())
    ):
        raise ValueError("Incomplete mission")
    if any(e["name"] == "diagnostic_restart" for e in events):
        raise ValueError("A diagnostic restart cannot be delivered as a continuous episode")
    if recovered:
        from verify_dock import certify

        metrics = certify(run)
        # Derive display identity from the unchanged manifest, without synthesizing an evaluator row on disk.
        manifest = Path(launch["config"]["manifest"]["path"])
        if not manifest.is_absolute():
            manifest = Path(launch["command"][2]).parents[4] / manifest
        raw = json.loads(manifest.read_text().splitlines()[launch["rows"][0]])
        row = dict(
            scenario_id=raw["scenario_id"],
            control_steps=metrics["control_steps"],
            gr00t_query_count=len(json.loads((run / "queries.json").read_text())),
        )
    else:
        row = next(csv.DictReader((run / "sim/sim_eval/episodes.csv").open()))
    end = int(row["control_steps"])
    where = {e["name"]: e for e in events}
    ordered = ["start", "delivered", "secure", "return", "worktop", "escort", "receipt", "complete"]
    if recovered:
        where["complete"] = where["dock_receiver_read"]
    if [where[k]["step"] for k in ordered] != sorted(where[k]["step"] for k in ordered) or where[
        "complete"
    ]["step"] != end:
        raise ValueError("Mission phases out of order")
    if (
        end < 3000
        or len(trace["root"]) != end + 1
        or np.max(np.linalg.norm(np.diff(trace["root"][:, :2], axis=0), axis=1)) > 0.085
    ):
        raise ValueError("Duration or base continuity check failed")
    if not np.isfinite(trace["joints"]).all() or trace["root"][:, 2].min() < 0.45:
        raise ValueError("Invalid robot state")
    first = where["delivered"]["step"]
    metrics["duration_s"] = end / 50
    metrics["scenario"] = row["scenario_id"]
    metrics["video_fps"] = 25
    metrics["transport_xy_m"] = float(
        np.linalg.norm(trace["object"][first, :2] - trace["object"][0, :2])
    )
    metrics["lift_m"] = float(np.max(trace["object"][: first + 1, 2]) - trace["object"][0, 2])
    metrics["max_base_step_m"] = float(
        np.linalg.norm(np.diff(trace["root"][:, :2], axis=0), axis=1).max()
    )
    metrics["phase_s"] = {k: where[k]["step"] / 50 for k in ordered}
    if (
        metrics["transport_xy_m"] < 1
        or metrics["lift_m"] < 0.10
        or metrics["live_queries"] != int(row["gr00t_query_count"])
    ):
        raise ValueError("Carry evidence or VLA query count failed")
    if (
        not all(where["delivered"]["checks"]["checks"].values())
        or abs(trace["secure"][-25:]).min() < STATIONS[name].limit * 0.85
    ):
        raise ValueError("Transfer/fixture not settled")
    for leg in ("return", "escort"):
        a = where[leg]["step"]
        b = where["worktop" if leg == "return" else "receipt"]["step"]
        metrics[leg + "_path_m"] = float(
            np.linalg.norm(np.diff(trace["root"][a : b + 1, :2], axis=0), axis=1).sum()
        )
        if metrics[leg + "_path_m"] < 1.0:
            raise ValueError(f"Missing locomotion in {leg}")
    if name == "home":
        metrics["door_final_deg"] = float(np.degrees(trace["door"][-1]))
        if metrics["door_final_deg"] < 50 or work["samples"][-1, 9] != 0:
            raise ValueError("Cupboard or lamp check failed")
    else:
        start = where["dispatch_start_pressed"]["step"]
        metrics["cart_load_travel_m"] = float(
            np.linalg.norm(trace["object"][-1, :2] - trace["object"][start, :2])
        )
        if min(metrics["cart_displacement_m"], metrics["cart_load_travel_m"]) < 0.60:
            raise ValueError("Cart or payload did not reach dispatch")
    query_file = run / "queries.json"
    if (
        not query_file.is_file()
        or len(json.loads(query_file.read_text())) != metrics["live_queries"]
    ):
        raise ValueError("Missing live policy observations")
    raw_source = launch["task"]
    if Path(json.loads(Path(checkpoint).read_text())["checkpoint"]).name != raw_source:
        raise ValueError("Wrong checkpoint for this mission")
    metrics["max_payload_step_m"] = float(
        np.linalg.norm(np.diff(trace["object"][:, :3], axis=0), axis=1).max()
    )
    if metrics["max_payload_step_m"] > 0.13:
        raise ValueError("Payload discontinuity")
    metrics["compliant_grasp"] = where.get("compliant_grasp")
    metrics["loaded_motion_reference"] = where.get("loaded_route")
    metrics["guided_placement"] = where.get("align_load")
    metrics["fixture_final"] = float(trace["secure"][-1])
    if not Path(checkpoint).is_file():
        raise ValueError("Missing checkpoint record")
    return name, trace, metrics, events, row


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--robot",
        type=Path,
        default=Path("GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"),
    )
    p.add_argument(
        "--skin",
        type=Path,
        default=Path("third_party/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"),
    )
    p.add_argument("--encoder", choices=("libx264", "h264_nvenc"), default="libx264")
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help="Package an explicitly labeled, incomplete physical rollout",
    )
    args = p.parse_args()
    if args.diagnostic:
        metrics = json.loads((args.run / "mission.json").read_text())
        events = json.loads((args.run / "events.json").read_text())
        name = metrics["mission"]
        row = next(csv.DictReader((args.run / "sim/sim_eval/episodes.csv").open()))
        trace = dict(np.load(args.run / "rollout.npz"))
        if (
            metrics["success"]
            or row["success"].lower() not in ("false", "0")
            or int(row["control_steps"]) != metrics["control_steps"]
            or len(trace["root"]) != metrics["control_steps"] + 1
            or any(ev["name"] == "diagnostic_restart" for ev in events)
            or json.loads((args.run / "replay/capture.json").read_text())["mission_success"]
        ):
            raise ValueError(
                "Expected an uninterrupted, genuinely failed run and its matching replay"
            )
        metrics["benchmark_terminal_reason"] = row["terminal_reason"]
        metrics["duration_s"] = metrics["control_steps"] / 50
    else:
        name, trace, metrics, events, row = verify(args.run, args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    dest = args.output / (FOLDERS[name] + ("-incomplete" if args.diagnostic else ""))
    if dest.exists():
        raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=args.output) as scratch:
        staged = Path(scratch) / dest.name
        staged.mkdir()
        movie(
            args.run, staged, name, trace, metrics, events, row, args.robot, args.skin, args.encoder
        )
        evidence = staged / "evidence"
        evidence.mkdir()
        for f in ("events.json", "launch.json", "rollout.npz", "station-trace.npz"):
            shutil.copy2(args.run / f, evidence / f)
        ref = Path(json.loads((args.run / "launch.json").read_text())["station_assets"]) / (
            name + "-route.npz"
        )
        pelvis = ref.with_name("pelvis-routes.npz")
        if pelvis.is_file():
            shutil.copy2(pelvis, evidence / pelvis.name)
            postures = ref.with_name("station-postures.npz")
            if postures.is_file():
                shutil.copy2(postures, evidence / postures.name)
            plan = ref.with_name("waypoint-plan.json")
            if plan.is_file():
                shutil.copy2(plan, evidence / plan.name)
        if ref.is_file():
            shutil.copy2(ref, evidence / "carry-reference.npz")
        if (args.run / "queries.json").is_file():
            shutil.copy2(args.run / "queries.json", evidence / "queries.json")
        if (args.run / "replay/capture.json").is_file():
            metrics["video_source"] = "synchronized Isaac replay of measured states"
        if (args.run / "verified-terminal.json").is_file():
            shutil.copy2(args.run / "verified-terminal.json", evidence)
        shutil.copy2(args.checkpoint, evidence / "checkpoint.json")
        (evidence / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        staged.rename(dest)
