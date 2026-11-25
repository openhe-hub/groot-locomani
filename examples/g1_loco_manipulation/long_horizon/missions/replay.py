"""Render the measured states of a completed mission in its original Isaac room."""

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
from scipy.spatial.transform import Rotation


def render(run, zero, previews=None, diagnostic=False):
    launch = json.loads((run / "launch.json").read_text())
    report = run / "mission.json"
    if not report.is_file():
        report = run / "verified-terminal.json"
    metrics = json.loads(report.read_text())
    if (
        not previews
        and not diagnostic
        and (not metrics["success"] or not all(metrics["final_checks"].values()))
    ):
        raise ValueError("Expected a completed mission")
    name = launch["mission"]
    home = name == "home"
    data = dict(np.load(run / "rollout.npz"))
    events = json.loads((run / "events.json").read_text())
    if any(e["name"] == "diagnostic_restart" for e in events):
        raise ValueError("Diagnostic run")
    reactions = {
        e["step"]: e
        for e in events
        if e["name"]
        in (
            "wipe_cell",
            "tag_read",
            "pantry_light_off",
            "dispatch_start_pressed",
            "dock_receiver_read",
        )
    }
    works = np.load(run / "station-trace.npz")["samples"]
    tools = {int(v[0]): v for v in works}
    sys.path[:0] = [
        str(Path(__file__).resolve().parents[1]),
        str(Path(launch["command"][2]).parent),
        str(zero / "src"),
        str(zero / "runtime"),
    ]
    os.environ.update(launch["config"]["env"])
    sys.argv = launch["command"][2:]
    mod = importlib.import_module("evaluate_" + launch["task"])
    native = mod.make_task_adapter()
    args = mod.shared.parse_args(native)
    from isaaclab.app import AppLauncher

    app = AppLauncher(args, multi_gpu=False).app
    try:
        import cv2
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux
        import torch

        from missions.stations import CART_TAG, LAMP_SWITCH, STATIONS
        from missions.workflow import setup_scene

        r = setup_scene(
            native,
            name,
            Path(launch["station_assets"]),
            args,
            1,
            native,
            detail_camera=True,
            gate_receipt=launch.get("gate_receipt", False),
        )
        env, robot = r["env"], r["robot"]
        native.reset_env_slot_to_manifest(
            env, 0, native.read_manifest(args.manifest)[launch["rows"][0]]
        )
        # Compiled tensors are addressed by name, just as in the original recording.
        indices = [data["joint_names"].tolist().index(j) for j in robot.data.joint_names]

        def tensor(v):
            return torch.as_tensor(np.asarray(v).copy(), device=env.device, dtype=torch.float32)

        joints = tensor(data["joints"][:, indices])
        root = tensor(data["root"])
        objects = tensor(data["object"])
        receivers = tensor(data["receiver"])
        qd = torch.zeros_like(joints[:1])
        scene = omni.usd.get_context().get_stage()
        panel = "/World/envs/env_0/mission_panel"
        secure = "/World/envs/env_0/secure/Base"
        receiver = env.scene["placement" if home else "hanger"]
        cam = env.scene["evaluation_camera"]
        detail = env.scene["detail_camera"]
        head = env.scene["head_camera"]
        spec = STATIONS[name]
        out = run / ("preview-views" if previews else "replay")
        out.mkdir(exist_ok=True)
        writers = []
        for v, (w, h) in [("third_person", (960, 720)), ("ego", (640, 480))]:
            writer = cv2.VideoWriter(
                str(out / (v + ".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Cannot open {v} video")
            writers.append(writer)
        secure_asset = env.scene["secure"]
        last = 0
        try:
            steps = (
                sorted({min(len(root) - 1, max(0, round(t * 50))) for t in previews})
                if previews
                else range(1, len(root), 2)
            )
            for frame, step in enumerate(steps):
                for a, v in ((robot, root), (env.scene["object"], objects), (receiver, receivers)):
                    a.write_root_pose_to_sim(v[step : step + 1, :7])
                    a.write_root_velocity_to_sim(v[step : step + 1, 7:])
                robot.write_joint_state_to_sim(joints[step : step + 1], qd)
                q = tensor([[data["secure"][step]]])
                secure_asset.write_joint_state_to_sim(q, torch.zeros_like(q))
                if home:
                    dq = tensor([[data["door"][step]]])
                    env.scene["scene"].write_joint_state_to_sim(dq, torch.zeros_like(dq))
                for i in range(last + 1, step + 1):
                    if i in reactions:
                        ev = reactions[i]
                        tag = ev["name"]
                        if tag == "wipe_cell":
                            UsdGeom.Imageable(
                                scene.GetPrimAtPath(panel + f"/Mark{ev['index']}")
                            ).MakeInvisible()
                        elif tag == "tag_read":
                            UsdGeom.Gprim(
                                scene.GetPrimAtPath(panel + f"/Mark{ev['index']}")
                            ).GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.8, 0.33)])
                        elif tag == "dock_receiver_read":
                            UsdGeom.Gprim(
                                scene.GetPrimAtPath("/World/DispatchReceiver/Indicator")
                            ).GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.88, 0.24)])
                        elif tag == "pantry_light_off":
                            UsdLux.SphereLight(
                                scene.GetPrimAtPath(secure + "/Bulb")
                            ).GetIntensityAttr().Set(0.0)
                            UsdGeom.Gprim(
                                scene.GetPrimAtPath(secure + "/Shade")
                            ).GetDisplayColorAttr().Set([Gf.Vec3f(0.16, 0.19, 0.18)])
                    if home and data["phase"][i] == "worktop" and i in tools:
                        tip = tools[i][1:4]
                        p = np.asarray(spec.panel)
                        if abs(tip[2] - p[2] - 0.016) < 0.039 and np.all(
                            abs(tip[:2] - p[:2]) < [0.17, 0.12]
                        ):
                            scene.GetPrimAtPath(panel + "/Pad").GetAttribute(
                                "xformOp:translate"
                            ).Set(Gf.Vec3d(float(tip[0]), float(tip[1]), float(p[2] + 0.012)))
                last = step
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(dt=1 / 25)
                x, y = data["root"][step, :2]
                ph = data["phase"][step]
                if home:
                    eye = np.array([np.clip(x + 0.52, -0.83, 0.77), max(-0.44, y - 1.95), 1.72])
                    aim = np.array([x - 0.06, y + 0.08, 0.78])
                else:
                    eye = np.array([x + 1.65, y - 2.05, 1.68])
                    aim = np.array([x + 0.07, y + 0.10, 0.80])
                if ph == "secure":
                    focus = spec.handle(0)
                    eye = eye if home else np.array([x - 1.4, y - 1.2, 1.65])
                elif ph == "worktop":
                    focus = np.array(spec.panel)
                elif ph == "receipt":
                    focus = (
                        np.array(spec.secure) + LAMP_SWITCH
                        if home
                        else data["receiver"][step, :3]
                        + Rotation.from_quat(data["receiver"][step, 3:7], scalar_first=True).apply(
                            CART_TAG
                        )
                    )
                else:
                    focus = data["object"][step, :3]
                if ph in ("secure", "worktop", "receipt"):
                    aim = 0.65 * aim + 0.35 * focus
                cam.set_world_poses_from_view(eyes=tensor([eye]), targets=tensor([aim]))
                deye = (
                    focus + np.array([0.62, -0.59, 0.44])
                    if ph != "secure" or home
                    else focus + [-0.55, -0.54, 0.36]
                )
                if not home and ph == "receipt":
                    deye = focus + Rotation.from_quat(
                        data["receiver"][step, 3:7], scalar_first=True
                    ).apply([-0.62, 0.11, 0.34])
                detail.set_world_poses_from_view(eyes=tensor([deye]), targets=tensor([focus]))
                for _ in range(6 if frame == 0 or previews else 2):
                    env.sim.render()
                close = detail if ph in ("secure", "worktop", "receipt") else head
                for wr, sensor in zip(writers, (cam, close)):
                    sensor.update(dt=max(0.06, sensor.cfg.update_period), force_recompute=True)
                    rgb = sensor.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    image = np.ascontiguousarray(rgb[..., ::-1])
                    wr.write(image)
                    if previews:
                        cv2.imwrite(
                            str(out / f"{step:05d}-{'scene' if sensor is cam else 'detail'}.jpg"),
                            image,
                        )
                if frame % 300 == 0:
                    print("Replayed", step, ph, flush=True)
        finally:
            for w in writers:
                w.release()
        (out / "capture.json").write_text(
            json.dumps(
                dict(
                    source="measured states",
                    fps=25,
                    frames=frame + 1,
                    room=name,
                    indices="1,3,...",
                    mission_success=bool(metrics["success"]),
                ),
                indent=2,
            )
            + "\n"
        )
        env.close()
    except Exception:
        traceback.print_exc()
        raise
    else:
        app.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--zero", type=Path, required=True)
    p.add_argument(
        "--preview", type=float, nargs="+", help="Inspect selected simulation timestamps"
    )
    p.add_argument(
        "--diagnostic",
        action="store_true",
        help="Replay the recorded physics of an incomplete attempt",
    )
    a = p.parse_args()
    render(a.run.resolve(), a.zero.resolve(), a.preview, a.diagnostic)
