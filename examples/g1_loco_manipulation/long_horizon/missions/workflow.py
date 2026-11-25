"""Continuous loaded delivery, station work and final inspection in one Isaac world."""

from dataclasses import replace
import hashlib
import json
import traceback
from unittest.mock import patch

import numpy as np

from .station_control import AfterDelivery
from .stations import STATIONS, bench_support, cart_support


class HeadRGB:
    def render_env(self, *, env, env_id, **kwargs):
        return (
            env.scene["head_camera"]
            .data.output["rgb"][env_id, ..., :3]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )


def setup_scene(
    native, name, assets, args, count, adapter, detail_camera=False, gate_receipt=False
):
    from importlib import import_module

    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.sim import UsdFileCfg

    utils = import_module("isaaclab_tasks.utils.parse_cfg")

    parse = utils.parse_env_cfg

    def configured(*a, **kw):
        fingers = [
            "{ENV_REGEX_NS}/robot/" + side + "_" + finger
            for side in (("left", "right") if name == "home" else ("left",))
            for finger in ("index_1", "index_2", "middle_1", "middle_2", "thumb_3", "thumb_4")
        ]

        cfg = parse(*a, **kw)
        if name == "factory":
            cfg.scene.scene.spawn.usd_path = str(assets / "dispatch-room.usda")
            from isaaclab.sensors import ContactSensorCfg

            cfg.scene.object.spawn.activate_contact_sensors = True
            cfg.scene.cart_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/object/basket",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/hanger", *fingers],
                update_period=0.0,
            )
            cfg.scene.light.spawn.intensity = 100.0
            from .rail import install_rail

            install_rail(cfg, args, assets, gate_receipt)

            cfg.viewer.eye = (2.7, -2.8, 1.9)
            cfg.viewer.lookat = (0.18, 0.48, 0.79)
        else:
            from isaaclab.sensors import ContactSensorCfg
            from isaaclab.sim import spawn_from_usd

            cfg.scene.object.spawn.activate_contact_sensors = True
            cfg.scene.placement.spawn.usd_path = str(assets / "serving-bench.usda")
            cfg.scene.placement.spawn.func = spawn_from_usd
            cfg.scene.bench_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/object/mesh",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/placement/mesh", *fingers],
                update_period=0.0,
            )
            cfg.viewer.eye = (-1.05, -0.10, 1.7)
            cfg.viewer.lookat = (-1.17, 1.73, 0.83)
        cfg.scene.secure = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/secure",
            init_state=ArticulationCfg.InitialStateCfg(pos=STATIONS[name].secure),
            spawn=UsdFileCfg(
                usd_path=str(assets / f"{name}-secure.usda"),
                semantic_tags=[("class", "target_placement")],
            ),
            actuators={
                "drive": ImplicitActuatorCfg(
                    joint_names_expr=["travel"],
                    stiffness=180.0 if name == "home" else 8.0,
                    damping=8.0 if name == "home" else 1.2,
                    effort_limit_sim=24.0,
                )
            },
        )
        cfg.scene.panel = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/mission_panel",
            spawn=UsdFileCfg(
                usd_path=str(assets / f"{name}-panel.usda"),
                semantic_tags=[("class", "target_placement")],
            ),
        )
        if detail_camera:
            import copy

            from isaaclab.sim import PinholeCameraCfg

            cam = copy.deepcopy(cfg.scene.head_camera)
            cam.prim_path = "{ENV_REGEX_NS}/detail_camera"
            cam.offset.pos = (0.0, 0.0, 0.0)
            cam.offset.rot = (1.0, 0.0, 0.0, 0.0)
            cam.offset.convention = "world"
            cam.width, cam.height = 640, 480
            cam.data_types = ["rgb"]
            cam.update_period = 0.0
            cam.spawn = PinholeCameraCfg(
                focal_length=2.9,
                horizontal_aperture=3.5,
                vertical_aperture=2.62,
                focus_distance=0.75,
                clipping_range=(0.02, 100.0),
                lock_camera=False,
            )
            cfg.scene.detail_camera = cam
        return cfg

    with patch.object(utils, "parse_env_cfg", configured):
        try:
            return native.setup_runtime(args, count, adapter)
        except Exception:
            traceback.print_exc()
            raise


def run(module, name, assets, output, pilot=False, state_only=False, gate_receipt=False):
    import onnxruntime as ort

    shared = module.shared
    native = module.make_task_adapter()
    args = shared.parse_args(native)
    sensor = []
    events = []
    result = {}
    extension = None
    handoff = None
    live = 0
    queries = []
    native_renderer = native.init_gs_renderer

    def start_renderer(args, env):
        if name == "factory":
            return HeadRGB()
        initial = native_renderer(args, env)
        raw = HeadRGB()

        class KitchenHead:
            def render_env(self, **kw):
                return (initial if extension is None else raw).render_env(**kw)

        return KitchenHead()

    def event(tag, step, **values):
        events.append(dict(name=tag, step=int(step), **values))
        (output / "events.json").write_text(
            json.dumps(events, indent=2, default=lambda x: x.item()) + "\n"
        )
        print("[MISSION]", json.dumps(events[-1]), flush=True)

    def policies(args, runtime, count):
        group = native.make_policies(args, runtime, count)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 3
        opts.inter_op_num_threads = 1
        for policy in group:
            policy.decoder_session = ort.InferenceSession(
                str(policy.decoder_model_path), opts, providers=["CPUExecutionProvider"]
            )
        return group

    def sample(runtime, slot):
        nonlocal result, extension, handoff
        env, robot = runtime["env"], runtime["robot"]

        def array(t):
            return t[slot.env_id].detach().cpu().numpy().copy()

        root = array(robot.data.root_state_w)
        joints = array(robot.data.joint_pos)
        payload = array(env.scene["object"].data.root_state_w)
        receiver = array(env.scene["placement" if name == "home" else "hanger"].data.root_state_w)
        secure = env.scene["secure"]
        phase = (
            ("align" if handoff is not None and handoff.guiding else "carry")
            if extension is None
            else extension.phase
        )
        record = dict(
            root=root,
            joints=joints,
            object=payload,
            receiver=receiver,
            phase=phase,
            secure=float(secure.data.joint_pos[slot.env_id, 0]),
            secure_anchor=array(secure.data.body_pos_w)[0],
            left=array(robot.data.body_state_w)[runtime["left_wrist"], :7],
            right=array(robot.data.body_state_w)[runtime["right_wrist"], :7],
            hand=np.array([slot.left_hand_state_open01, slot.right_hand_state_open01]),
        )
        from scipy.spatial.transform import Rotation

        for side in ("left", "right"):
            fingertip = array(robot.data.body_state_w)[runtime[side + "_finger"]]
            record[side + "_tip"] = fingertip[:3] + Rotation.from_quat(
                fingertip[3:7], scalar_first=True
            ).apply([0, 0.048, 0.003 if side == "left" else -0.003])
        if name == "home":
            force = env.scene["bench_contact"].data.force_matrix_w
            record["bench_support_n"] = float(abs(force[0, :, 0, 2].sum()))
            record["grip_contact_n"] = float(force[0, :, 1:, :].norm(dim=-1).sum())
            measured = shared.build_gr00t_state_values_env(
                robot, slot.env_id, runtime["sim_joint_names"], runtime["joint_limits"]
            )
            record["hand"] = np.array(
                [measured["left_hand_gripper"][0], measured["right_hand_gripper"][0]]
            )
            record["door"] = float(env.scene["scene"].data.joint_pos[0, 0])
            record["door_latched"] = bool(getattr(env, "_task3_door_contact_latched")[0])
        if name == "factory":
            forces = env.scene["cart_contact"].data.force_matrix_w
            record["cart_support_n"] = float(abs(forces[0, :, 0, 2].sum()))
            record["grip_contact_n"] = float(forces[0, :, 1:, :].norm(dim=-1).sum())
        if handoff is None:
            from .carry_assist import Handoff

            handoff = Handoff(runtime, name == "home", event)
        record["assisted_grip"] = handoff.held
        if (
            name == "home"
            and record["door_latched"]
            and not any(e["name"] == "cupboard_latched" for e in events)
        ):
            event("cupboard_latched", slot.steps, robot_xyz=root[:3].tolist())
        record["guided_release"] = handoff.guiding
        sensor.append(record)
        if len(sensor) == 1:
            event(
                "start",
                slot.steps,
                secure_anchor=record["secure_anchor"].tolist(),
                robot=root[:7].tolist(),
            )
        if slot.steps % 150 == 0:
            print(
                "[STATE]",
                slot.steps,
                phase,
                root[:3].round(3).tolist(),
                payload[:3].round(3).tolist(),
                round(record["secure"], 3),
                flush=True,
            )
        if extension is None:
            result = native.update_success_state(runtime, slot)
            if name == "factory":
                deck = cart_support(sensor)
                deck.update(
                    max_object_lift=result["max_object_lift"], original_cabinet_evaluator=result
                )
                result = deck
            if name == "home":
                bench = bench_support(
                    sensor, runtime["replay_success_evaluator"], runtime["replay_success_cfg"]
                )
                bench.update(
                    max_object_lift=result["max_object_lift"], original_stool_evaluator=result
                )
                result = bench
            if result["ok"] and result["max_object_lift"] > 0.10 and not handoff.held:
                event("delivered", slot.steps, queries=live, checks=result)
                np.savez_compressed(
                    output / "delivery-prefix.npz",
                    **{k: np.array([a[k] for a in sensor]) for k in sensor[0]},
                    joint_names=np.array(runtime["sim_joint_names"]),
                )
                if pilot:
                    return result

                extension = AfterDelivery(name, runtime, slot, args, sensor, event)
                slot.chunk = None
            elif slot.steps > (3100 if handoff.guiding else 1650):
                slot.terminal_reason = "delivery_timeout"
            return dict(result, ok=False)
        ok = extension.update(slot, record)
        return dict(result, ok=bool(ok), mission_complete=bool(ok))

    def request(args, runtime, client, modalities, gs, slot, adapter):
        nonlocal live
        if extension is None and not (handoff is not None and handoff.guiding):
            native.request_chunk(args, runtime, client, modalities, gs, slot, adapter)
            live += 1
            queries.append(
                dict(
                    step=int(slot.steps),
                    rgb_sha256=hashlib.sha256(
                        np.ascontiguousarray(slot.final_frame).tobytes()
                    ).hexdigest(),
                )
            )
        else:
            ones = np.ones((1, args.chunk_horizon, 1), dtype=np.float32)
            slot.chunk = shared.SonicActionChunk(
                np.zeros((1, args.chunk_horizon, 64), np.float32), ones, ones.copy()
            )
            slot.chunk_pos = 0

    def before(args, runtime, slots):
        if extension is None:
            if native.before_step:
                native.before_step(args, runtime, slots)
            if slots[0].active and sensor:
                handoff.before(sensor[-1], slots[0], args)
        elif slots[0].active:
            extension.before(slots[0])

    def advance(args, runtime, slots, gs, adapter):
        slot = slots[0]
        video = slot.video
        if video is not None:
            stride = 4 if pilot else 2
            video.fps = video.ego.fps = video.third_person.fps = 50 / stride
            if slot.steps % stride:
                slot.video = None
        env = runtime["env"]
        original = env.step
        if state_only:
            slot.video = None
        if slot.video is not None and sensor:
            import torch

            x, y = sensor[-1]["root"][:2]
            if name == "home":
                eye, target = (x + 0.40, y - 1.80, 1.67), (x - 0.07, y + 0.08, 0.79)
            else:
                eye, target = (x + 1.85, y - 2.12, 1.68), (x + 0.08, y + 0.09, 0.79)
            env.scene["evaluation_camera"].set_world_poses_from_view(
                eyes=torch.tensor([eye], device=env.device, dtype=torch.float32),
                targets=torch.tensor([target], device=env.device, dtype=torch.float32),
            )

        try:
            if extension is not None:

                def action_step(action):
                    extension.arm_action(slot, action)
                    return original(action)

                env.step = action_step
            elif handoff is not None and handoff.guiding:

                def aligned(action):
                    handoff.correct(sensor[-1], slot, action)
                    return original(action)

                env.step = aligned
            native.step_active_slots(args, runtime, slots, gs, adapter)
        finally:
            slot.video = video
            env.step = original

    def save(args, runtime, slot, row):
        (output / "queries.json").write_text(json.dumps(queries, indent=2) + "\n")
        arrays = {k: np.array([r[k] for r in sensor]) for k in sensor[0]}
        arrays["joint_names"] = np.array(runtime["sim_joint_names"])
        np.savez_compressed(output / "rollout.npz", **arrays)
        info = dict(
            mission=name,
            control_steps=slot.steps,
            control_hz=50,
            live_queries=live,
            terminal_reason=slot.terminal_reason,
            delivery=result,
            success=bool(row["success"]),
            path_m=float(np.linalg.norm(np.diff(arrays["root"][:, :2], axis=0), axis=1).sum()),
            min_root_z=float(arrays["root"][:, 2].min()),
        )
        if extension is not None:
            info.update(extension.summary())
            np.savez_compressed(
                output / "station-trace.npz",
                samples=np.asarray(extension.telemetry),
                columns=np.array(
                    [
                        "step",
                        "tip_x",
                        "tip_y",
                        "tip_z",
                        "goal_x",
                        "goal_y",
                        "goal_z",
                        "substage",
                        "cells",
                        "lamp_on",
                        "handle_target",
                        "tool_error",
                    ]
                ),
            )
        (output / "mission.json").write_text(
            json.dumps(info, indent=2, default=lambda x: x.item()) + "\n"
        )

    def setup(args, count, adapter):
        r = setup_scene(native, name, assets, args, count, adapter, gate_receipt=gate_receipt)
        r["gate_receipt"] = gate_receipt
        print("[SCENE]", list(r["env"].scene.keys()), flush=True)
        pelvis_file = assets / "pelvis-routes.npz"
        if pelvis_file.is_file():
            with np.load(pelvis_file) as routes:
                r["pelvis_routes"] = {
                    k: routes[k].copy() for k in ("loaded", "return", "escort") if k in routes
                }
            r["transport_reference"] = (
                pelvis_file if "loaded" in r["pelvis_routes"] else assets / (name + "-route.npz")
            )
            postures = assets / "station-postures.npz"
            if postures.is_file():
                with np.load(postures) as reference:
                    r["station_postures"] = dict(reference)
        else:
            r["transport_reference"] = assets / (name + "-route.npz")
        for side in ("left", "right"):
            r[side + "_finger"] = r["robot"].find_bodies(side + "_middle_2")[0][0]
            r[side + "_wrist"] = r["robot"].find_bodies(side + "_wrist_yaw_link")[0][0]
        print(
            "[FIXTURE]",
            r["env"].scene["secure"].data.body_names,
            r["env"].scene["secure"].data.joint_names,
            flush=True,
        )
        r["env"].cfg.sim.render_interval = 1000000 if state_only else 8
        return r

    def initialize(env, env_id, row):
        if name == "factory":
            pose = np.array(row.hanger.pos)
            pose[2] += 0.025
            row = replace(row, hanger=shared.Pose.from_arrays(pose, row.hanger.quat_wxyz))
        if name == "home":
            table = shared.Pose.from_arrays([-2.10, 1.49, 0], row.placement.quat_wxyz)
            row = replace(row, placement=table)
        native.reset_env_slot_to_manifest(env, env_id, row)

    adapter = replace(
        native,
        reset_env_slot_to_manifest=initialize,
        setup_runtime=setup,
        init_gs_renderer=start_renderer,
        make_policies=policies,
        update_success_state=sample,
        request_chunk=request,
        before_step=before,
        step_active_slots=advance,
        after_chunk=None,
        after_episode=save,
    )
    module.run_evaluation(args, adapter)
