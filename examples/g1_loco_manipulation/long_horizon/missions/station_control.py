"""SONIC empty transit and measured hand interactions following a live GR00T delivery."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .dock_steps import dock_reference, wrap
from .fk_control import ToolServo


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


class AfterDelivery:
    def __init__(self, name, runtime, slot, args, records, event):
        import onnxruntime as ort
        import sonic_policy as sp

        from .stations import CART_SHIFT, LAMP_SWITCH, MARKS, STATIONS, WIPE

        self.sp = sp
        self.name = name
        self.home = name == "home"
        self.spec = STATIONS[name]
        self.cells = np.array([self.spec.surface(u, v) for u, v in (WIPE if self.home else MARKS)])
        self.switch = np.array(self.spec.secure) + LAMP_SWITCH
        self.cart_delta = CART_SHIFT.copy()
        self.r = runtime
        self.env = runtime["env"]
        self.robot = runtime["robot"]
        self.mechanism = self.env.scene["secure"]
        self.receiver = self.env.scene["placement" if self.home else "hanger"]
        self.event = event
        self.records = records
        self.slot = slot
        self.body_ids = [runtime["sim_joint_names"].index(j) for j in sp.G1_MUJOCO_BODY_JOINT_NAMES]
        history = np.array([np.r_[r["root"][:7], r["joints"][self.body_ids]] for r in records])
        self.source = history[0].copy()
        self.destination = history[-1].copy()
        delta_xy = np.linalg.norm(history[:, :2] - history[0, :2], axis=1)
        orientation = Rotation.from_quat(history[:, 3:7], scalar_first=True).as_euler("ZYX")[:, 0]
        turn = np.abs(np.unwrap(orientation) - orientation[0])
        moving = np.flatnonzero((delta_xy > 0.09) | (turn > 0.16))
        if not len(moving):
            raise ValueError("No outward transport leg")
        self.route = history[max(0, moving[0] - 25) :].copy()
        self.route[:, 22:] = self.source[22:]
        self.cart_start = records[-1]["receiver"][:7].copy()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(Path(args.models_dir) / "model_encoder.onnx"),
            opts,
            providers=["CPUExecutionProvider"],
        )
        self.phase = ""
        self.ref = None
        self.goal = None
        self.side = "left"
        self.secured = False
        self.covered = np.zeros(len(self.cells), bool)
        self.service_ready = False
        self.confirmed = False
        self.lamp_on = self.home
        self.coupled = False
        self.coupling_offset = None
        self.secure_target = 0.0
        self.grip = False
        self.arm_cmd = None
        self.arm_bias = None
        self.arrival_window = 0
        self.done_step = None
        self.telemetry = []
        self.visual_updates = {}
        self.cart_live = False
        self.cart_step = 0
        self.final_checks = {}
        self.last = records[-1]
        self.start_phase("secure", slot)

    def tip(self, side=None):
        key = side or self.side
        return self.last[key + "_tip"]

    def qpos(self):
        return np.r_[self.last["root"][:7], self.last["joints"][self.body_ids]]

    def set_reference(self, slot, q, fps=50):
        self.ref = self.sp.ReferenceMotion.from_qpos(q, source_fps=fps, control_hz=50)
        slot.policy.reference_motion = self.ref
        slot.policy.init_ref_quat = q[0, 3:7].copy()
        slot.policy.init_base_quat = self.last["root"][3:7].copy()
        slot.chunk = None

    def start_phase(self, phase, slot):
        self.phase = phase
        self.adjustments = 0
        self.docking = False
        self.start = slot.steps
        self.goal = None
        self.grip = False
        self.arm_cmd = None
        self.arm_bias = None
        self.arrival_window = 0
        self.done_step = None
        self.stage_index = 0
        self.stage_step = slot.steps
        self.side = "left" if phase == "secure" or (self.home and phase == "receipt") else "right"
        self.arm_start = self.tip().copy()
        self.park_arms = {}
        self.park_origins = {side: self.tip(side).copy() for side in ("left", "right")}
        q = self.qpos()
        for ids in self.r.get("station_arm_gains", {}).values():
            self.robot.write_joint_stiffness_to_sim(
                self.robot.data.default_joint_stiffness[:, ids], joint_ids=ids
            )
            self.robot.write_joint_damping_to_sim(
                self.robot.data.default_joint_damping[:, ids], joint_ids=ids
            )
        if phase in ("return", "escort"):
            segment = self.route[::-1].copy() if phase == "return" else self.route.copy()
            if "pelvis_routes" in self.r:
                segment = self.r["pelvis_routes"][phase].copy()
                segment[:, 22:] = self.source[22:]
                self.event("kimodo_walk", slot.steps, leg=phase, frames=len(segment))
            # Preserve world heading even when the return ends slightly off its nominal pose.
            pre = np.linspace(q, segment[0], 55)
            if np.dot(q[3:7], segment[0, 3:7]) < 0:
                pre[:, 3:7] = np.linspace(-q[3:7], segment[0, 3:7], 55)
            pre[:, 3:7] /= np.linalg.norm(pre[:, 3:7], axis=1)[:, None]
            tail = np.tile(segment[-1], (65, 1))
            traj = np.vstack([pre, segment, tail])
            self.set_reference(slot, traj, fps=50 if "pelvis_routes" in self.r else 44)
            if phase == "escort" and not self.home:
                self.cart_live = True
                self.cart_step = slot.steps
        else:
            hold = np.tile(q, (92, 1))
            blend = (np.arange(92) / 91) ** 2 * (3 - 2 * np.arange(92) / 91)
            hold[:, 22:] += blend[:, None] * (self.source[22:] - q[22:])
            if "pelvis_routes" in self.r and (
                phase != "secure" or "loaded" in self.r["pelvis_routes"]
            ):
                # After a generated walking clip, settle the legs from the stepping
                # posture into the measured standing configuration before reaching.
                legs = self.r.get("station_postures", {}).get(phase, self.source[7:])[:12]
                hold[:, 7:19] += blend[:, None] * (legs - q[7:19])
            self.set_reference(slot, hold)
        self.event(phase, slot.steps, base=self.last["root"][:3].tolist())

    def _paint(self, index, clean):
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        p = stage.GetPrimAtPath(f"/World/envs/env_0/mission_panel/Mark{index}")
        if clean:
            UsdGeom.Imageable(p).MakeInvisible()
        else:
            UsdGeom.Gprim(p).GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.8, 0.33)])

    def _pad(self, tip):
        import omni.usd
        from pxr import Gf

        p = omni.usd.get_context().get_stage().GetPrimAtPath("/World/envs/env_0/mission_panel/Pad")
        p.GetAttribute("xformOp:translate").Set(
            Gf.Vec3d(float(tip[0]), float(tip[1]), float(self.spec.panel[2] + 0.012))
        )

    def _lamp_off(self):
        import omni.usd
        from pxr import Gf, UsdGeom, UsdLux

        stage = omni.usd.get_context().get_stage()
        prefix = "/World/envs/env_0/secure/Base"
        lamp = UsdLux.SphereLight(stage.GetPrimAtPath(prefix + "/Bulb"))
        lamp.GetIntensityAttr().Set(0.0)
        UsdGeom.Gprim(stage.GetPrimAtPath(prefix + "/Shade")).GetDisplayColorAttr().Set(
            [Gf.Vec3f(0.16, 0.19, 0.18)]
        )
        self.lamp_on = bool(lamp.GetIntensityAttr().Get() > 0.0)

    def before(self, slot):
        import torch

        i = min(slot.steps - self.start, self.ref.num_frames - 1)
        obs = slot.policy.build_encoder_observation(i)
        token = self.encoder.run(None, {"obs_dict": obs.reshape(1, -1)})[0].reshape(-1)
        slot.chunk.token[0, slot.chunk_pos] = token
        slot.chunk.left_hand_gripper[0, slot.chunk_pos, 0] = (
            0.0 if self.side == "left" and self.grip else 1.0
        )
        slot.chunk.right_hand_gripper[0, slot.chunk_pos, 0] = (
            0.0 if self.side == "right" and self.grip else 1.0
        )
        self.mechanism.set_joint_position_target(
            torch.tensor([[self.secure_target]], device=self.env.device, dtype=torch.float32)
        )
        if not self.home:
            # A force-limited motor propels a dynamic cart along a real prismatic
            # guide. The hat moves only through contact with its padded deck.
            frac = smooth((slot.steps - self.cart_step) / 420.0) if self.cart_live else 0.0
            direction = self.cart_delta / np.linalg.norm(self.cart_delta)
            state = self.last["receiver"]
            x = float((state[:3] - self.cart_start[:3]) @ direction)
            speed = float(state[7:10] @ direction)
            u = np.clip((slot.steps - self.cart_step) / 420.0, 0.0, 1.0) if self.cart_live else 0.0
            desired_speed = np.linalg.norm(self.cart_delta) * 6 * u * (1 - u) / 8.4
            force = np.clip(
                310 * (np.linalg.norm(self.cart_delta) * frac - x) + 82 * (desired_speed - speed),
                -90,
                90,
            )
            local = Rotation.from_quat(state[3:7], scalar_first=True).inv().apply(force * direction)
            t = torch.as_tensor(local, device=self.env.device, dtype=torch.float32).reshape(1, 1, 3)
            self.receiver.set_external_force_and_torque(t, torch.zeros_like(t))

    def arm_action(self, slot, action):
        if self.phase == "secure" and "station_postures" in self.r:
            root = self.last["root"]
            yaw = Rotation.from_quat(root[3:7], scalar_first=True).as_euler("ZYX")[0]
            turn = Rotation.from_euler("z", yaw)
            u = smooth((slot.steps - self.start) / 70)
            for side, sign in (("left", 1), ("right", -1)):
                if side == self.side and self.goal is not None:
                    continue
                if side not in self.park_arms:
                    self.park_arms[side] = ToolServo(self.r, side)
                # Retract each hand on its own side with vertical clearance over the basket.
                neutral = root[:3] + turn.apply([0.13, sign * 0.30, 0.13])
                origin = self.park_origins[side]
                target = (1 - u) * origin + u * neutral + [0, 0, 0.10 * np.sin(np.pi * u)]
                self.park_arms[side].apply(action, self.last, self.tip(side), target)
            if self.arm_cmd is None and self.side in self.park_arms:
                self.arm_cmd = self.park_arms[self.side]
        if self.goal is None:
            return
        if self.arm_cmd is None:
            self.arm_cmd = ToolServo(self.r, self.side)
        self.arm_cmd.apply(action, self.last, self.tip(), self.goal)

    def reach(self, target, slot, duration=95):
        self.goal = self.stage_origin + (target - self.stage_origin) * smooth(
            (slot.steps - self.stage_step) / duration
        )

    def next_stage(self, slot):
        self.stage_step = slot.steps
        self.stage_origin = self.tip().copy()
        self.stage_index += 1

    def adjust_dock(self, slot, location):
        q = self.qpos()
        if location == "source":
            goal = self.source[:2] + [0.035, 0.012]
            yaw = Rotation.from_quat(self.source[3:7], scalar_first=True).as_euler("ZYX")[0]
            if "station_postures" in self.r and self.home:
                goal, yaw = np.array([0.32, 1.70]), 0.05
        elif self.home:
            goal = np.asarray(self.spec.secure[:2]) + [0.13, 0.41]
            yaw = np.pi
        elif location == "receiver":
            yaw = Rotation.from_quat(self.destination[3:7], scalar_first=True).as_euler("ZYX")[0]
            c, s = np.cos(yaw), np.sin(yaw)
            goal = self.last["object"][:2] - np.array([[c, -s], [s, c]]) @ np.array([0.32, -0.23])
        else:
            goal = np.asarray(self.spec.secure[:2]) + [0.24, -0.35]
            yaw = 0.94
        actual_yaw = Rotation.from_quat(q[3:7], scalar_first=True).as_euler("ZYX")[0]
        error = np.linalg.norm(goal - q[:2])
        turn = abs(wrap(yaw - actual_yaw))
        if location == "source":
            targets = self.cells
            side = "right"
        elif location == "receiver" and not self.home:
            targets = [self.last["object"][:3] + [0, 0, 0.085]]
            side = "right"
        else:
            targets = [
                self.spec.handle(0),
                self.spec.handle(1),
                self.switch if self.home else self.spec.handle(0.5),
            ]
            side = "left"
        sid = self.robot.find_bodies(side + "_shoulder_pitch_link")[0][0]
        shoulder = self.robot.data.body_pos_w[0, sid].detach().cpu().numpy()
        reach = max(np.linalg.norm(p - shoulder) for p in targets)
        if reach < 0.55 and turn < 0.8:
            return False
        if (error < 0.16 and turn < 0.50) or self.adjustments >= 4:
            return False
        motion = dock_reference(q, self.route, goal, yaw)
        self.adjustments += 1
        self.start = self.stage_step = slot.steps
        self.set_reference(slot, motion)
        self.docking = True
        self.event(
            "dock_adjustment",
            slot.steps,
            station=location,
            attempt=self.adjustments,
            start_error_m=float(error),
            goal=goal.tolist(),
        )
        return True

    def secure_step(self, slot):
        tip = self.tip()
        elapsed = slot.steps - self.stage_step
        if self.stage_index == 0:
            if elapsed >= (self.ref.num_frames if self.docking else 65):
                if self.adjust_dock(slot, "secure"):
                    return
                self.docking = False
                self.arm_start = self.tip().copy()
                self.next_stage(slot)
            return
        if self.stage_index == 1:
            target = self.spec.handle(0.0)
            if self.home:
                # Lift clear of the table edge before reaching across the lid.
                clear = np.array(
                    [
                        max(target[0] + 0.055, self.last["root"][0] + 0.035),
                        target[1] + 0.045,
                        target[2] + 0.055,
                    ]
                )
                if elapsed < 75:
                    self.reach(clear, slot, 70)
                else:
                    self.goal = clear + (target - clear) * smooth((elapsed - 75) / 58)
            else:
                self.reach(target, slot, 100)
            if elapsed > 82 and np.linalg.norm(tip - target) < 0.055:
                self.coupled = True
                self.grip = False
                self.coupling_offset = tip - target
                self.event(
                    "handle_engaged", slot.steps, distance_m=float(np.linalg.norm(tip - target))
                )
                self.next_stage(slot)
        elif self.stage_index == 2:
            f = 1.18 * smooth(elapsed / 145.0)
            target = self.spec.handle(f) + self.coupling_offset
            self.goal = target
            measured = tip - self.coupling_offset - np.array(self.spec.secure)
            if self.home:
                projected = float(np.clip(-measured[1], 0, self.spec.limit))
                self.secure_target = -projected
            else:
                self.secure_target = float(
                    np.clip(np.arctan2(measured[1], measured[0]), 0, self.spec.limit)
                )
            travel = abs(self.last["secure"]) / self.spec.limit
            if travel >= 0.89 and elapsed >= 112:
                self.secured = True
                self.coupled = False
                self.grip = False
                self.secure_target = (-1 if self.home else 1) * self.spec.limit
                self.event(
                    "lid_seated" if self.home else "guard_locked",
                    slot.steps,
                    measured=float(self.last["secure"]),
                )
                self.next_stage(slot)
        else:
            self.reach(self.arm_start, slot, 80)
            if elapsed >= 85:
                self.start_phase("return", slot)

    def panel_step(self, slot):
        tip = self.tip()
        elapsed = slot.steps - self.stage_step
        panel = np.array(self.spec.panel)
        if self.stage_index == 0:
            if elapsed >= 45:
                self.next_stage(slot)
            return
        if self.stage_index <= len(self.cells):
            idx = self.stage_index - 1
            target = self.cells[idx] + [0, 0, 0.016 if self.home else 0.045]
            self.reach(target, slot, 95 if idx == 0 else (22 if self.home else 65))
            near = (
                np.linalg.norm(tip - target) < 0.040
                if self.home
                else np.linalg.norm(tip[:2] - target[:2]) < 0.035
                and 0.012 < tip[2] - panel[2] < 0.18
            )
            if (
                self.home
                and abs(tip[2] - panel[2] - 0.016) < 0.039
                and np.all(np.abs(tip[:2] - panel[:2]) < [0.17, 0.12])
            ):
                self._pad(tip)
            self.arrival_window = self.arrival_window + 1 if near else 0
            required = 6 if self.home else 19
            if self.arrival_window >= required and elapsed >= (16 if self.home else 63):
                self.covered[idx] = True
                self._paint(idx, self.home)
                tag = {}
                if not self.home:
                    import omni.usd

                    prim = (
                        omni.usd.get_context()
                        .get_stage()
                        .GetPrimAtPath(f"/World/envs/env_0/mission_panel/Mark{idx}")
                    )
                    tag["tag_id"] = prim.GetAttribute("mission:tagId").Get()
                    if not tag["tag_id"]:
                        raise ValueError("Tag has no programmed identifier")
                self.event(
                    "wipe_cell" if self.home else "tag_read",
                    slot.steps,
                    index=idx,
                    **tag,
                    distance_m=float(np.linalg.norm(tip - target)),
                )
                self.arrival_window = 0
                self.next_stage(slot)
        elif not self.home and self.stage_index == len(self.cells) + 1:
            from .stations import START_UV

            target = panel + [*START_UV, 0.023]
            if elapsed < 55:
                self.reach(target + [0, 0, 0.075], slot, 48)
            else:
                self.goal = target + np.array([0, 0, 0.075]) * (1 - smooth((elapsed - 55) / 38))
            if (
                elapsed > 80
                and np.linalg.norm(tip[:2] - target[:2]) < 0.031
                and abs(tip[2] - target[2]) < 0.035
            ):
                self.service_ready = True
                self.event(
                    "dispatch_start_pressed",
                    slot.steps,
                    distance_m=float(np.linalg.norm(tip - target)),
                )
                self.next_stage(slot)
        else:
            self.reach(self.arm_start, slot, 80)
            if elapsed >= 82:
                self.service_ready = bool(self.covered.all())
                self.event(
                    "board_complete" if self.home else "dispatch_authorized",
                    slot.steps,
                    positions=int(self.covered.sum()),
                )
                self.start_phase("escort", slot)

    def _stable(self):
        if self.home and "bench_support_n" in self.last:
            from .stations import bench_support

            return bench_support(
                self.records, self.r["replay_success_evaluator"], self.r["replay_success_cfg"]
            )
        if not self.home and "cart_support_n" in self.last:
            from .stations import cart_support

            return cart_support(self.records)
        r = self.records[-10:]
        states = np.array([e["object"] for e in r])
        receiver = np.array([e["receiver"] for e in r])
        payload = dict(object_root_pose=states[:, :7], object_root_velocity=states[:, 7:])
        if self.home:
            payload.update(
                placement_root_pose=receiver[:, :7],
                door_joint_position=np.array([e["door"] for e in r]),
                door_contact_latched=np.array([e["door_latched"] for e in r]),
                robot_hand_open=np.array([e["hand"] for e in r]),
            )
        else:
            payload["hanger_root_pose"] = receiver[:, :7]
        return self.r["replay_success_evaluator"](payload, self.r["replay_success_cfg"])

    def receipt(self, slot):
        from .stations import CART_ID, CART_TAG

        tip = self.tip()
        elapsed = slot.steps - self.stage_step
        if self.home:
            target = self.switch.copy()
        else:
            cart = self.last["receiver"]
            rot = Rotation.from_quat(cart[3:7], scalar_first=True)
            tag = cart[:3] + rot.apply(CART_TAG)
            # Read the outward-facing transponder from the accessible side of the cart.
            target = tag + rot.apply([-0.085, 0, 0])
        if not self.home and self.r.get("gate_receipt"):
            import omni.usd
            from pxr import Gf, UsdGeom

            stage = omni.usd.get_context().get_stage()
            reader = stage.GetPrimAtPath("/World/DispatchReceiver")
            antenna = np.asarray(
                UsdGeom.Xformable(reader).GetLocalTransformation().ExtractTranslation()
            )
            distance = float(np.linalg.norm(tag - antenna))
            robot_distance = float(np.linalg.norm(self.last["root"][:2] - cart[:2]))
            valid = (
                distance < 0.125
                and robot_distance < 1.25
                and self._stable()["ok"]
                and np.linalg.norm(cart[7:10]) < 0.035
                and np.linalg.norm(cart[:2] - self.cart_start[:2]) > 0.65
            )
            self.arrival_window = self.arrival_window + 1 if valid else 0
            if self.arrival_window >= 40:
                code = (
                    stage.GetPrimAtPath("/World/envs/env_0/hanger/DispatchTag")
                    .GetAttribute("mission:tagId")
                    .Get()
                )
                self.confirmed = code == reader.GetAttribute("mission:expected").Get()
                if self.confirmed:
                    UsdGeom.Gprim(
                        stage.GetPrimAtPath("/World/DispatchReceiver/Indicator")
                    ).GetDisplayColorAttr().Set([Gf.Vec3f(0.05, 0.88, 0.24)])
                    self.event(
                        "dock_receiver_read",
                        slot.steps,
                        tag_id=code,
                        antenna_distance_m=distance,
                        robot_distance_m=robot_distance,
                        support_n=self.last["cart_support_n"],
                        stable_frames=self.arrival_window,
                    )
                    self.final_checks = dict(
                        item_settled=True,
                        secure=bool(abs(self.last["secure"]) / self.spec.limit > 0.85),
                        panel=self.service_ready,
                        receipt=True,
                        cart_at_dock=True,
                    )
                    if all(self.final_checks.values()):
                        self.event("complete", slot.steps, checks=self.final_checks)
                        return True
            return False
        if self.stage_index == 0:
            if elapsed >= 55:
                self.next_stage(slot)
        elif self.stage_index == 1:
            self.reach(target, slot, 105)
            d = float(np.linalg.norm(tip - target))
            if self.home:
                in_range = d < 0.055
            else:
                offset = rot.inv().apply(tip - tag)
                in_range = (
                    -0.175 < offset[0] < -0.005
                    and abs(offset[1]) < 0.060
                    and abs(offset[2]) < 0.065
                    and np.linalg.norm(cart[7:10]) < 0.04
                    and self._stable()["ok"]
                )
            self.arrival_window = self.arrival_window + 1 if in_range else 0
            if elapsed >= 95 and self.arrival_window >= 14:
                if self.home:
                    self._lamp_off()
                if not self.home:
                    import omni.usd

                    code = (
                        omni.usd.get_context()
                        .get_stage()
                        .GetPrimAtPath("/World/envs/env_0/hanger/DispatchTag")
                        .GetAttribute("mission:tagId")
                        .Get()
                    )
                    if code != CART_ID:
                        raise ValueError("Unexpected dock transponder")
                    self.event(
                        "cart_id_verified",
                        slot.steps,
                        tag_id=code,
                        antenna_offset_m=offset.tolist(),
                        support_n=self.last["cart_support_n"],
                    )
                self.confirmed = not self.lamp_on
                self.event(
                    "pantry_light_off" if self.home else "dock_item_read", slot.steps, distance_m=d
                )
                self.next_stage(slot)
        else:
            self.reach(self.arm_start, slot, 85)
            if elapsed >= 85:
                stable = self._stable()
                self.final_checks = dict(
                    item_settled=bool(stable["ok"]),
                    secure=float(abs(self.last["secure"]) / self.spec.limit) > 0.85,
                    panel=self.service_ready,
                    receipt=self.confirmed,
                )
                if not self.home:
                    dist = float(np.linalg.norm(self.last["receiver"][:2] - self.cart_start[:2]))
                    self.final_checks["cart_at_dock"] = dist > 0.63
                if all(self.final_checks.values()):
                    self.event("complete", slot.steps, checks=self.final_checks)
                    return True
                slot.terminal_reason = "final_state_failed"
        return False

    def update(self, slot, record):
        self.last = record
        elapsed = slot.steps - self.start
        if self.phase == "secure":
            self.secure_step(slot)
        elif self.phase in ("return", "escort"):
            if elapsed >= self.ref.num_frames:
                if self.adjust_dock(slot, "source" if self.phase == "return" else "receiver"):
                    return False
                self.event("arrived", slot.steps, leg=self.phase, xy=record["root"][:2].tolist())
                self.start_phase("worktop" if self.phase == "return" else "receipt", slot)
        elif self.phase == "worktop":
            self.panel_step(slot)
        else:
            if self.receipt(slot):
                return True
        tip = self.tip()
        err = float(np.linalg.norm(tip - self.goal)) if self.goal is not None else 0.0
        self.telemetry.append(
            [
                slot.steps,
                *tip,
                *(self.goal if self.goal is not None else [np.nan] * 3),
                self.stage_index,
                int(self.covered.sum()),
                int(self.lamp_on),
                self.secure_target,
                err,
            ]
        )
        if slot.steps % 100 == 0:
            print(
                "[WORK]",
                self.phase,
                self.stage_index,
                tip.round(3).tolist(),
                round(err, 3),
                flush=True,
            )
        if self.phase not in ("return", "escort") and slot.steps - self.stage_step > 315:
            self.event(
                "station_timeout",
                slot.steps,
                phase=self.phase,
                index=self.stage_index,
                goal=self.goal.tolist() if self.goal is not None else [],
                tip=tip.tolist(),
            )
            slot.terminal_reason = "station_timeout"
        return False

    def summary(self):
        return dict(
            final_checks=self.final_checks,
            covered=int(self.covered.sum()),
            secured=self.secured,
            worktop_complete=self.service_ready,
            receipt=self.confirmed,
            cart_displacement_m=float(
                np.linalg.norm(self.last["receiver"][:2] - self.cart_start[:2])
            )
            if not self.home
            else None,
        )
