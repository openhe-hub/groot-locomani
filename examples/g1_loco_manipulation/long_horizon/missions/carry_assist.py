"""A closure-gated compliant grasp and a receiver-relative, physically actuated set-down."""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from .arm_ik import ArmIK
from .stations import BENCH_CENTRE, BENCH_HALF, BENCH_HEIGHT, TRAY_CENTRE, TRAY_HALF


def ease(t):
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t)


class Handoff:
    def __init__(self, runtime, home, event):
        self.r = runtime
        self.home = home
        self.event = event
        self.robot = runtime["robot"]
        self.env = runtime["env"]
        self.asset = self.env.scene["object"]
        self.sides = ("left", "right") if home else ("left",)
        self.mass = float(self.asset.root_physx_view.get_masses()[0, 0])
        self.held = False
        self.engaged = False
        self.guiding = False
        self.released = False
        self.arms = {}
        self.refs = {}
        self.encoder = None
        self.last_target = None
        self.first = None

    def target(self, s):
        seat = s["receiver"]
        rot = R.from_quat(seat[3:7], scalar_first=True)
        rel = rot.inv().apply(s["object"][:3] - seat[:3])
        centre, half = (BENCH_CENTRE[:2], BENCH_HALF) if self.home else (TRAY_CENTRE, TRAY_HALF)
        margin = np.array([0.19, 0.19]) if self.home else np.array([0.11, 0.10])
        safe = np.clip(rel[:2], centre - half + margin, centre + half - margin)
        z = BENCH_HEIGHT + 0.017 if self.home else 0.887
        return seat[:3] + rot.apply([*safe, z]), float(np.linalg.norm(rel[:2] - safe))

    def activate(self, s, slot):
        self.held = self.engaged = True
        obj = s["object"]
        rot = R.from_quat(obj[3:7], scalar_first=True)
        for side in self.sides:
            w = s[side]
            wr = R.from_quat(w[3:], scalar_first=True)
            self.refs[side] = (wr.inv().apply(obj[:3] - w[:3]), wr.inv() * rot)
        self.last_target = obj[:3].copy()
        self.event(
            "compliant_grasp",
            slot.steps,
            object_xyz=obj[:3].tolist(),
            measured_contact_n=float(s["grip_contact_n"]),
        )

    def wrench(self, record):
        import torch

        obj = record["object"]
        ro = R.from_quat(obj[3:7], scalar_first=True)
        force = np.zeros(3)
        torque = np.zeros(3)
        if self.held:
            poses = []
            rotations = []
            for side in self.sides:
                w = record[side]
                wr = R.from_quat(w[3:], scalar_first=True)
                p, rot = self.refs[side]
                poses.append(w[:3] + wr.apply(p))
                rotations.append((wr * rot).as_quat())
            target = np.mean(poses, axis=0)
            rotation = R.from_quat(rotations).mean()
            vel = np.clip((target - self.last_target) * 50, -1.4, 1.4)
            self.last_target = target
            force = self.mass * np.clip(
                330 * (target - obj[:3]) + 23 * (vel - obj[7:10]) + [0.0, 0.0, 9.81], -55.0, 55.0
            )
            torque = (
                self.mass
                * 0.012
                * np.clip(120 * (rotation * ro.inv()).as_rotvec() - 15 * obj[10:], -40, 40)
            )

        def ten(v):
            return torch.as_tensor(v, device=self.env.device, dtype=torch.float32)

        self.asset.set_external_force_and_torque(
            ten(ro.inv().apply(force)).reshape(1, 1, 3),
            ten(ro.inv().apply(torque)).reshape(1, 1, 3),
        )
        forces = []
        torques = []
        ids = []
        for side in self.sides:
            w = record[side]
            wr = R.from_quat(w[3:], scalar_first=True)
            f = -force / len(self.sides)
            tau = -torque / len(self.sides) + np.cross(obj[:3] - w[:3], f)
            forces.append(wr.inv().apply(f))
            torques.append(wr.inv().apply(tau))
            ids.append(self.r[side + "_wrist"])
        self.robot.set_external_force_and_torque(
            ten(np.array(forces))[None], ten(np.array(torques))[None], body_ids=ids
        )
        return force

    def _q(self, s):
        return np.r_[s["root"][:7], s["joints"][self.body]]

    def reference(self, slot, q):
        import sonic_policy as sp

        self.ref = sp.ReferenceMotion.from_qpos(q, source_fps=50, control_hz=50)
        slot.policy.reference_motion = self.ref
        slot.policy.init_ref_quat = q[0, 3:7].copy()
        slot.policy.init_base_quat = q[0, 3:7].copy()
        self.start = slot.steps

    def begin_route(self, s, slot, args):
        import onnxruntime as ort
        import sonic_policy as sp

        self.guiding = True
        self.mode = "route"
        self.corrections = 0
        self.body = [self.r["sim_joint_names"].index(j) for j in sp.G1_MUJOCO_BODY_JOINT_NAMES]
        raw = self.r["transport_reference"]
        q = self._q(s)
        if "loaded" in self.r.get("pelvis_routes", {}):
            # Retain the measured arm posture so that generating footsteps does not open the grasp.
            ref = self.r["pelvis_routes"]["loaded"].copy()
            ref[:, 19:] = q[19:]
            if "station_postures" in self.r:
                # Historical upper-body targets retain the basket grasp during the novel gait.
                upper = self.r["station_postures"]["carry_upper"]
                u = np.linspace(0, len(upper) - 1, len(ref))
                ref[:, 19:] = np.stack(
                    [np.interp(u, np.arange(len(upper)), col) for col in upper.T], axis=1
                )
            first = 0
        else:
            with np.load(raw) as data:
                names = data["joint_names"].tolist()
                ref = np.c_[
                    data["root"],
                    data["joints"][:, [names.index(j) for j in sp.G1_MUJOCO_BODY_JOINT_NAMES]],
                ]
            cost = np.linalg.norm(ref[:, :2] - q[:2], axis=1) + 0.32 * (
                1 - abs(ref[:, 3:7] @ q[3:7])
            )
            first = int(np.argmin(cost[: max(1, len(ref) - 100)]))
        self.walk_reference = ref.copy()
        sequence = ref[first:].copy()
        pre = np.linspace(q, sequence[0], 40)
        if q[3:7] @ sequence[0, 3:7] < 0:
            pre[:, 3:7] = np.linspace(-q[3:7], sequence[0, 3:7], 40)
        pre[:, 3:7] /= np.linalg.norm(pre[:, 3:7], axis=1)[:, None]
        self.reference(slot, np.vstack([pre, sequence, np.tile(sequence[-1], (20, 1))]))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(
            str(Path(args.models_dir) / "model_encoder.onnx"),
            opts,
            providers=["CPUExecutionProvider"],
        )
        self.event(
            "loaded_route",
            slot.steps,
            reference=raw.name,
            join_frame=first,
            object_xyz=s["object"][:3].tolist(),
        )

    def release(self, s, slot, gap):
        self.held = False
        self.released = True
        self.release_step = slot.steps
        self.mode = "release"
        self.release_wrists = {k: s[k][:3].copy() for k in self.sides}
        self.event(
            "released_over_receiver",
            slot.steps,
            object_xyz=s["object"][:3].tolist(),
            gap_m=float(gap),
            height_above_deck_m=float(s["object"][2] - (BENCH_HEIGHT if self.home else 0.812)),
        )

    def track_corridor(self, s, slot, i):
        positions = self.ref.root_pos
        j = min(len(positions) - 1, i + 24)
        k = max(0, i - 4)
        v = (positions[j, :2] - positions[k, :2]) * 50 / max(1, j - k)
        err = positions[i, :2] - s["root"][:2]
        if np.linalg.norm(v) > 0.10:
            desired = v + np.clip(1.7 * err, -0.30, 0.30)
            angle = np.arctan2(v[0] * desired[1] - v[1] * desired[0], v @ desired)
            offset = np.clip(angle, -0.38, 0.38)
        else:
            offset = 0.0
        self.steer = 0.92 * getattr(self, "steer", 0.0) + 0.08 * offset
        slot.policy.init_base_quat = (
            R.from_euler("z", self.steer)
            * R.from_quat(slot.policy.init_ref_quat, scalar_first=True)
        ).as_quat(scalar_first=True)

    def begin_place(self, s, slot):
        obj = s["object"]
        self.start_pos = obj[:3].copy()
        self.start_rot = R.from_quat(obj[3:7], scalar_first=True)
        self.goal, gap = self.target(s)
        # Aim further inside the deck so the load clears the retaining lip.
        if not self.home:
            centre = s["receiver"][:3] + R.from_quat(s["receiver"][3:7], scalar_first=True).apply(
                [*TRAY_CENTRE, 0]
            )
            self.goal[:2] += 0.18 * (centre[:2] - self.goal[:2])
        self.clear_z = max(self.start_pos[2], self.goal[2] + (0.11 if self.home else 0.13))
        self.end_rot = (
            R.from_euler("z", self.start_rot.as_euler("ZYX")[0]) if self.home else self.start_rot
        )
        for side in self.sides:
            w = s[side]
            wr = R.from_quat(w[3:], scalar_first=True)
            self.refs[side] = (wr.inv().apply(obj[:3] - w[:3]), wr.inv() * self.start_rot)
            self.arms[side] = ArmIK(self.r, side)
        self.last_target = obj[:3].copy()
        self.mode = "lower" if gap < 0.055 else "raise"
        self.place_start = slot.steps
        q = self._q(s)
        hold = np.tile(q, (90, 1))
        if "station_postures" in self.r:
            w = np.linspace(0, 1, len(hold))[:, None]
            hold[:, 7:19] += (w * w * (3 - 2 * w)) * (
                self.r["station_postures"]["secure"][:12] - q[7:19]
            )
            if gap < 0.20:
                self.mode = "over" if gap >= 0.055 else "lower"
        self.reference(slot, hold)
        self.event(
            "align_load",
            slot.steps,
            target_xyz=self.goal.tolist(),
            object_xyz=obj[:3].tolist(),
            gap_m=gap,
        )

    def before(self, s, slot, args):
        if self.first is None:
            self.first = s["object"][:3].copy()
        pos = s["object"][:3]
        moved = np.linalg.norm(pos[:2] - self.first[:2])
        close = all(
            getattr(slot.chunk, side + "_hand_gripper")[0, slot.chunk_pos, 0] < 0.45
            for side in self.sides
        )
        if (
            not self.engaged
            and close
            and s["grip_contact_n"] > 0.003
            and pos[2] > self.first[2] - 0.025
        ):
            self.activate(s, slot)
        goal, gap = self.target(s)
        # Let the vision policy acquire, lift and begin transporting. The recorded
        # GR00T gait then supplies a room-specific loaded route to SONIC.
        if (
            self.held
            and not self.guiding
            and pos[2] > self.first[2] + 0.10
            and moved > (0.32 if self.home else 0.40)
        ):
            self.begin_route(s, slot, args)
        if self.guiding:
            elapsed = slot.steps - self.start
            if self.mode == "route":
                self.track_corridor(s, slot, min(elapsed, self.ref.num_frames - 1))
            in_zone = (
                gap < 0.055
                and moved > (1.5 if self.home else 1.25)
                and (not self.home or s["door"] > 0.95)
                and pos[2] > (BENCH_HEIGHT + 0.035 if self.home else 0.852)
            )
            if self.mode == "route" and (elapsed >= self.ref.num_frames - 1 or in_zone):
                self.begin_place(s, slot)
            # Real contact or a bounded drop to the deck permits release before
            # the fingertips strike its rim.
            support = s["bench_support_n" if self.home else "cart_support_n"]
            clearance = pos[2] - (BENCH_HEIGHT if self.home else 0.812)
            precise = self.home and "loaded" in self.r.get("pelvis_routes", {})
            if (
                self.held
                and (not precise or (gap < 0.12 and clearance < 0.105))
                and self.mode in ("raise", "over", "lower")
                and slot.steps - self.place_start > 12
                and gap < (0.095 if self.home else 0.062)
                and 0.005 < clearance < (0.155 if self.home else 0.205)
                and (support > 0.008 or np.linalg.norm(s["object"][7:9]) < 0.65)
            ):
                self.release(s, slot, gap)
            if (
                self.mode == "raise"
                and slot.steps - self.place_start > 46
                and pos[2] > self.goal[2] + 0.055
            ):
                delta = self.goal[:2] - pos[:2]
                if np.linalg.norm(delta) > 0.13 and self.corrections < 4:
                    from .dock_steps import dock_reference

                    q = self._q(s)
                    yaw = R.from_quat(q[3:7], scalar_first=True).as_euler("ZYX")[0]
                    ref = dock_reference(q, self.walk_reference, q[:2] + delta, yaw)
                    self.reference(slot, ref)
                    self.mode = "approach"
                    self.corrections += 1
                    self.event("loaded_footsteps", slot.steps, error_m=float(np.linalg.norm(delta)))
                else:
                    self.mode = "over"
                    self.place_start = slot.steps
                    self.start_pos = pos.copy()
            if self.mode == "approach" and slot.steps - self.start >= self.ref.num_frames - 1:
                self.begin_place(s, slot)
            if (
                self.mode == "over"
                and slot.steps - self.place_start > 48
                and gap < (0.12 if "station_postures" in self.r else 0.042)
            ):
                self.mode = "lower"
                self.start_pos = pos.copy()
                self.place_start = slot.steps
            if (
                self.mode == "lower"
                and slot.steps - self.place_start > 32
                and gap < 0.046
                and goal[2] - 0.04 < pos[2] < goal[2] + (0.082 if self.home else 0.046)
            ):
                self.held = False
                self.released = True
                self.release_step = slot.steps
                self.mode = "release"
                self.release_wrists = {k: s[k][:3].copy() for k in self.sides}
                self.event("released_over_receiver", slot.steps, object_xyz=pos.tolist(), gap_m=gap)
            i = min(slot.steps - self.start, self.ref.num_frames - 1)
            obs = slot.policy.build_encoder_observation(i)
            slot.chunk.token[0, slot.chunk_pos] = self.encoder.run(
                None, {"obs_dict": obs.reshape(1, -1)}
            )[0].reshape(-1)
        if self.engaged:
            for side in self.sides:
                getattr(slot.chunk, side + "_hand_gripper")[0, slot.chunk_pos, 0] = (
                    0.0 if self.held else 1.0
                )
        force = self.wrench(s)
        if slot.steps % 100 == 0:
            print(
                "[HANDOFF]",
                slot.steps,
                getattr(self, "mode", "live"),
                round(gap, 3),
                round(float(np.linalg.norm(force)), 3),
                flush=True,
            )

    def correct(self, s, slot, action):
        if not self.guiding or self.mode in ("route", "approach"):
            return
        pos = s["object"][:3]
        ro = R.from_quat(s["object"][3:7], scalar_first=True)
        elapsed = slot.steps - self.place_start
        desired = self.start_pos.copy()
        if self.mode == "raise":
            desired[2] += ease(elapsed / 55) * (self.clear_z - desired[2])
        elif self.mode == "over":
            desired[:2] += (self.goal[:2] - desired[:2]) * ease(elapsed / 75)
            desired[2] = self.clear_z
        else:
            desired[:2] = self.goal[:2]
            desired[2] += (self.goal[2] - desired[2]) * ease(elapsed / 90)
        delta = np.clip(desired - pos, -0.085, 0.085)
        rotation = R.from_rotvec(np.clip((self.end_rot * ro.inv()).as_rotvec(), -0.15, 0.15))
        for side in self.sides:
            w = s[side]
            wr = R.from_quat(w[3:], scalar_first=True)
            target = pos + delta + rotation.apply(w[:3] - pos)
            if self.released:
                retreat = s["root"][:3] - pos
                retreat[2] = 0
                retreat /= max(np.linalg.norm(retreat), 0.1)
                target = self.release_wrists[side] + ease((slot.steps - self.release_step) / 50) * (
                    retreat * 0.10 + [0, 0, 0.085]
                )
                if "station_postures" in self.r:
                    # Free the fingers on their own side of the basket before tucking.
                    outward = self.release_wrists[side][:2] - self.start_pos[:2]
                    outward /= max(np.linalg.norm(outward), 0.08)
                    target = (
                        self.release_wrists[side]
                        + ease((slot.steps - self.release_step) / 45) * np.r_[0.10 * outward, 0.11]
                    )
                rotation = R.identity()
            self.arms[side].apply(action, s, w[:3], target, rotation * wr)
