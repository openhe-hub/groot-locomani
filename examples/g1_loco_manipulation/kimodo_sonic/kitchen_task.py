"""Compose language-labelled primitives and use scene state to guide the final reach."""

import json

import mujoco as mj
import numpy as np
from stitch import join


SPECS = {
    "soap": ("Walk up to the soap dispenser.", "Press and release the soap pump twice.", 0.62),
    "microwave": (
        "Walk up to the open microwave.",
        "Push the microwave door closed with your right hand.",
        0.25,
    ),
    "groceries": (
        "Walk to the cereal carton at the kitchen island.",
        "Push the cereal carton onto the green receiving mat.",
        0.97,
    ),
}


def prepare(root, name):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    walk = np.loadtxt(root / "motions/walk/reference.csv", delimiter=",")[4:64]
    stand = np.loadtxt(root / "motions/stand/reference.csv", delimiter=",")
    hold = np.tile(stand[40:41], (270 if name == "soap" else 210, 1))
    vals = join(walk, hold, 15)
    np.savetxt(folder / "reference.csv", vals, delimiter=",", fmt="%.9f")
    a, b, _ = SPECS[name]
    meta = dict(
        model="Kimodo-G1-RP-v1",
        seed=42,
        fps=30,
        approach_seconds=2.65,
        prompt_phases=[
            [a, 2.65],
            [b, 5.55 if name == "soap" else 3.55],
            ["Withdraw the hand and stand clear.", 2.30],
        ],
        readout={"microwave": "Door opening", "groceries": "Carton advance", "soap": "Pump stroke"}[
            name
        ],
        control="Kimodo walk/stand, SONIC body tracking, state-guided right-arm IK",
    )
    (folder / "prompt.json").write_text(json.dumps(meta, indent=2) + "\n")


class Task:
    def __init__(self, name, model, data, offset=0):
        self.name = name
        self.m = model
        self.d = data
        side = "right"
        self.side = side
        self.arm = slice(15, 22) if side == "left" else slice(22, 29)
        self.wrist = model.body(side + "_wrist_yaw_link").id
        names = (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
        self.q = np.array([model.joint(f"{side}_{n}_joint").qposadr[0] for n in names])
        self.v = np.array([model.joint(f"{side}_{n}_joint").dofadr[0] for n in names])
        self.limits = np.array(
            [model.jnt_range[model.joint(f"{side}_{n}_joint").id] for n in names]
        )
        self.track = data.qpos[self.q].copy()
        self.tip_local = np.array([0.105, -0.005, 0])
        self.sim = mj.MjData(model)
        self.aim = None
        self.start = None
        self.contacts = []
        self.other = set()
        self.progress = []
        self.first = None
        self.joint = model.joint("microjoint").qposadr[0]
        self.carton = model.joint("carton_free").qposadr[0]
        self.target = model.body(
            {"microwave": "microdoorroot", "groceries": "carton", "soap": "pump_head"}[name]
        ).id
        if name == "soap":
            self.pump = model.joint("pump_slide").qposadr[0]
        data.qpos[self.joint] = -1.06
        data.qpos[0] += (0.03 if name == "groceries" else 0.05) + offset
        data.qpos[1] += SPECS[name][2]
        mj.mj_forward(model, data)
        self.origin = data.qpos[self.carton : self.carton + 3].copy()
        self.qoffset = data.qpos[:2].copy()
        self.report = []

    @staticmethod
    def ramp(t):
        t = np.clip(t, 0, 1)
        return t * t * (3 - 2 * t)

    def tip(self, d):
        return d.xpos[self.wrist] + d.xmat[self.wrist].reshape(3, 3) @ self.tip_local

    def correct(self, sec, desired):
        d = self.d
        m = self.m
        if sec < 2.65:
            return desired
        if self.start is None:
            self.start = self.tip(d).copy()
            self.track = d.qpos[self.q].copy()
            self.initial_target = self.handle().copy()
            if self.name == "microwave":
                self.initial_target = d.xpos[self.target] + d.xmat[self.target].reshape(
                    3, 3
                ) @ np.array([0.36, -0.063, 0])
        if self.name == "soap":
            surface = self.initial_target
            high = self.start + np.array([-0.14, 0, 0.41])
            above = surface + np.array([-0.015, 0, 0.11])
            if sec < 3.55:
                u = self.ramp((sec - 2.65) / 0.9)
                goal = (1 - u) * self.start + u * high
            else:
                u = self.ramp((sec - 3.55) / 0.85)
                goal = (1 - u) * high + u * above
                stroke = sum(
                    self.ramp((sec - begin) / 0.42) - self.ramp((sec - begin - 0.72) / 0.45)
                    for begin in (4.55, 6.22)
                )
                goal = goal + np.array([0.005, 0, -0.18]) * stroke
        else:
            if self.name == "groceries":
                stop = self.origin + np.array([-0.070, 0.00, -0.025])
                dest = stop + np.array([0.37, 0, 0.0])
            else:
                # Transform a material point of the door from hinge coordinates into world space.
                axis = d.xaxis[m.joint("microjoint").id]
                anchor = d.xanchor[m.joint("microjoint").id]
                from scipy.spatial.transform import Rotation

                angle = 1.10 * self.ramp((sec - 3.85) / 1.85)
                target = anchor + Rotation.from_rotvec(axis * angle).apply(
                    self.initial_target - anchor
                )
                tangent = np.cross(axis, target - anchor)
                tangent /= np.linalg.norm(tangent)
                stop = target - 0.10 * tangent
                dest = target + 0.09 * tangent
            approach = self.ramp((sec - 2.65) / 1.12)
            push = self.ramp((sec - 3.77) / 1.80)
            goal = (1 - approach) * self.start + approach * ((1 - push) * stop + push * dest)
            # Lift the palm above the island lip before extending across the counter.
            if sec < 3.85:
                goal[2] += 0.16 * np.sin(np.pi * self.ramp((sec - 2.65) / 1.20))
            if self.name == "groceries":
                ready = np.array([2.01, self.origin[1], 1.01])
                near = self.origin + [-0.136, 0, -0.017]
                if sec < 3.45:
                    u = self.ramp((sec - 2.65) / 0.80)
                    goal = (1 - u) * self.start + u * ready
                elif sec < 4.15:
                    u = self.ramp((sec - 3.45) / 0.70)
                    goal = (1 - u) * ready + u * near
                else:
                    u = self.ramp((sec - 4.15) / 1.85)
                    goal = near + np.array([0.42 * u, 0, -0.028 * u])
        release = self.ramp((sec - (8.2 if self.name == "soap" else 6.2)) / 1.15)
        if self.name == "soap":
            # Clear the paddle horizontally before lowering the wrist past the worktop.
            clear = self.initial_target + np.array([-0.20, 0, 0.10])
            retract = self.ramp((sec - 7.5) / 0.7)
            goal = (1 - retract) * goal + retract * clear
            rest = self.start.copy()
            rest[0] = min(rest[0], clear[0] - 0.04)
            goal = (1 - release) * goal + release * rest
        else:
            goal = (1 - release) * goal + release * self.start
        self.aim = goal
        tmp = self.sim
        tmp.qpos[:] = d.qpos
        tmp.qpos[self.q] = self.track
        for _ in range(6):
            mj.mj_forward(m, tmp)
            tip = self.tip(tmp)
            J = np.zeros((3, m.nv))
            mj.mj_jac(m, tmp, J, None, tip, self.wrist)
            A = J[:, self.v]
            step = A.T @ np.linalg.solve(A @ A.T + 0.0025 * np.eye(3), goal - tip)
            # Damping at singularities and keeping joint targets inside physical limits.
            tmp.qpos[self.q] = np.clip(
                tmp.qpos[self.q] + np.clip(step, -0.14, 0.14),
                self.limits[:, 0] + 0.015,
                self.limits[:, 1] - 0.015,
            )
        self.track += np.clip(tmp.qpos[self.q] - self.track, -0.085, 0.085)
        desired = desired.copy()
        desired[self.arm] = self.track
        return desired

    def handle(self):
        if self.name == "soap":
            return self.d.site("pump_surface").xpos
        return (
            self.d.site("microhandle_site").xpos
            if self.name == "microwave"
            else self.d.qpos[self.carton : self.carton + 3]
        )

    def sample(self, seen, frame):
        d = self.d
        self.contacts.append(any(self.side + "_wrist" in n for n in seen))
        self.other.update(
            n for n in seen if not n.startswith(self.side + "_wrist") and n.endswith("_link")
        )
        if self.contacts[-1] and self.first is None:
            self.first = frame
        if self.name == "soap":
            value = 1000 * d.qpos[self.pump]
        elif self.name == "microwave":
            value = abs(np.rad2deg(d.qpos[self.joint]))
        else:
            value = 100 * (d.qpos[self.carton] - self.origin[0])
        self.progress.append(value)
        self.report.append(
            np.r_[self.tip(d), self.handle(), self.aim if self.aim is not None else self.tip(d)]
        )

    def metrics(self):
        v = np.asarray(self.progress)
        d = self.d
        stable = not any(self.contacts[-25:])
        extra = {}
        if self.name == "soap":
            peaks = []
            returns = []
            peak = 0.0
            for frame, (depth, contact) in enumerate(zip(v, self.contacts)):
                if contact and depth >= 25:
                    peak = max(peak, float(depth))
                if depth <= 5 and peak:
                    peaks.append(peak)
                    returns.append(frame)
                    peak = 0.0
            extra = {
                "completed_strokes": len(peaks),
                "stroke_peaks_mm": peaks,
                "return_frames": returns,
                "peak_travel_mm": float(v.max()),
            }
            passed = len(peaks) == 2 and max(v[-25:]) < 5
        elif self.name == "microwave":
            passed = max(v[-25:]) < 10
        else:
            center = d.qpos[self.carton : self.carton + 3]
            passed = (
                abs(center[0] - 2.46) < 0.09
                and abs(center[1] - 0.78) < 0.11
                and abs(center[2] - 0.895) < 0.025
                and np.linalg.norm(d.qvel[-6:-3]) < 0.04
            )
        return dict(
            **extra,
            initial_readout=v[0],
            final_readout=v[-1],
            hand_contact_steps=sum(self.contacts),
            other_robot_contacts=sorted(self.other),
            released=stable,
            first_contact_s=self.first / 50 if self.first is not None else None,
            task_success=bool(passed and stable and any(self.contacts)),
        )
