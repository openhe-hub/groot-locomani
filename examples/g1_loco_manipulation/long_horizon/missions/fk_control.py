"""Calibrated G1 fingertip IK for stationary appliance and console operation."""

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class ToolServo:
    def __init__(self, runtime, side):
        xml = Path(
            os.environ.get(
                "G1_MISSION_ROBOT_XML",
                "/work/GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml",
            )
        )
        if importlib.util.find_spec("mujoco") is None:
            sys.path.append(
                str(
                    xml.parents[3]
                    / "station-env/lib"
                    / f"python{sys.version_info.major}.{sys.version_info.minor}"
                    / "site-packages"
                )
            )
        import mujoco as mj

        self.mj = mj
        if "ik_model" not in runtime:
            runtime["ik_model"] = mj.MjModel.from_xml_path(str(xml))
        self.m = runtime["ik_model"]
        self.d = mj.MjData(self.m)
        self.side = side
        self.robot = runtime["robot"]
        self.body = self.m.body(side + "_wrist_yaw_link").id
        self.joints = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        self.sim = [runtime["sim_joint_names"].index(n) for n in self.joints]
        names = [
            n
            for n in self.joints
            if n.startswith(side + "_") and any(v in n for v in ("shoulder", "elbow", "wrist"))
        ]
        self.sidx = [runtime["sim_joint_names"].index(n) for n in names]
        self.actions = [runtime["action_name_to_idx"][n] for n in names]
        self.qidx = [self.m.joint(n).qposadr[0] for n in names]
        self.dofs = [self.m.joint(n).dofadr[0] for n in names]
        self.limits = (
            self.robot.data.soft_joint_pos_limits[0, self.sidx]
            .detach()
            .cpu()
            .numpy()
            .T.astype(float)
        )
        self.cmd = self.robot.data.joint_pos[0, self.sidx].detach().cpu().numpy().copy()
        self.solution = self.cmd.copy()
        self.bias = np.zeros(7)
        self.rest = np.array([-0.34, 0.38 if side == "left" else -0.38, 0.0, 1.0, 0.0, -0.10, 0.0])
        self.jp = np.zeros((3, self.m.nv))
        self.jr = self.jp.copy()
        import torch

        kp = torch.maximum(
            self.robot.data.default_joint_stiffness[:, self.sidx] * 3,
            torch.tensor([[65.0, 65.0, 55.0, 55.0, 18.0, 16.0, 16.0]], device=self.robot.device),
        )
        kd = torch.maximum(
            self.robot.data.default_joint_damping[:, self.sidx] * 1.8,
            torch.tensor([[3.0, 3.0, 2.4, 2.6, 0.8, 0.7, 0.7]], device=self.robot.device),
        )
        self.robot.write_joint_stiffness_to_sim(kp, joint_ids=self.sidx)
        self.robot.write_joint_damping_to_sim(kd, joint_ids=self.sidx)
        runtime.setdefault("station_arm_gains", {})[side] = self.sidx
        self.kp = kp[0].detach().cpu().numpy()

    def apply(self, action, record, tip, goal, orientation=None):
        import torch

        self.d.qpos[:7] = record["root"][:7]
        self.d.qpos[7:36] = record["joints"][self.sim]
        wrist = record[self.side]
        local = Rotation.from_quat(wrist[3:], scalar_first=True).inv().apply(tip - wrist[:3])
        self.mj.mj_kinematics(self.m, self.d)
        correction = self.d.xpos[self.body] + self.d.xmat[self.body].reshape(3, 3) @ local - tip
        calibrated = goal + correction

        def fk(q):
            self.d.qpos[self.qidx] = q
            self.mj.mj_kinematics(self.m, self.d)
            rot = self.d.xmat[self.body].reshape(3, 3)
            return self.d.xpos[self.body] + rot @ local, rot

        def fun(q):
            at, rot = fk(q)
            v = at - calibrated
            if orientation is not None:
                v = np.r_[v, 0.14 * (Rotation.from_matrix(rot) * orientation.inv()).as_rotvec()]
            return np.r_[v, 0.009 * (q - self.rest)]

        def jac(q):
            at, _ = fk(q)
            self.mj.mj_comPos(self.m, self.d)
            self.mj.mj_jac(self.m, self.d, self.jp, self.jr, at, self.body)
            j = self.jp[:, self.dofs]
            if orientation is not None:
                j = np.vstack([j, 0.14 * self.jr[:, self.dofs]])
            return np.vstack([j, 0.009 * np.eye(7)])

        lo, hi = self.limits

        def solve(x):
            return least_squares(
                fun,
                np.clip(x, lo + 1e-5, hi - 1e-5),
                jac=jac,
                bounds=(lo, hi),
                max_nfev=18,
                ftol=1e-5,
                xtol=1e-4,
                gtol=1e-5,
            )

        res = solve(self.solution)
        if np.linalg.norm(res.fun[:3]) > 0.043:
            alt = solve(self.rest)
            if alt.cost + 1e-5 < res.cost:
                res = alt
        self.solution = res.x
        self.cmd += np.clip(0.60 * (res.x - self.cmd), -0.072, 0.072)
        q = record["joints"][self.sidx]
        self.bias = np.clip(self.bias + 0.008 * (self.cmd - q), -0.16, 0.16)
        v = self.robot.data.joint_vel[0, self.sidx].detach().cpu().numpy()
        gravity = (
            self.robot.root_physx_view.get_gravity_compensation_forces()[
                0, np.asarray(self.sidx) + 6
            ]
            .detach()
            .cpu()
            .numpy()
        )
        target = np.clip(
            self.cmd + self.bias + np.clip(gravity / self.kp, -0.14, 0.14) - 0.022 * v, lo, hi
        )
        action[0, self.actions] = torch.as_tensor(target, device=action.device, dtype=action.dtype)
