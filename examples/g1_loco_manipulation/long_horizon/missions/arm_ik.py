"""Damped arm IK from measured PhysX screw axes; leaves the floating base actuated by SONIC."""

import numpy as np
from scipy.spatial.transform import Rotation as R


class ArmIK:
    def __init__(self, runtime, side):
        import sonic_policy as sp

        self.robot = runtime["robot"]
        self.side = side
        self.wrist = runtime[side + "_wrist"]
        names = [
            n
            for n in sp.G1_MUJOCO_BODY_JOINT_NAMES
            if n.startswith(side + "_") and any(k in n for k in ("shoulder", "elbow", "wrist"))
        ]
        self.jidx = [runtime["sim_joint_names"].index(n) for n in names]
        self.aidx = [runtime["action_name_to_idx"][n] for n in names]
        self.limits = self.robot.data.soft_joint_pos_limits[0, self.jidx].detach().cpu().numpy()
        self.cmd = self.robot.data.joint_pos[0, self.jidx].detach().cpu().numpy().copy()
        self.bias = np.zeros(7)
        robot = self.robot
        kp = robot.data.default_joint_stiffness[:, self.jidx] * 2.8
        kd = robot.data.default_joint_damping[:, self.jidx] * 1.75
        robot.write_joint_stiffness_to_sim(kp, joint_ids=self.jidx)
        robot.write_joint_damping_to_sim(kd, joint_ids=self.jidx)
        self.kp = kp[0].detach().cpu().numpy().copy()
        runtime.setdefault("station_arm_gains", {})[side] = self.jidx

    def apply(self, action, record, tip, goal, orientation=None):
        import torch

        J = (
            self.robot.root_physx_view.get_jacobians()[0, self.wrist]
            .detach()
            .cpu()
            .numpy()[:, np.array(self.jidx) + 6]
        )
        wrist = record[self.side]
        off = tip - wrist[:3]
        x, y, z = off
        jac = J[:3] - np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]]) @ J[3:]
        q = record["joints"][self.jidx]
        axes = J[3:].T
        origins = tip + np.cross(axes, jac.T)
        trial = self.cmd.copy()
        for _ in range(6):
            transform = np.eye(3)
            offset = np.zeros(3)
            pivots = []
            directions = []
            turns = R.from_rotvec(axes * (trial - q)[:, None]).as_matrix()
            for direction, origin, rot in zip(axes, origins, turns):
                pivot = transform @ origin + offset
                pivots.append(pivot)
                directions.append(transform @ direction)
                transform = transform @ rot
                offset = pivot - transform @ origin
            reached = transform @ tip + offset
            j = np.cross(directions, reached - np.asarray(pivots)).T
            err = goal - reached
            if orientation is not None:
                predicted = R.from_matrix(transform) * R.from_quat(wrist[3:], scalar_first=True)
                err = np.r_[err, 0.19 * (orientation * predicted.inv()).as_rotvec()]
                j = np.vstack([j, 0.19 * np.asarray(directions).T])
            if np.linalg.norm(err) < 0.002:
                break
            trial = np.clip(
                trial
                + np.clip(
                    j.T @ np.linalg.solve(j @ j.T + np.eye(len(err)) * 0.0012, err), -0.15, 0.15
                ),
                self.limits[:, 0],
                self.limits[:, 1],
            )
        self.cmd += np.clip(0.65 * (trial - self.cmd), -0.075, 0.075)
        self.bias = np.clip(self.bias + 0.014 * (self.cmd - q), -0.30, 0.30)
        vel = self.robot.data.joint_vel[0, self.jidx].detach().cpu().numpy()
        gravity = (
            self.robot.root_physx_view.get_gravity_compensation_forces()[
                0, np.asarray(self.jidx) + 6
            ]
            .detach()
            .cpu()
            .numpy()
        )
        compensation = np.clip(gravity / np.maximum(self.kp, 1.0), -0.25, 0.25)
        action[0, self.aidx] = torch.as_tensor(
            np.clip(
                self.cmd + self.bias + compensation - 0.035 * vel,
                self.limits[:, 0],
                self.limits[:, 1],
            ),
            device=action.device,
            dtype=action.dtype,
        )
