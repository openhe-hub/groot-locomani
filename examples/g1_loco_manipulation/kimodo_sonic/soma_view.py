"""Retarget G1 forward kinematics onto the official SOMA 77-joint weighted skin."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation


def numbers(values):
    return " ".join(str(v) for v in np.asarray(values).ravel())


def swing(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cross = np.cross(a, b)
    cosine = np.clip(a @ b, -1, 1)
    if cosine < -0.99999:
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return Rotation.from_quat(np.r_[cross, 1 + cosine]).as_matrix()


class SomaView:
    def __init__(self, archive: Path):
        with np.load(archive) as raw:
            # SOMA uses X left, Y up, Z forward. MuJoCo uses X forward, Y left, Z up.
            self.rest = raw["bind_rig_transform"][:, [2, 0, 1], 3].copy()
            names = list(raw["rig_joint_names"].astype(str))
            self.index = {name: i for i, name in enumerate(names)}
            self.parents = {
                int(child): int(parent) for parent, child in raw["rig_joint_connections"]
            }
            xml = ET.fromstring("""<mujoco><visual><global offwidth="1000" offheight="800"/>
            <headlight ambient=".46 .46 .48" diffuse=".72 .72 .72" specular=".13 .13 .13"/></visual>
            <asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".69 .76 .8" rgb2=".30 .38 .42" width="128" height="768"/>
            <texture name="tile" type="2d" builtin="checker" rgb1=".41 .46 .46" rgb2=".37 .42 .43" width="128" height="128"/>
            <material name="ground" texture="tile" texrepeat="2 2" texuniform="true"/></asset>
            <worldbody><geom type="plane" size="0 0 .1" material="ground"/>
            <light pos="1 -2 4" dir="-.2 .3 -1" directional="true"/></worldbody><deformable/></mujoco>""")
            skin = ET.SubElement(
                xml.find("deformable"),
                "skin",
                name="SOMA",
                rgba=".33 .57 .63 1",
                vertex=numbers(raw["bind_vertices"][:, [2, 0, 1]]),
                face=numbers(raw["faces"]),
            )
            weights, indices = raw["lbs_weights"], raw["lbs_indices"]
            for i, name in enumerate(names):
                ET.SubElement(
                    xml.find("worldbody"),
                    "body",
                    name=name,
                    mocap="true",
                    pos=numbers(self.rest[i]),
                )
                rows, cols = np.where((indices == i) & (weights > 0))
                if len(rows):
                    ET.SubElement(
                        skin,
                        "bone",
                        body=name,
                        bindpos=numbers(self.rest[i]),
                        bindquat="1 0 0 0",
                        vertid=numbers(rows),
                        vertweight=numbers(weights[rows, cols]),
                    )
        self.m = mj.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.d = mj.MjData(self.m)
        self.limbs = []
        for side in ("Left", "Right"):
            b = side.lower()
            for bone, child, prox, dist in [
                ("Leg", "Shin", "hip_pitch", "knee"),
                ("Shin", "Foot", "knee", "ankle_pitch"),
                ("Arm", "ForeArm", "shoulder_pitch", "elbow"),
                ("ForeArm", "Hand", "elbow", "wrist_yaw"),
            ]:
                self.limbs.append(
                    (
                        self.index[side + bone],
                        self.index[side + child],
                        b + "_" + prox + "_link",
                        b + "_" + dist + "_link",
                    )
                )
        self.names = names

    def update(self, robot, state):
        positions = np.empty_like(self.rest)
        rotations = np.empty((len(positions), 3, 3))
        pelvis = state.xmat[robot.body("pelvis").id].reshape(3, 3)
        torso = state.xmat[robot.body("torso_link").id].reshape(3, 3)
        limbs = {
            i: (j, state.xpos[robot.body(dist).id] - state.xpos[robot.body(prox).id])
            for i, j, prox, dist in self.limbs
        }
        for i, name in enumerate(self.names):
            if i == 0:
                positions[i] = state.qpos[:3] * 1.29
                rotations[i] = pelvis
                continue
            p = self.parents[i]
            positions[i] = positions[p] + rotations[p] @ (self.rest[i] - self.rest[p])
            rot = rotations[p]
            if name == "Chest":
                rot = torso
            if i in limbs:
                j, direction = limbs[i]
                rot = swing(rot @ (self.rest[j] - self.rest[i]), direction) @ rot
            if name.endswith("Foot"):
                rot = state.xmat[robot.body(name[:-4].lower() + "_ankle_roll_link").id].reshape(
                    3, 3
                )
            rotations[i] = rot
        self.d.mocap_pos[:] = positions
        self.d.mocap_quat[:] = Rotation.from_matrix(rotations).as_quat(scalar_first=True)
        mj.mj_forward(self.m, self.d)
