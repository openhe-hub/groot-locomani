"""Track a Kimodo CSV with SONIC and record a free-base MuJoCo rollout at 50 Hz."""

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation


TRAVEL = 0.18


def box(body, name, pos, size, color):
    return ET.SubElement(
        body,
        "geom",
        name=name,
        type="box",
        pos=" ".join(map(str, pos)),
        size=" ".join(map(str, size)),
        rgba=color,
    )


def scene(robot_path, drawer):
    root = ET.parse(robot_path).getroot()
    root.find("compiler").set("meshdir", str(robot_path.resolve().parent / "meshes"))
    root.find("visual/global").attrib.update(offwidth="800", offheight="720")
    root.find("visual/headlight").attrib.update(ambient=".45 .45 .45", specular=".15 .15 .15")
    root.find(".//texture[@type='skybox']").attrib.update(
        rgb1=".63 .72 .77", rgb2=".38 .49 .56", builtin="gradient"
    )
    root.find(".//texture[@name='groundplane']").attrib.update(
        rgb1=".38 .43 .45", rgb2=".32 .37 .39", markrgb=".46 .51 .53"
    )
    root.find(".//material[@name='groundplane']").attrib.update(texrepeat="2 2", reflectance=".04")
    if drawer is not None:
        x, y, z = drawer
        world = root.findall("worldbody")[-1]
        cabinet = ET.SubElement(world, "body", name="workbench", pos=f"{x} {y} {z}")
        wood, metal = ".62 .43 .25 1", ".19 .25 .28 1"
        box(cabinet, "bench_top", (0.44, 0, 0.12), (0.25, 0.31, 0.022), wood)
        for side in [-1, 1]:
            box(cabinet, f"side_{side}", (0.44, side * 0.285, -0.34), (0.24, 0.018, 0.44), metal)
        box(cabinet, "lower_shelf", (0.44, 0, -0.45), (0.24, 0.27, 0.019), wood)
        moving = ET.SubElement(cabinet, "body", name="drawer")
        ET.SubElement(
            moving,
            "joint",
            name="drawer_slide",
            type="slide",
            axis="1 0 0",
            range=f"0 {TRAVEL}",
            damping="6",
            frictionloss="1.2",
        )
        box(moving, "drawer_front", (0.016, 0, 0), (0.016, 0.26, 0.085), ".17 .47 .49 1")
        box(moving, "drawer_bottom", (0.21, 0, -0.07), (0.18, 0.248, 0.008), wood)
        for side in [-1, 1]:
            box(
                moving,
                f"drawer_wall_{side}",
                (0.21, side * 0.25, 0.002),
                (0.18, 0.008, 0.074),
                wood,
            )
        box(moving, "pull", (-0.018, 0, 0.01), (0.018, 0.095, 0.013), ".8 .82 .81 1")
    return mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


def rollout(args, *, use_cuda=False, reset_pose=None):
    sys.path.insert(0, str(args.zero / "runtime/scripts/sonic_kimodo_run"))
    import sonic_policy as sp

    class Tracker(sp.SonicPolicy):
        def _make_session(self, model_path, use_cuda):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            provider = "CUDAExecutionProvider" if use_cuda else "CPUExecutionProvider"
            providers = [(provider, {"use_tf32": "0"})] if use_cuda else [provider]
            session = ort.InferenceSession(str(model_path), options, providers=providers)
            if session.get_providers()[0] != provider:
                raise RuntimeError(f"Could not initialize {provider} for {model_path.name}")
            return session

    folder = args.folder
    source = np.loadtxt(folder / "reference.csv", delimiter=",")
    heading = Rotation.from_quat(source[0, 3:7], scalar_first=True).as_euler("ZYX")[0]
    align = Rotation.from_euler("z", -heading)
    source[:, :2] -= source[0, :2].copy()
    source[:, :3] = align.apply(source[:, :3])
    source[:, 3:7] = (align * Rotation.from_quat(source[:, 3:7], scalar_first=True)).as_quat(
        scalar_first=True
    )
    ref = sp.ReferenceMotion.from_qpos(source)
    policy = Tracker(ref, args.zero / "assets/shared/sonic", use_cuda=use_cuda)

    kitchen = getattr(args, "kitchen", None)
    if kitchen:
        from kitchen_scene import make_kitchen
        from kitchen_task import Task

        m = make_kitchen(args.robot, Path.cwd(), soap=kitchen == "soap")
    else:
        m = scene(args.robot, args.drawer)
    m.opt.timestep = 0.002
    d = mj.MjData(m)
    names = sp.G1_MUJOCO_BODY_JOINT_NAMES
    qidx = np.array([m.joint(n).qposadr[0] for n in names])
    vidx = np.array([m.joint(n).dofadr[0] for n in names])
    actuators = np.array([m.actuator(n).id for n in names])
    large, medium, small, wrist = (
        sp.ARMATURE_7520_22,
        sp.ARMATURE_7520_14,
        sp.ARMATURE_5020,
        sp.ARMATURE_4010,
    )
    armature = np.array(
        [large, large, medium, large, small, small] * 2
        + [medium, small, small]
        + [small] * 5
        + [wrist] * 2
        + [small] * 5
        + [wrist] * 2
    )
    kp = armature * (2 * np.pi * 10) ** 2
    kd = armature * 4 * (2 * np.pi * 10)
    for i in [4, 5, 10, 11, 13, 14]:
        kp[i] *= 2
        kd[i] *= 2
    m.dof_armature[vidx] = armature
    initial = source[0] if reset_pose is None else reset_pose
    d.qpos[:7] = initial[:7]
    d.qpos[qidx] = initial[7:]
    mj.mj_forward(m, d)
    feet = [
        i
        for i in range(m.ngeom)
        if m.geom_type[i] == mj.mjtGeom.mjGEOM_SPHERE
        and "ankle_roll" in m.body(m.geom_bodyid[i]).name
    ]
    d.qpos[2] -= min(d.geom_xpos[i, 2] - m.geom_size[i, 0] for i in feet)
    mj.mj_forward(m, d)

    def state():
        q = d.qpos[:7].copy()
        joints = d.qpos[qidx][sp.MUJOCO_INDEX_FOR_ISAACLAB]
        return sp.RobotState(
            q[3:],
            joints - sp.G1_DEFAULT_ANGLES_ISAACLAB,
            d.qvel[vidx][sp.MUJOCO_INDEX_FOR_ISAACLAB].copy(),
            d.qvel[3:6].copy(),
            sp.projected_gravity_from_root_quat(q[3:]),
        )

    task = Task(kitchen, m, d, args.start_offset) if kitchen else None
    policy.reset(state())
    actual, target, wrists, contact, drawer_pos, hands, other = [], [], [], [], [], [], []
    airborne_steps = 0
    foot_bodies = {m.body(f"{side}_ankle_roll_link").id for side in ("left", "right")}
    floor_id = m.geom("floor").id
    right_id = m.body("right_wrist_yaw_link").id
    drawer_id = m.body("drawer").id if args.drawer else -1
    for t in range(ref.num_frames):
        desired = policy.act(state(), frame_idx=t).q_target_isaaclab[sp.ISAACLAB_INDEX_FOR_MUJOCO]
        if task and not args.approach_only:
            desired = task.correct(t / 50, desired)
        touches, nonhand = set(), set()
        seen = set()
        for _ in range(10):
            d.ctrl[actuators] = kp * (desired - d.qpos[qidx]) - kd * d.qvel[vidx]
            if task and task.aim is not None:
                ids = actuators[task.arm]
                err = desired[task.arm] - d.qpos[qidx[task.arm]]
                d.ctrl[ids] = (
                    3.5 * kp[task.arm] * err
                    - 1.8 * kd[task.arm] * d.qvel[vidx[task.arm]]
                    + d.qfrc_bias[vidx[task.arm]]
                )
            mj.mj_step(m, d)
            if task:
                for pair in d.contact:
                    a, b = m.geom_bodyid[pair.geom1], m.geom_bodyid[pair.geom2]
                    if task.target in (a, b):
                        seen.add(m.body(b if a == task.target else a).name)
            if drawer_id >= 0:
                for pair in d.contact:
                    bodies = (m.geom_bodyid[pair.geom1], m.geom_bodyid[pair.geom2])
                    if drawer_id in bodies:
                        name = m.body(bodies[1] if bodies[0] == drawer_id else bodies[0]).name
                        if "wrist" in name:
                            touches.add(name)
                        elif name not in ("drawer", "workbench", "world"):
                            nonhand.add(name)
        grounded = any(
            floor_id in (c.geom1, c.geom2)
            and bool(foot_bodies.intersection((m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])))
            for c in d.contact
        )
        airborne_steps += int(not grounded)
        actual.append(d.qpos.copy())
        target.append(
            np.r_[
                ref.root_pos[t],
                ref.root_quat_wxyz[t],
                ref.joint_pos_isaaclab[t, sp.ISAACLAB_INDEX_FOR_MUJOCO],
            ]
        )
        if task:
            target[-1][:2] += task.qoffset
            if task.aim is not None:
                target[-1][7:][task.arm] = desired[task.arm]
            task.sample(seen, t)
        wrists.append(d.xpos[right_id].copy())
        contact.append(bool(touches))
        hands.extend(touches)
        other.extend(nonhand)
        if args.drawer:
            drawer_pos.append(float(d.qpos[-1]))
    actual, target = np.array(actual), np.array(target)
    upright = Rotation.from_quat(actual[:, 3:7], scalar_first=True).as_matrix()[:, 2, 2]
    metrics = dict(
        steps=len(actual),
        control_hz=50,
        minimum_pelvis_height_m=float(actual[:, 2].min()),
        minimum_pelvis_up_z=float(upright.min()),
        pelvis_height_range_m=float(np.ptp(actual[:, 2])),
        root_displacement_m=float(np.linalg.norm(actual[-1, :2] - actual[0, :2])),
        joint_rmse_rad=float(np.mean((target[:, 7:] - actual[:, qidx]) ** 2) ** 0.5),
        upright=bool(upright.min() > 0.5),
        airborne_steps=airborne_steps,
        max_right_knee_flexion_rad=float(actual[:, qidx[9]].max()),
    )
    if args.drawer:
        metrics.update(
            drawer_origin=args.drawer,
            travel_m=TRAVEL,
            final_travel_m=drawer_pos[-1],
            wrist_contact_steps=sum(contact),
            contact_bodies=sorted(set(hands)),
            other_robot_contacts=sorted(set(other)),
            closed=bool(min(drawer_pos[-25:]) >= TRAVEL - 0.015),
        )
    if task:
        metrics.update(
            task.metrics(), start_offset_m=args.start_offset, approach_only=args.approach_only
        )
        metrics["task_success"] &= metrics["upright"]
    out = args.output or folder
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "rollout.npz",
        reference=target,
        actual=actual,
        right_wrist=wrists,
        hand_contact=contact,
        drawer=drawer_pos,
        **(
            dict(progress=task.progress, task_hand_contact=task.contacts, interaction=task.report)
            if task
            else {}
        ),
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    p.add_argument("--zero", type=Path, required=True)
    p.add_argument("--robot", type=Path, required=True)
    p.add_argument("--drawer", type=float, nargs=3, metavar=("X", "Y", "Z"))
    p.add_argument("--output", type=Path)
    p.add_argument("--kitchen", choices=("microwave", "groceries", "soap"))
    p.add_argument("--start-offset", type=float, default=0)
    p.add_argument(
        "--approach-only",
        action="store_true",
        help="Run the locomotion reference with the task arm controller disabled",
    )
    rollout(p.parse_args())
