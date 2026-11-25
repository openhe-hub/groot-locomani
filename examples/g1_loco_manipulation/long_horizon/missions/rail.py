"""Install the powered, single-axis transfer cart and its fixed dispatch guide."""

import json

import numpy as np
from scipy.spatial.transform import Rotation

from .stations import CART_ID, CART_SHIFT, CART_TAG


def install_rail(cfg, args, assets, gate_receipt):
    from isaaclab.sim import RigidBodyPropertiesCfg, spawn_from_usd
    import omni.usd
    from pxr import Gf, Sdf, UsdPhysics

    row = json.loads(args.manifest.read_text().splitlines()[args.manifest_row_start])
    pos = np.array(row["hanger"]["pos"])
    pos[2] += 0.025
    quat = np.array(row["hanger"]["quat_wxyz"])
    dest = cfg.scene.hanger
    dest.init_state.pos = tuple(map(float, pos))
    dest.init_state.rot = tuple(map(float, quat))
    dest.spawn.usd_path = str(assets / "kit-cart.usda")
    dest.spawn.rigid_props = RigidBodyPropertiesCfg(
        kinematic_enabled=False,
        disable_gravity=True,
        linear_damping=0.2,
        angular_damping=0.5,
        max_depenetration_velocity=1.0,
    )

    def spawn(prim_path, config, translation=None, orientation=None):
        prim = spawn_from_usd(prim_path, config, translation=translation, orientation=orientation)
        stage = omni.usd.get_context().get_stage()
        joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/DispatchRail")
        joint.CreateBody1Rel().SetTargets([Sdf.Path(str(prim.GetPath()))])
        axis = Rotation.from_euler("z", np.arctan2(CART_SHIFT[1], CART_SHIFT[0]))
        joint.CreateAxisAttr("X")
        joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos)))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot0Attr(Gf.Quatf(*map(float, axis.as_quat(scalar_first=True))))
        rel = Rotation.from_quat(quat, scalar_first=True).inv() * axis
        joint.CreateLocalRot1Attr(Gf.Quatf(*map(float, rel.as_quat(scalar_first=True))))
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(float(np.linalg.norm(CART_SHIFT)))
        if not gate_receipt:
            return prim
        from pxr import UsdGeom

        # Fixed transponder receiver at the end of the guide, offset from the cart face.
        station = UsdGeom.Xform.Define(stage, "/World/DispatchReceiver")
        at = (
            pos
            + CART_SHIFT
            + Rotation.from_quat(quat, scalar_first=True).apply(CART_TAG + [-0.085, 0, 0])
        )
        station.AddTranslateOp().Set(Gf.Vec3d(*map(float, at)))
        station.AddOrientOp().Set(Gf.Quatf(*map(float, quat)))
        for name, pivot, half, rgb in [
            ("Post", (-0.042, 0, -at[2] / 2), (0.018, 0.025, at[2] / 2), (0.22, 0.29, 0.32)),
            ("Antenna", (0, 0, 0), (0.012, 0.053, 0.040), (0.07, 0.40, 0.40)),
            ("Indicator", (-0.028, 0, 0.046), (0.015, 0.027, 0.014), (0.86, 0.53, 0.09)),
        ]:
            body = UsdGeom.Cube.Define(stage, "/World/DispatchReceiver/" + name)
            body.CreateSizeAttr(2.0)
            body.AddTranslateOp().Set(Gf.Vec3d(*map(float, pivot)))
            body.AddScaleOp().Set(Gf.Vec3f(*map(float, half)))
            body.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        station.GetPrim().CreateAttribute("mission:expected", Sdf.ValueTypeNames.String).Set(
            CART_ID
        )
        return prim

    dest.spawn.func = spawn
