"""Author passive workcell mechanics, controls and a furnished factory stage for the live camera."""

import argparse
import math
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from stations import BENCH_CENTRE, BENCH_HALF, BENCH_HEIGHT, LAMP_SWITCH, MARKS, STATIONS, WIPE


def stage_at(path, name):
    s = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(s, "/" + name)
    s.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageUpAxis(s, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(s, 1.0)
    return s


def shape(s, path, pos, half, color, cylinder=False, collision=False):
    if cylinder:
        g = UsdGeom.Cylinder.Define(s, path)
        g.CreateRadiusAttr(1.0)
        g.CreateHeightAttr(2.0)
    else:
        g = UsdGeom.Cube.Define(s, path)
        g.CreateSizeAttr(2.0)
    g.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
    g.AddScaleOp().Set(Gf.Vec3f(*map(float, half)))
    g.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    if collision:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g


def drive(s, root, origin, kind, axis, limit):
    # Fixed base articulation so its measured DOF is available in PhysX.
    base = UsdGeom.Xform.Define(s, root + "/Base")
    mobile = UsdGeom.Xform.Define(s, root + "/Handle")
    for link in (base, mobile):
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass.CreateMassAttr(0.18)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.002, 0.002, 0.002))
        link.AddTranslateOp().Set(Gf.Vec3d(*origin))
    joint = UsdPhysics.FixedJoint.Define(s, root + "/Anchor")
    joint.CreateBody1Rel().SetTargets([base.GetPath()])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
    j = (UsdPhysics.PrismaticJoint if kind == "slide" else UsdPhysics.RevoluteJoint).Define(
        s, root + "/travel"
    )
    j.CreateBody0Rel().SetTargets([base.GetPath()])
    j.CreateBody1Rel().SetTargets([mobile.GetPath()])
    j.CreateAxisAttr(axis)
    j.CreateLowerLimitAttr(min(0.0, limit))
    j.CreateUpperLimitAttr(max(0.0, limit))
    d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if kind == "slide" else "angular")
    d.CreateTypeAttr("force")
    d.CreateDampingAttr(8.0 if kind == "slide" else 0.10)
    d.CreateStiffnessAttr(160.0 if kind == "slide" else 0.9)
    d.CreateMaxForceAttr(32.0)
    return str(base.GetPath()), str(mobile.GetPath())


STEEL = (0.19, 0.27, 0.29)
LIGHT = (0.70, 0.69, 0.61)
WOOD = (0.52, 0.31, 0.13)
GRIP = (0.89, 0.44, 0.09)
TEAL = (0.05, 0.48, 0.48)


def build(name, folder, hall, cabinet):
    spec = STATIONS[name]
    s = stage_at(folder / f"{name}-secure.usda", "Mechanism")
    home = name == "home"
    base, moving = drive(
        s,
        "/Mechanism",
        (0.0, 0.0, 0.0),
        spec.kind,
        "Y" if home else "Z",
        -spec.limit if home else math.degrees(spec.limit),
    )
    if home:
        # Small covered bread bin at the delivery end. A wooden lid slides over its opening.
        shape(s, base + "/Support", (0, -0.12, -0.46), (0.15, 0.13, 0.44), WOOD)
        shape(s, base + "/Housing", (0, -0.12, -0.074), (0.18, 0.155, 0.065), LIGHT)
        shape(s, base + "/Recess", (0, -0.11, -0.006), (0.151, 0.11, 0.006), (0.05, 0.09, 0.09))
        shape(s, moving + "/Lid", (0, 0, 0), (0.157, 0.107, 0.010), WOOD)
        shape(s, moving + "/Grip", (0, 0, 0.032), (0.052, 0.014, 0.020), STEEL)
        shape(s, base + "/Switch", LAMP_SWITCH, (0.024, 0.024, 0.01), TEAL, True)
        shape(s, base + "/Stem", (0.145, -0.22, 0.12), (0.012, 0.012, 0.12), STEEL, True)
        shape(
            s,
            base + "/Shade",
            (0.145, -0.22, 0.27),
            (0.065, 0.065, 0.044),
            (0.89, 0.62, 0.25),
            True,
        )
        bulb = UsdLux.SphereLight.Define(s, base + "/Bulb")
        bulb.AddTranslateOp().Set(Gf.Vec3d(0.145, -0.22, 0.23))
        bulb.CreateRadiusAttr(0.035)
        bulb.CreateIntensityAttr(15000.0)
        bulb.CreateColorAttr((1.0, 0.72, 0.37))
    else:
        shape(s, base + "/GuardPost", (0, 0, -0.48), (0.055, 0.055, 0.46), STEEL)
        shape(s, base + "/Box", (0, 0, -0.052), (0.14, 0.13, 0.045), LIGHT)
        shape(s, moving + "/Arm", (spec.radius / 2, 0, 0), (spec.radius / 2, 0.016, 0.010), GRIP)
        shape(s, moving + "/Roller", (spec.radius, 0, 0.023), (0.019, 0.019, 0.029), STEEL, True)
        shape(s, base + "/Spindle", (0, 0, 0), (0.025, 0.025, 0.024), TEAL, True)
    s.GetRootLayer().Save()
    p = stage_at(folder / f"{name}-panel.usda", "Panel")
    x, y, z = spec.panel
    shape(
        p,
        "/Panel/Stand",
        (x, y, z / 2 - 0.04),
        (0.125, 0.13, z / 2 - 0.04),
        WOOD if home else STEEL,
    )
    shape(
        p,
        "/Panel/Counter",
        (x, y, z - 0.02),
        (0.178, 0.135, 0.018),
        (0.53, 0.61, 0.59) if home else LIGHT,
    )
    from stations import START_UV, TAG_CODES

    for i, (u, v) in enumerate(WIPE if home else MARKS):
        shape(
            p,
            f"/Panel/Mark{i}",
            (x + u, y + v, z + 0.001),
            (0.019 if home else 0.026, 0.021 if home else 0.026, 0.0018),
            (0.27, 0.12, 0.035) if home else (0.06, 0.11, 0.12),
            True,
        )
        if not home:
            p.GetPrimAtPath(f"/Panel/Mark{i}").CreateAttribute(
                "mission:tagId", Sdf.ValueTypeNames.String
            ).Set(TAG_CODES[i])
            for a in range(3 + i):
                shape(
                    p,
                    f"/Panel/Bar{i}_{a}",
                    (x + u - 0.014 + 0.009 * a, y + v, z + 0.003),
                    (0.002, 0.013, 0.001),
                    LIGHT,
                )
    if home:
        shape(
            p,
            "/Panel/Pad",
            (x + MARKS[0][0], y + MARKS[0][1], z + 0.012),
            (0.037, 0.025, 0.009),
            (0.09, 0.48, 0.59),
        )
    else:
        shape(
            p,
            "/Panel/Start",
            (x + START_UV[0], y + START_UV[1], z + 0.024),
            (0.030, 0.030, 0.012),
            TEAL,
            True,
        )
    p.GetRootLayer().Save()
    if home:
        bench = stage_at(folder / "serving-bench.usda", "ServingBench")
        body = UsdGeom.Xform.Define(bench, "/ServingBench/mesh")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(25.0)
        px, py, _ = BENCH_CENTRE
        shape(
            bench,
            "/ServingBench/mesh/OakTop",
            (px, py, BENCH_HEIGHT - 0.02),
            (*BENCH_HALF, 0.020),
            WOOD,
            collision=True,
        )
        for j, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            shape(
                bench,
                f"/ServingBench/mesh/Leg{j}",
                (
                    px + sx * (BENCH_HALF[0] - 0.045),
                    py + sy * (BENCH_HALF[1] - 0.045),
                    (BENCH_HEIGHT - 0.035) / 2,
                ),
                (0.028, 0.028, (BENCH_HEIGHT - 0.035) / 2),
                WOOD,
                collision=True,
            )
        shape(
            bench,
            "/ServingBench/mesh/Apron",
            (px, py, BENCH_HEIGHT - 0.07),
            (0.49, 0.407, 0.026),
            (0.36, 0.21, 0.10),
        )
        bench.GetRootLayer().Save()
    if not home:
        cart = stage_at(folder / "kit-cart.usda", "Cart")
        cart.GetDefaultPrim().GetReferences().AddReference(
            os.path.relpath(cabinet.resolve(), folder.resolve()), "/World"
        )
        shape(cart, "/Cart/Chassis", (0, 0, 0.075), (0.328, 0.326, 0.037), STEEL)
        for ix, xw in enumerate((-0.28, 0.28)):
            for iy, yw in enumerate((-0.293, 0.293)):
                wheel = shape(
                    cart,
                    f"/Cart/Wheel{ix}{iy}",
                    (xw, yw, 0.059),
                    (0.059, 0.059, 0.039),
                    (0.03, 0.04, 0.045),
                    True,
                )
                tr, scale = wheel.GetOrderedXformOps()
                turn = wheel.AddRotateXOp()
                turn.Set(90.0)
                wheel.SetXformOpOrder([tr, turn, scale])
        shape(cart, "/Cart/Stripe", (-0.30, 0, 0.58), (0.005, 0.24, 0.032), TEAL)
        # Padded loading tray with a shallow retaining rim on the powered cart.
        from stations import TRAY_CENTRE, TRAY_HALF

        tx, ty = TRAY_CENTRE
        hx, hy = TRAY_HALF
        shape(
            cart,
            "/Cart/LoadingDeck",
            (tx, ty, 0.805),
            (hx, hy, 0.007),
            (0.21, 0.35, 0.37),
            collision=True,
        )
        for sign in (-1, 1):
            suffix = "A" if sign < 0 else "B"
            shape(
                cart,
                "/Cart/End" + suffix,
                (tx + sign * hx, ty, 0.820),
                (0.007, hy, 0.013),
                STEEL,
                collision=True,
            )
            shape(
                cart,
                "/Cart/Side" + suffix,
                (tx, ty + sign * hy, 0.820),
                (hx, 0.007, 0.013),
                STEEL,
                collision=True,
            )
        from stations import CART_ID, CART_TAG

        transponder = shape(
            cart, "/Cart/DispatchTag", CART_TAG, (0.008, 0.071, 0.046), (0.035, 0.23, 0.25)
        )
        transponder.GetPrim().CreateAttribute("mission:tagId", Sdf.ValueTypeNames.String).Set(
            CART_ID
        )
        for j, width in enumerate((0.003, 0.006, 0.003, 0.008, 0.005)):
            pos = CART_TAG + [-0.009, -0.041 + 0.018 * j, 0]
            shape(cart, f"/Cart/TagBar{j}", pos, (0.002, width, 0.026), LIGHT)
        cart.GetRootLayer().Save()
        s = stage_at(folder / "dispatch-room.usda", "Factory")
        library = UsdGeom.Xform.Define(s, "/Factory/Workshop")
        library.GetPrim().GetReferences().AddReference(
            os.path.relpath(hall.resolve(), folder.resolve()), "/World"
        )
        for c in list(library.GetPrim().GetChildren()):
            if c.GetName() not in ("Hall", "Ambient", "Looks"):
                c.SetActive(False)
        # Translate the work zone to the middle bay of the furnished room.
        library.AddTranslateOp().Set(Gf.Vec3d(-11.4, -3.65, 0))
        # The native pick pedestal and mobile destination fit within the marked work zone.
        for j, (px, py) in enumerate(((0.9, 1.35), (1.03, 0.67))):
            for k in (-1, 1):
                shape(
                    s,
                    f"/Factory/Dock{j}_{'a' if k == -1 else 'b'}",
                    (px, py + k * 0.43, 0.016),
                    (0.42, 0.012, 0.002),
                    GRIP,
                )
        s.GetRootLayer().Save()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hall", type=Path, required=True)
    p.add_argument("--cabinet", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name in STATIONS:
        build(name, args.output, args.hall, args.cabinet)
