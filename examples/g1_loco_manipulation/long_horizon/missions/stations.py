"""Fixed locations and dimensions of the two supplementary workstations (metres, world frame)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Station:
    secure: tuple
    panel: tuple
    kind: str
    radius: float
    limit: float

    def handle(self, fraction):
        centre = np.asarray(self.secure)
        if self.kind == "slide":
            return centre + [0, -self.limit * fraction, 0.035]
        angle = self.limit * fraction
        return centre + [self.radius * np.cos(angle), self.radius * np.sin(angle), 0.035]

    def surface(self, u, v):
        return np.asarray(self.panel) + [u, v, 0]


STATIONS = {
    "home": Station((-1.245, 1.015, 0.925), (0.69, 1.84, 0.89), "slide", 0, 0.095),
    "factory": Station(
        (0.045, 1.39, 0.95), (-0.015, -0.145, 0.895), "hinge", 0.105, np.deg2rad(65)
    ),
}

MARKS = ((-0.093, -0.045), (0.0, 0.045), (0.087, -0.045))

WIPE = tuple(
    (u, v)
    for row, v in enumerate((-0.063, 0.0, 0.063))
    for u in ((-0.100, -0.05, 0.0, 0.05, 0.100) if row != 1 else (0.100, 0.05, 0.0, -0.05, -0.100))
)
CART_SHIFT = np.array([0.10, -0.68, 0.0])
LAMP_SWITCH = np.array([0.12, -0.085, 0.04])


TRAY_CENTRE = np.array([-0.13, -0.045])
TRAY_HALF = np.array([0.37, 0.31])


def cart_support(records):
    """Check the receiving deck itself, including measured cap-to-cart normal force."""
    from scipy.spatial.transform import Rotation

    tail = records[-12:]
    cap = np.array([r["object"] for r in tail])
    cart = np.array([r["receiver"] for r in tail])
    local = (
        Rotation.from_quat(cart[:, 3:7], scalar_first=True).inv().apply(cap[:, :3] - cart[:, :3])
    )
    normal = np.array([r["cart_support_n"] for r in tail])
    # The transfer tray projects beyond the drawer pedestal on the loading side.
    checks = dict(
        settled_window=len(tail) == 12,
        on_receiving_deck=bool(
            np.all(abs(local[:, :2] - TRAY_CENTRE) < TRAY_HALF)
            and np.all((local[:, 2] > 0.78) & (local[:, 2] < 1.02))
        ),
        supported_by_cart=bool(normal.mean() > 0.06 and normal.min() > 0.015),
        linear_settled=bool(np.linalg.norm(cap[:, 7:10], axis=1).max() < 0.025),
        angular_settled=bool(np.linalg.norm(cap[:, 10:], axis=1).max() < 0.32),
    )
    return dict(
        ok=all(checks.values()),
        checks=checks,
        min_support_n=float(normal.min()),
        cap_local=local[-1].tolist(),
    )


BENCH_CENTRE = np.array([-0.15, 0.12, 0.0])
BENCH_HALF = np.array([0.54, 0.46])
BENCH_HEIGHT = 0.735


def bench_support(records, evaluator, cfg):
    from scipy.spatial.transform import Rotation

    # Include the initial door angle as well as all frames in the release window.
    r = [records[0], *records[-12:]]
    obj = np.array([e["object"] for e in r])
    receiver = np.array([e["receiver"] for e in r])
    receiver[:, :3] += Rotation.from_quat(receiver[:, 3:7], scalar_first=True).apply(
        np.tile(BENCH_CENTRE, (len(r), 1))
    )
    payload = dict(
        object_root_pose=obj[:, :7],
        object_root_velocity=obj[:, 7:],
        placement_root_pose=receiver[:, :7],
        door_joint_position=np.array([e["door"] for e in r]),
        door_contact_latched=np.array([e["door_latched"] for e in r]),
        robot_hand_open=np.array([e["hand"] for e in r]),
    )
    m = evaluator(
        payload,
        dict(cfg, stool_seat_half_xy_m=(BENCH_HALF - 0.02).tolist(), stool_seat_z_m=BENCH_HEIGHT),
    )
    normal = np.array([e["bench_support_n"] for e in records[-12:]])
    m["checks"].update(
        catch_surface_contact=bool(len(normal) == 12 and normal.min() > 0.008),
        door_closed=bool(m["door_closed_angle_reached"]),
    )
    m["min_support_n"] = float(normal.min())
    m["ok"] = all(m["checks"].values())
    return m


TAG_CODES = ("PPE-17", "QA-B4", "DOCK-02")
START_UV = (-0.09, 0.079)

CART_TAG = np.array([-0.31, -0.07, 0.68])
CART_ID = "DISPATCH-PPE-17"
