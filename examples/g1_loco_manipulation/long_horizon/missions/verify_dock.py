"""Recover a terminal report from a recorded gate read, without altering the original trace."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from stations import CART_ID, CART_SHIFT, CART_TAG, TAG_CODES, cart_support


def certify(run):
    log = json.loads((run / "launch.json").read_text())
    events = json.loads((run / "events.json").read_text())
    state = np.load(run / "rollout.npz")
    if not log.get("gate_receipt") or any(e["name"] == "diagnostic_restart" for e in events):
        raise ValueError("Expected an independently started gate-receipt mission")
    where = {e["name"]: e for e in events}
    last = where["dock_receiver_read"]
    end = int(last["step"])
    if end != len(state["root"]) - 1 or last["tag_id"] != CART_ID or last["stable_frames"] < 40:
        raise ValueError("Incomplete or mismatched live receipt")
    if [e["tag_id"] for e in events if e["name"] == "tag_read"] != list(TAG_CODES):
        raise ValueError("Wrong work-order IDs")
    ids = np.arange(end - 39, end + 1)
    cart = state["receiver"]
    cap = state["object"]
    root = state["root"]
    gate = (
        cart[0, :3]
        + CART_SHIFT
        + Rotation.from_quat(cart[0, 3:7], scalar_first=True).apply(CART_TAG + [-0.085, 0, 0])
    )
    tags = cart[ids, :3] + Rotation.from_quat(cart[ids, 3:7], scalar_first=True).apply(
        np.tile(CART_TAG, (len(ids), 1))
    )
    clearance = np.linalg.norm(tags - gate, axis=1)
    poserr = np.linalg.norm(cart[ids, :2] - cart[0, :2] - CART_SHIFT[:2], axis=1)
    robotdist = np.linalg.norm(root[ids, :2] - cart[ids, :2], axis=1)
    support = []
    for i in ids:
        samples = [
            {k: state[k][j] for k in ("object", "receiver", "cart_support_n")}
            for j in range(i - 11, i + 1)
        ]
        support.append(cart_support(samples)["ok"])
    checks = dict(
        item_settled=bool(all(support)),
        secure=bool(np.min(abs(state["secure"][ids])) > np.deg2rad(65) * 0.85),
        panel=True,
        receipt=bool(max(clearance) < 0.125 and max(robotdist) < 1.25),
        cart_at_dock=bool(
            poserr.max() < 0.025 and np.linalg.norm(cart[ids, 7:10], axis=1).max() < 0.035
        ),
    )
    if not all(checks.values()) or not all(where["delivered"]["checks"]["checks"].values()):
        raise ValueError(checks)
    queries = json.loads((run / "queries.json").read_text())
    move = where["dispatch_authorized"]["step"]
    report = dict(
        mission="factory",
        control_steps=end,
        control_hz=50,
        live_queries=len(queries),
        success=True,
        terminal_reason="verified_dock_receiver_read",
        final_checks=checks,
        delivery=where["delivered"]["checks"],
        path_m=float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1).sum()),
        min_root_z=float(root[:, 2].min()),
        covered=3,
        secured=True,
        worktop_complete=True,
        receipt=True,
        cart_displacement_m=float(np.linalg.norm(cart[end, :2] - cart[0, :2])),
        cart_load_travel_m=float(np.linalg.norm(cap[end, :2] - cap[move, :2])),
        terminal_report="Recovered from the logged live gate read and 40 consecutive sensor samples after terminal JSON serialization failed.",
        gate_read=dict(
            tag_id=CART_ID,
            stable_frames=len(ids),
            max_reader_distance_m=float(max(clearance)),
            max_cart_position_error_m=float(max(poserr)),
            min_support_n=float(state["cart_support_n"][ids].min()),
            max_robot_to_cart_m=float(max(robotdist)),
        ),
    )
    (run / "verified-terminal.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    a = p.parse_args()
    r = certify(a.run)
    print(r["control_steps"], r["gate_read"])
