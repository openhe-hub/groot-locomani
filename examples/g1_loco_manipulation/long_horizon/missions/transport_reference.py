"""Extract only a loaded motion segment from a measured online GR00T transport trial."""

import argparse
import json
from pathlib import Path

import numpy as np


def export(run, out, end):
    records = np.load(run / "rollout.npz")
    start = int(np.flatnonzero(records["object"][:, 2] - records["object"][0, 2] > 0.10)[0])
    if not start + 100 < end < len(records["root"]):
        raise ValueError("Invalid loaded interval")
    sl = slice(start, end + 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        root=records["root"][sl, :7],
        joints=records["joints"][sl],
        object=records["object"][sl, :7],
        joint_names=records["joint_names"],
        source_steps=[start, end],
        scenario=json.loads((run / "launch.json").read_text())["rows"][0],
        source_run=run.name,
        control_hz=50,
    )
    return start, end


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--end", type=int, required=True)
    a = p.parse_args()
    print(export(a.run, a.out, a.end))
