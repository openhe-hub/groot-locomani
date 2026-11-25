"""Side-by-side synchronized physical replay of a matched soap pair, with input and output paths."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess

from kitchen_scene import make_kitchen
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(study, robot, seed, output):
    target = json.loads((study / "results.json").read_text())["dock_target_xy_m"]
    folders = [study / str(seed) / arm for arm in ("text", "pelvis")]
    data = [dict(np.load(f / "soap/rollout.npz")) for f in folders]
    clips = [np.loadtxt(f / "motions/walk/reference.csv", delimiter=",") for f in folders]
    keys = json.loads((folders[1] / "motions/walk/constraints.json").read_text())[0]
    refs = np.asarray(keys["smooth_root_2d"])[:, ::-1]
    base = Image.new("RGB", (1600, 900), "#f1f1ea")
    draw = ImageDraw.Draw(base)

    def text(d, x, y, msg, size, bold=False, color="#162c30"):
        d.text(
            (x, y),
            msg,
            font=ImageFont.truetype("DejaVuSans" + ("-Bold" if bold else "") + ".ttf", size),
            fill=color,
        )

    text(draw, 28, 14, "Kitchen  /  press the soap pump twice", 40, True)
    text(
        draw,
        30,
        73,
        f"Kimodo + SONIC   /   matched seed {seed}   /   identical start, task arm controller and prompt",
        20,
    )
    for i, name in enumerate(("Text only", "Text + pelvis waypoints")):
        x = 24 + i * 792
        text(draw, x, 116, name, 29, True)
        text(draw, x, 651, "Generated walking reference", 22, True)
        text(draw, x, 684, "Start-relative top view   /   first 2.1 s", 17)
    model = make_kitchen(robot, Path.cwd(), soap=True)
    state = mj.MjData(model)
    option = mj.MjvOption()
    option.geomgroup[4:] = 0
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = 31, -15, 3.45
    total = len(data[0]["actual"])
    if len(data[1]["actual"]) != total:
        raise ValueError("Paired trajectories have unequal lengths")
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1600x900",
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with ExitStack() as stack:
        renderer = stack.enter_context(mj.Renderer(model, 472, 768))
        proc = stack.enter_context(subprocess.Popen(cmd, stdin=subprocess.PIPE))
        for t in range(0, total, 2):
            image = base.copy()
            d = ImageDraw.Draw(image)
            text(d, 1450, 22, f"{t / 50:05.2f} s", 25)
            for i, (trace, clip) in enumerate(zip(data, clips)):
                x = 24 + 792 * i
                pos = trace["actual"][t]
                state.qpos[:] = pos
                mj.mj_forward(model, state)
                cam.lookat[:] = [0.60 * pos[0] + 0.40 * 2.4, 0.62 * pos[1] + 0.38 * 0.3, 0.95]
                renderer.update_scene(state, camera=cam, scene_option=option)
                image.paste(Image.fromarray(renderer.render()), (x, 166))

                def project(point):
                    return (x + 14 + 175 * point[0], 764 - 175 * point[1])

                path = [project(q) for q in (clip[:64, :2] - clip[0, :2])]
                d.line(path, fill="#14796d", width=3)
                if i:
                    for point in refs:
                        u, v = project(point)
                        d.ellipse((u - 5, v - 5, u + 5, v + 5), outline="#a24f22", width=3)
                text(
                    d,
                    x,
                    797,
                    "Green: generated pelvis path" + ("   /   circles: input points" if i else ""),
                    16,
                )
                dock = np.linalg.norm(trace["actual"][132, :2] - target)
                status = (
                    "Approaching pump" if t < 132 else f"Dock error at 2.65 s   {dock * 100:.1f} cm"
                )
                text(d, x, 842, status, 21, True)
                text(d, x + 439, 844, f"Pump travel   {trace['progress'][t]:.1f} mm", 19)
            proc.stdin.write(image.tobytes())
        proc.stdin.close()
        if proc.wait():
            raise RuntimeError("FFmpeg failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("study", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--robot",
        type=Path,
        default=Path("GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"),
    )
    a = p.parse_args()
    render(a.study, a.robot, a.seed, a.output)
