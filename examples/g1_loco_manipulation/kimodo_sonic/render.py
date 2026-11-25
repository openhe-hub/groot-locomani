"""Render synchronized Kimodo reference and recorded SONIC simulation states."""

import argparse
import json
import os
from pathlib import Path
import subprocess


os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from track import TRAVEL, scene


W, H = 1280, 720
PANEL_W, PANEL_H = 620, 448
INK, PAPER, ACCENT = "#202c34", "#eef1ee", "#0b6f73"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans"


def font(size, bold=False):
    return ImageFont.truetype(FONT + ("-Bold" if bold else "") + ".ttf", size)


def render(folder, robot):
    metrics = json.loads((folder / "metrics.json").read_text())
    prompt = json.loads((folder / "prompt.json").read_text())
    with np.load(folder / "rollout.npz") as data:
        ref, actual, drawer = data["reference"], data["actual"], data["drawer"]
        touch = data["hand_contact"]
    drawer_origin = metrics.get("drawer_origin")
    models = [scene(robot, None), scene(robot, drawer_origin)]
    sims = [mj.MjData(m) for m in models]
    renderers = [mj.Renderer(m, PANEL_H, PANEL_W) for m in models]
    phases = prompt["prompt_phases"]
    boundaries = np.cumsum([seconds for _, seconds in phases])
    titles = {
        "stand": "Standing",
        "walk": "Walking",
        "run": "Running",
        "squat": "Squat and rise",
        "sequence": "Stand, walk, squat and rise",
        "drawer": "Close the workbench drawer",
    }
    fonts = {k: font(k, k in (32, 22)) for k in (16, 18, 20, 22, 32)}
    total = len(ref) / 50
    name = folder.name
    out = folder / f"{name}.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        "25",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "3",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = (60, -17, 2.8) if drawer_origin else (121, -12, 2.72)
    closeup = None
    if drawer_origin:
        closeup = mj.Renderer(models[1], 148, 208)
        detail_cam = mj.MjvCamera()
        mj.mjv_defaultCamera(detail_cam)
        detail_cam.azimuth, detail_cam.elevation, detail_cam.distance = 43, -25, 0.91
        detail_cam.lookat[:] = np.asarray(drawer_origin) + [0.07, 0.08, 0.01]
    base = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(base)
    dr.text((20, 12), titles.get(name, name), font=fonts[32], fill=INK)
    dr.text((1259, 24), "UNITREE G1   /   SIMULATION", font=fonts[16], fill=ACCENT, anchor="ra")
    dr.text((20, 108), "Kimodo   /   Reference motion", font=fonts[22], fill=INK)
    dr.text((640, 108), "SONIC   /   Physics rollout", font=fonts[22], fill=INK)
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        for t in range(0, len(ref), 2):
            time = t / 50
            phase = min(int(np.searchsorted(boundaries, time, side="right")), len(phases) - 1)
            frame = base.copy()
            draw = ImageDraw.Draw(frame)
            prefix = f"PROMPT   {phase + 1}/{len(phases)}" if len(phases) > 1 else "PROMPT"
            draw.text((21, 72), prefix, font=fonts[16], fill=ACCENT)
            line = phases[phase][0]
            prompt_font = fonts[22] if fonts[22].getlength(line) < W - 182 else fonts[20]
            draw.text((160, 66), line, font=prompt_font, fill=INK)
            for i, states in enumerate((ref, actual)):
                m, d, renderer = models[i], sims[i], renderers[i]
                d.qpos[:] = states[t]
                mj.mj_forward(m, d)
                cam.lookat[:] = [d.qpos[0] + 0.06, d.qpos[1], 0.72]
                if i == 1 and drawer_origin:
                    delta = np.array(drawer_origin[:2]) - d.qpos[:2]
                    cam.lookat[:2] += np.clip(delta, -0.36, 0.36) * 0.6
                renderer.update_scene(d, camera=cam)
                frame.paste(Image.fromarray(renderer.render()), (20 + 620 * i, 147))
            if closeup and np.linalg.norm(actual[t, :2] - drawer_origin[:2]) < 1.3:
                closeup.update_scene(sims[1], camera=detail_cam)
                frame.paste(Image.fromarray(closeup.render()), (1044, 153))
                draw.rectangle((1044, 301, 1251, 325), fill=INK)
                draw.text((1051, 303), "Drawer close-up", font=fonts[16], fill=PAPER)
            draw = ImageDraw.Draw(frame)
            draw.text(
                (20, 607),
                f"Reference travel   {np.linalg.norm(ref[t, :2] - ref[0, :2]):.2f} m",
                font=fonts[18],
                fill=INK,
            )
            draw.text(
                (640, 607),
                f"Robot travel   {np.linalg.norm(actual[t, :2] - actual[0, :2]):.2f} m",
                font=fonts[18],
                fill=INK,
            )
            if drawer_origin:
                draw.text(
                    (1259, 607),
                    f"Drawer   {drawer[t] * 100:.1f} / {TRAVEL * 100:.0f} cm",
                    font=fonts[18],
                    fill=ACCENT,
                    anchor="ra",
                )
            draw.rectangle((20, 657, 1259, 662), fill="#c5d0cc")
            draw.rectangle((20, 657, 20 + int(1239 * t / max(1, len(ref) - 1)), 662), fill=ACCENT)
            for boundary in boundaries[:-1]:
                x = 20 + int(1239 * boundary / total)
                draw.rectangle((x, 653, x + 2, 666), fill="#6d817c")
            status = (
                "Hand contact" if drawer_origin and touch[t] else "50 Hz closed-loop joint control"
            )
            draw.text((20, 680), status, font=fonts[16], fill=ACCENT)
            draw.text(
                (1259, 678),
                f"{time:04.1f} s   /   {total:.1f} s",
                font=fonts[18],
                fill=INK,
                anchor="ra",
            )
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("ffmpeg failed")
    if closeup:
        closeup.close()
    for renderer in renderers:
        renderer.close()
    print(out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--robot", required=True, type=Path)
    args = p.parse_args()
    for folder in args.folders:
        render(folder, args.robot)
