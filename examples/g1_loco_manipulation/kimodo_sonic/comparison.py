"""Render synchronized human, G1 reference, and physical execution in three columns."""

import argparse
import json
import os
from pathlib import Path
import subprocess


os.environ.setdefault("MUJOCO_GL", "egl")
from kitchen_scene import make_kitchen
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from soma_view import SomaView
from track import scene


TITLES = {
    "soap": "Press the soap dispenser twice",
    "stand": "Stand at ease",
    "walk": "Forward walk",
    "run": "Jog forward",
    "squat": "Squat and rise",
    "wave": "Wave hello",
    "sidestep": "Side steps",
    "backward": "Walk backwards",
    "reach": "Overhead reach",
    "turn": "Turn in place",
    "march": "March in place",
    "clap": "Clap hands",
    "sequence": "Walk, squat and recover",
    "greeting": "Approach, greet and turn",
    "mobility": "Side step, reach and retreat",
    "microwave": "Close the microwave",
    "groceries": "Put the cereal on the receiving mat",
    "banana": "Pick up the banana and place it in the basket",
    "drawer": "Close the workbench drawer",
}
FONTS = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
BG = "#f1f2ec"
FG = "#1e322f"
TEAL = "#216e68"
W, H = 1600, 720
WIDTHS = (382, 382, 800)
XS = (12, 404, 796)
PH = 492


def camera():
    c = mj.MjvCamera()
    mj.mjv_defaultCamera(c)
    return c


def render(folder, root, robot):
    p = json.loads((folder / "prompt.json").read_text())
    m = json.loads((folder / "metrics.json").read_text())
    data = np.load(folder / "rollout.npz")
    refs, actual = data["reference"], data["actual"]
    home = scene(robot, None)
    ref = mj.MjData(home)
    isaac = folder.name == "banana"
    feeds = []
    if isaac:
        for source, w, h in [
            (folder / "third_person.mp4", 800, 492),
            (folder / "ego.mp4", 204, 144),
        ]:
            vf = f"select=not(mod(n\\,2)),scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=0x36403c"
            feeds.append(
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-threads",
                        "2",
                        "-i",
                        str(source),
                        "-vf",
                        vf,
                        "-vsync",
                        "0",
                        "-threads",
                        "2",
                        "-pix_fmt",
                        "rgb24",
                        "-f",
                        "rawvideo",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                )
            )
    work = folder.name in ("microwave", "groceries", "soap")
    live = (
        make_kitchen(robot, root, soap=folder.name == "soap")
        if work
        else scene(robot, m.get("drawer_origin"))
    )
    state = mj.MjData(live)
    human = SomaView(
        root / "third_party/kimodo/kimodo/assets/skeletons/somaskel77/skin_standard.npz"
    )
    renderers = [
        mj.Renderer(model, PH, width) for model, width in zip((human.m, home, live), WIDTHS)
    ]
    cams = [camera() for _ in range(3)]
    detail = mj.Renderer(live, 145, 234) if work else None
    lens = camera()
    fonts = {
        n: ImageFont.truetype(FONTS + ("-Bold" if n in (33, 23) else "") + ".ttf", n)
        for n in (16, 19, 21, 23, 33)
    }
    base = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(base)
    draw.text((17, 9), TITLES.get(folder.name, folder.name), font=fonts[33], fill=FG)
    draw.text(
        (1584, 20), "KIMODO  /  SONIC     •     UNITREE G1", anchor="ra", font=fonts[16], fill=TEAL
    )
    for x, label in zip(
        XS, ("SOMA  /  Human retarget", "G1  /  Reference", "SONIC  /  Physics execution")
    ):
        draw.text((x + 4, 105), label, font=fonts[23], fill=FG)
    phases = p["prompt_phases"]
    boundaries = np.cumsum([seconds for _, seconds in phases])
    out = folder / (folder.name + ".mp4")
    tmp = out.with_suffix(".tmp.mp4")
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
        str(tmp),
    ]
    option = mj.MjvOption()
    mj.mjv_defaultOption(option)
    option.geomgroup[4:] = 0
    option.sitegroup[:] = 0
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as ff:
        for t in range(0, len(refs), 2):
            sec = t / 50
            phase = min(len(phases) - 1, np.searchsorted(boundaries, sec, side="right"))
            ref.qpos[:] = refs[t, :36]
            state.qpos[:] = actual[t]
            mj.mj_forward(home, ref)
            mj.mj_forward(live, state)
            human.update(home, ref)
            im = base.copy()
            dr = ImageDraw.Draw(im)
            dr.text((17, 66), "TEXT PROMPT", font=fonts[16], fill=TEAL)
            dr.text((155, 60), phases[phase][0], font=fonts[23], fill=FG)
            for j, (rr, cam, d) in enumerate(zip(renderers, cams, (human.d, ref, state))):
                pos = human.d.mocap_pos[0] if j == 0 else d.qpos[:3]
                cam.azimuth, cam.elevation, cam.distance = 112, -9, (3.35 if j == 0 else 2.65)
                cam.lookat[:] = [pos[0], pos[1], 0.90 if j == 0 else 0.73]
                if j == 2 and work:
                    cam.azimuth, cam.elevation, cam.distance = 31, -15, 3.45
                    cam.lookat[:] = [0.6 * pos[0] + 0.4 * 2.4, 0.62 * pos[1] + 0.38 * 0.3, 0.95]
                if isaac and j == 2:
                    buf = feeds[0].stdout.read(WIDTHS[j] * PH * 3)
                    im.paste(Image.frombytes("RGB", (WIDTHS[j], PH), buf), (XS[j], 139))
                    buf = feeds[1].stdout.read(204 * 144 * 3)
                    im.paste(Image.frombytes("RGB", (204, 144), buf), (1382, 145))
                    dr.rectangle((1382, 289, 1585, 312), fill=FG)
                    dr.text((1388, 290), "Head camera", font=fonts[16], fill=BG)
                    continue
                rr.update_scene(d, camera=cam, scene_option=option)
                im.paste(Image.fromarray(rr.render()), (XS[j], 139))
            if work and sec > p["approach_seconds"] - 0.6:
                if folder.name == "soap":
                    target = state.xpos[live.body("soap_dispenser").id] + np.array([0, 0, 0.25])
                elif folder.name == "microwave":
                    target = state.site("microhandle_site").xpos
                else:
                    target = state.xpos[live.body("carton").id]
                lens.lookat[:] = target
                lens.azimuth = 26
                lens.elevation = -24
                lens.distance = 0.48 if folder.name == "soap" else 0.88
                detail.update_scene(state, camera=lens, scene_option=option)
                im.paste(Image.fromarray(detail.render()), (1357, 145))
                dr.rectangle((1357, 290, 1590, 312), fill=FG)
                dr.text((1364, 290), "Interaction close-up", font=fonts[16], fill=BG)
            dr.text((16, 646), "SOMA 77   •   same reference frame", font=fonts[16], fill=TEAL)
            travel = np.linalg.norm(actual[t, :2] - actual[0, :2])
            dr.text((800, 644), f"Robot travel   {travel:.2f} m", font=fonts[19], fill=FG)
            if isaac:
                dr.rectangle((400, 98, 788, 137), fill=BG)
                dr.text(
                    (408, 105),
                    "G1  /  Reference" if phase == 0 else "G1  /  Joint command",
                    font=fonts[23],
                    fill=FG,
                )
                dr.text(
                    (1586, 644),
                    "Kimodo + SONIC" if phase == 0 else "GR00T N1.7 + SONIC",
                    font=fonts[19],
                    fill=TEAL,
                    anchor="ra",
                )
            if work:
                if folder.name == "soap":
                    count = np.searchsorted(m["return_frames"], t, side="right")
                    dr.text((1060, 644), f"Presses   {count} / 2", font=fonts[19], fill=TEAL)
                unit = {"microwave": "°", "groceries": "cm", "soap": " mm"}[folder.name]
                dr.text(
                    (1585, 644),
                    f"{p['readout']}   {data['progress'][t]:.1f}{unit}",
                    font=fonts[19],
                    fill=FG,
                    anchor="ra",
                )
            dr.rectangle((16, 685, 1583, 690), fill="#c7d0c8")
            dr.rectangle((16, 685, 16 + int(1567 * t / max(1, len(refs) - 1)), 690), fill=TEAL)
            for b in boundaries[:-1]:
                x = 16 + int(1567 * b / (len(refs) / 50))
                dr.line((x, 681, x, 695), fill=FG, width=2)
            ff.stdin.write(im.tobytes())
        ff.stdin.close()
        if ff.wait():
            raise RuntimeError("ffmpeg failed")
    for rr in renderers:
        rr.close()
    if detail:
        detail.close()
    for feed in feeds:
        feed.stdout.close()
        if feed.wait():
            raise RuntimeError("Failed to decode source camera")
    tmp.replace(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folders", nargs="+", type=Path)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument(
        "--robot",
        type=Path,
        default=Path("../GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml"),
    )
    a = p.parse_args()
    for f in a.folders:
        render(f, a.root, a.robot)
        print(f, flush=True)
