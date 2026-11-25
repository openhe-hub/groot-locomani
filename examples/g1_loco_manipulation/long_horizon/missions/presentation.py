"""Compose the room cameras, retargeted SOMA motion and telemetry from one G1 episode."""

from contextlib import ExitStack
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys


os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kimodo_sonic"))
from soma_view import SomaView


PHASES = ("carry", "secure", "return", "worktop", "escort", "receipt")
LABELS = {
    "home": (
        "Deliver the basket",
        "Close the bread bin",
        "Return to the sink",
        "Wipe the draining board",
        "Return to the pantry",
        "Switch off the task light",
    ),
    "factory": (
        "Load the dispatch cart",
        "Arm the selector",
        "Return to receiving",
        "Read tags and press start",
        "Meet the transfer cart",
        "Confirm arrival at the gate",
    ),
}
# Each mark is tied to the controller event that confirms that physical operation.
TASKS = {
    "home": (
        ("Grip basket", "compliant_grasp"),
        ("Latch cupboard", "cupboard_latched"),
        ("Set basket down", "delivered"),
        ("Close bread bin", "lid_seated"),
        ("Return to sink", "worktop"),
        ("Wipe board", "board_complete"),
        ("Return to pantry", "receipt"),
        ("Switch off light", "pantry_light_off"),
    ),
    "factory": (
        ("Grip work cap", "compliant_grasp"),
        ("Load cart", "delivered"),
        ("Turn selector", "guard_locked"),
        ("Return to console", "worktop"),
        ("Read three tags", "three_tags_read"),
        ("Press start", "dispatch_start_pressed"),
        ("Escort cart", "receipt"),
        ("Confirm dispatch", "dock_receiver_read"),
    ),
}
TITLES = {"home": "Kitchen close-down", "factory": "Workwear dispatch"}
FOLDERS = {"home": "kitchen-close-down", "factory": "workwear-dispatch"}


def probe(path):
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=width,height,avg_frame_rate,nb_read_frames,duration",
                "-of",
                "json",
                str(path),
            ]
        )
    )["streams"][0]
    return int(data["nb_read_frames"]), data["avg_frame_rate"], float(data["duration"])


def pipe(path, size):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            "-i",
            str(path),
            "-vf",
            f"scale={size[0]}:{size[1]}",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "2",
            "-f",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
    )


CANVAS = (1920, 1080)
PAPER, INK, SUB, RULE = "#f4f2ec", "#142a2d", "#4d6263", "#b8c6bf"
ACCENT, WARM = "#007366", "#ac4e23"
CAMERA_SIZE = (896, 672)
DETAIL_SIZE = (448, 336)


@lru_cache
def font(size, style):
    faces = {"normal": "DejaVuSans", "bold": "DejaVuSans-Bold", "mono": "DejaVuSansMono"}
    return ImageFont.truetype(faces[style] + ".ttf", size)


def label(draw, xy, string, size=20, color=INK, style="normal", anchor="lt"):
    draw.text(xy, string, font=font(size, style), fill=color, anchor=anchor)


def canvas(name, kimodo=False):
    image = Image.new("RGB", CANVAS, PAPER)
    d = ImageDraw.Draw(image)
    label(
        d,
        (40, 28),
        "UNITREE G1    /    GR00T + KIMODO ROOT WAYPOINTS + SONIC    /    GPT Astra 6.2 (Orchestrator)"
        if kimodo
        else "UNITREE G1    /    GR00T N1.7 + SONIC    /    GPT Astra 6.2 (Orchestrator)",
        18,
        ACCENT,
        "mono",
    )
    label(d, (37, 60), TITLES[name], 58, INK, "bold")
    label(d, (1880, 31), "CONTINUOUS SIMULATION", 18, SUB, "mono", "rt")
    for x, title in (
        (40, "Room camera"),
        (962, "SOMA  /  motion retarget"),
        (1432, "Interaction camera"),
    ):
        label(d, (x, 150), title, 24, INK, "bold")
    for x in (948, 1417):
        d.line((x, 148, x, 986), fill=RULE)
    d.line((40, 132, 1880, 132), fill=RULE, width=2)
    label(d, (962, 705), "Robot path", 24, INK, "bold")
    label(
        d,
        (1401, 711),
        "○  INPUT WAYPOINTS" if kimodo else "TOP VIEW",
        16,
        WARM if kimodo else SUB,
        "mono",
        "rt",
    )
    label(d, (1432, 589), "Object height", 24, INK, "bold")
    label(d, (1432, 619), "Relative to its starting height", 17, SUB)
    d.line((1432, 796, 1880, 796), fill=RULE)
    label(d, (1432, 823), "Workstation", 24, INK, "bold")
    label(d, (40, 881), "Task list", 24, INK, "bold")
    return image


def task_list(name, events):
    steps = {event["name"]: event["step"] for event in events}
    tags = set()
    for event in events:
        if event["name"] == "tag_read":
            tags.add(event["tag_id"])
            if len(tags) == 3:
                steps["three_tags_read"] = event["step"]
                break
    if "dock_receiver_read" not in steps and "dock_item_read" in steps:
        steps["dock_receiver_read"] = steps["dock_item_read"]
    return [(title, steps.get(event)) for title, event in TASKS[name]]


def draw_tasks(draw, tasks, step):
    done = [end is not None and end <= step for _, end in tasks]
    current = next((i for i, checked in enumerate(done) if not checked), None)
    label(draw, (936, 885), f"{sum(done)} / {len(tasks)} complete", 17, ACCENT, "mono", "rt")
    for i, (title, _) in enumerate(tasks):
        x, y = 40 + (i % 4) * 224, 917 + (i // 4) * 41
        box = (x + 6, y + 6, x + 26, y + 26)
        if done[i]:
            draw.rounded_rectangle(box, radius=4, fill=ACCENT)
            draw.line([(x + 10, y + 16), (x + 15, y + 21), (x + 23, y + 11)], fill=PAPER, width=3)
        elif i == current:
            draw.rounded_rectangle((x, y, x + 217, y + 33), radius=5, fill="#d6e9e2")
            draw.polygon([(x + 11, y + 9), (x + 11, y + 24), (x + 22, y + 16)], fill=ACCENT)
        else:
            draw.rounded_rectangle(box, radius=4, outline=SUB, width=2)
        label(
            draw,
            (x + 34, y + 8),
            title,
            19,
            INK if i == current else SUB,
            "bold" if i == current else "normal",
        )


class PathMap:
    def __init__(self, xy, waypoints=None):
        self.waypoints = waypoints or {}
        bounds = np.concatenate([xy, *self.waypoints.values()])
        centre = (bounds.min(axis=0) + bounds.max(axis=0)) / 2
        scale = min(396 / max(0.1, np.ptp(bounds[:, 0])), 172 / max(0.1, np.ptp(bounds[:, 1])))
        self.points = np.rint((xy - centre) * [scale, -scale] + [1181, 860]).astype(int).tolist()
        self.scale = scale
        self.centre = centre

    def draw(self, d, step, leg=None):
        for x in range(975, 1402, 32):
            for y in range(760, 964, 32):
                d.point((x, y), fill="#9caeaa")
        x, y = self.points[0]
        d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=SUB, width=2)
        points = [tuple(p) for p in self.points[:step:8]] + [tuple(self.points[step])]
        if len(points) > 1:
            d.line(points, fill=ACCENT, width=4)
        for point in self.waypoints.get(leg, []):
            u, v = (point - self.centre) * [self.scale, -self.scale] + [1181, 860]
            d.ellipse((u - 5, v - 5, u + 5, v + 5), outline=WARM, width=3)
        x, y = points[-1]
        d.ellipse((x - 7, y - 7, x + 7, y + 7), fill=INK, outline=PAPER, width=2)
        d.line((968, 979, 968 + 0.5 * self.scale, 979), fill=SUB, width=2)
        label(d, (976 + 0.5 * self.scale, 971), "0.5 m", 15, SUB, "mono")


def waveform(draw, height_cm, step, end, limits):
    lo, hi = limits
    x0, y0, width, depth = 1434, 664, 443, 94

    def y(v):
        return int(y0 + (hi - v) / (hi - lo) * depth)

    for value in (lo, 0, hi):
        draw.line((x0, y(value), x0 + width, y(value)), fill=RULE)
    indices = np.unique(np.r_[np.linspace(0, end, width, dtype=int), step])
    points = [(x0 + int(width * i / end), y(height_cm[i])) for i in indices if i <= step]
    if len(points) > 1:
        draw.line(points, fill=WARM, width=3)
    x, yy = points[-1]
    draw.ellipse((x - 4, yy - 4, x + 4, yy + 4), fill=WARM)
    label(draw, (1432, 766), "0 s", 15, SUB, "mono")
    label(draw, (1880, 766), f"{end / 50:.1f} s", 15, SUB, "mono", "rt")


def stop_process(proc):
    if proc.poll() is None:
        proc.kill()
    proc.wait()
    for stream in (proc.stdin, proc.stdout):
        if stream and not stream.closed:
            stream.close()


def read_image(proc, size):
    length = size[0] * size[1] * 3
    raw = proc.stdout.read(length)
    if len(raw) != length:
        raise RuntimeError("Incomplete source camera frame")
    return Image.frombytes("RGB", size, raw)


def movie(run, dest, name, trace, metrics, events, row, robot, skin, encoder):
    replay = run / "replay/capture.json"
    directory = (
        replay.parent if replay.exists() else run / "sim/sim_eval/videos" / row["scenario_id"]
    )
    sources = [directory / f"{view}.mp4" for view in ("third_person", "ego")]
    count, fps, duration = probe(sources[0])
    end = metrics["control_steps"]
    if probe(sources[1]) != (count, fps, duration) or fps != "25/1" or count != (end + 1) // 2:
        raise ValueError("Camera timing disagrees with the 50 Hz trajectory")
    if replay.exists() and json.loads(replay.read_text())["indices"] != "1,3,...":
        raise ValueError("Unrecognized Isaac replay sampling")

    g1 = mj.MjModel.from_xml_path(str(robot))
    state = mj.MjData(g1)
    names = trace["joint_names"].tolist()
    joint_ids = [names.index(g1.joint(i).name) for i in range(1, g1.njnt)]
    human = SomaView(skin)
    human.m.skin_rgba[0] = [0.22, 0.65, 0.61, 1.0]
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.azimuth, camera.elevation, camera.distance = 40, -10, 2.60
    marker = {e["name"]: e["step"] for e in events}
    tasks = task_list(name, events)
    starts = [0] + [marker[k] for k in PHASES[1:] if k in marker]
    xy = trace["root"][:, :2]
    metres = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    waypoint_map = {}
    if "kimodo_walk" in marker:
        launch = json.loads((run / "launch.json").read_text())
        with np.load(Path(launch["station_assets"]) / "pelvis-routes.npz") as refs:
            waypoint_map = {
                k: refs[k + "_points"][:, 1:] for k in ("loaded", "return", "escort") if k in refs
            }
    route = PathMap(xy, waypoint_map)
    height = 100 * (trace["object"][:, 2] - trace["object"][0, 2])
    limits = (min(-5, 5 * np.floor(height.min() / 5)), max(5, 5 * np.ceil(height.max() / 5)))
    home = name == "home"
    reads = [e["step"] for e in events if e["name"] == ("wipe_cell" if home else "tag_read")]
    queries = [q["step"] for q in json.loads((run / "queries.json").read_text())]
    bg = canvas(name, kimodo=marker.get("kimodo_walk") is not None)
    if not metrics["success"]:
        d = ImageDraw.Draw(bg)
        d.rectangle((1390, 22, 1885, 57), fill=PAPER)
        reason = metrics["terminal_reason"].replace("_", " ").upper()
        label(d, (1880, 31), f"INCOMPLETE RUN  /  {reason}", 18, WARM, "mono", "rt")
    output = dest / (dest.name + ".mp4")
    settings = (
        ["-gpu", "0", "-rc", "vbr", "-cq", "18", "-b:v", "0"]
        if encoder == "h264_nvenc"
        else ["-preset", "fast", "-crf", "18", "-threads", "4"]
    )
    args = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        "25",
        "-i",
        "-",
        "-c:v",
        encoder,
        *settings,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    with ExitStack() as pool:
        stage = pool.enter_context(mj.Renderer(human.m, 490, 440))
        streams = [pipe(src, size) for src, size in zip(sources, (CAMERA_SIZE, DETAIL_SIZE))]
        for stream in streams:
            pool.callback(stop_process, stream)
        ff = subprocess.Popen(args, stdin=subprocess.PIPE)
        pool.callback(stop_process, ff)
        for frame in range(count):
            step = 2 * frame + 1
            state.qpos[:7] = trace["root"][step, :7]
            state.qpos[7:] = trace["joints"][step, joint_ids]
            mj.mj_kinematics(g1, state)
            human.update(g1, state)
            camera.lookat[:] = [*human.d.mocap_pos[0, :2], 0.92]
            stage.update_scene(human.d, camera=camera)
            image = bg.copy()
            image.paste(read_image(streams[0], CAMERA_SIZE), (40, 192))
            image.paste(Image.fromarray(stage.render()), (962, 192))
            image.paste(read_image(streams[1], DETAIL_SIZE), (1432, 192))
            d = ImageDraw.Draw(image)
            label(d, (1880, 69), f"{step / 50:05.2f}  /  {end / 50:.2f} s", 34, INK, "mono", "rt")
            phase = int(np.searchsorted(starts, step, side="right") - 1)
            draw_tasks(d, tasks, step)
            detail = replay.exists() and trace["phase"][step] in ("secure", "worktop", "receipt")
            label(d, (1432, 544), "Station close-up" if detail else "Head-mounted view", 18, SUB)
            leg = (
                "loaded"
                if phase == 0 and step >= marker.get("loaded_route", end + 1)
                else PHASES[phase]
            )
            route.draw(d, step, leg)
            label(d, (1401, 971), f"{metres[step]:.2f} m walked", 16, ACCENT, "mono", "rt")
            label(d, (1880, 594), f"{height[step]:+.1f} cm", 26, WARM, "mono", "rt")
            waveform(d, height, step, end, limits)
            qty = int(np.searchsorted(reads, step, side="right"))
            readouts = [
                ("Board zones wiped" if home else "Tags read", f"{qty} / {15 if home else 3}"),
                (
                    "Bread-bin travel" if home else "Cart travel",
                    f"{abs(trace['secure'][step]) * 100:.1f} cm"
                    if home
                    else f"{np.linalg.norm(trace['receiver'][step, :2] - trace['receiver'][0, :2]) * 100:.1f} cm",
                ),
                ("VLA queries", str(np.searchsorted(queries, step, side="right"))),
            ]
            for j, (key, v) in enumerate(readouts):
                y = 870 + j * 39
                label(d, (1432, y), key, 21, SUB)
                label(d, (1880, y), v, 23, INK, "mono", "rt")
            for j, (a, b) in enumerate(zip(starts, starts[1:] + [end])):
                left, right = 40 + 1840 * a / end, 40 + 1840 * b / end - 3
                d.rectangle((left, 1014, right, 1024), fill=RULE)
                if step >= a:
                    d.rectangle(
                        (left, 1014, max(left, min(right, 40 + 1840 * step / end)), 1024),
                        fill=ACCENT,
                    )
            label(
                d,
                (40, 1037),
                f"Phase {phase + 1} of {len(PHASES)}   /   {LABELS[name][phase]}",
                16,
                SUB,
                "mono",
            )
            label(d, (1880, 1037), "50 Hz state   /   25 fps replay", 16, SUB, "mono", "rt")
            if frame == max(1, marker.get("delivered", end // 2) // 2 - 90):
                image.save(dest / "preview.jpg", quality=94)
            ff.stdin.write(image.tobytes())
        ff.stdin.close()
        if ff.wait() or any(s.wait() for s in streams):
            raise RuntimeError("Video encoding failed")
    if probe(output) != (count, fps, duration):
        raise ValueError("Encoded video timing mismatch")
