import io
import os
from pathlib import Path

from gr00t.policy.server_client import PolicyClient
from isaaclab_arena_gr00t.policy.config.gr00t_closedloop_policy_config import (
    Gr00tClosedloopPolicyConfig,
    TaskMode,
)
from isaaclab_arena_gr00t.policy.gr00t_core import (
    build_gr00t_action_np,
    build_gr00t_policy_observations,
    load_gr00t_joint_configs,
)
from isaaclab_arena_gr00t.utils.io_utils import load_gr00t_modality_config_from_file
import numpy as np
import yaml
import zmq


root = Path(__file__).resolve().parent
os.chdir(root / "arena-remote")
raw = yaml.safe_load(
    Path("isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml").read_text()
)
raw["model_path"] = str(root / "checkpoint-box")
for key in (
    "state_joints_config_path",
    "action_joints_config_path",
    "policy_joints_config_path",
    "modality_config_path",
):
    raw[key] = Path(raw[key])
cfg = Gr00tClosedloopPolicyConfig(**raw)
pj, aj, sj = load_gr00t_joint_configs(cfg)
modalities = load_gr00t_modality_config_from_file(cfg.modality_config_path, cfg.embodiment_tag)
client = PolicyClient(host="127.0.0.1", port=16556, strict=False)
assert client.ping()
sock = zmq.Context().socket(zmq.REP)
sock.bind("ipc://" + str(root / "box.sock"))
print("bridge ready", flush=True)
while True:
    data = sock.recv()
    if data == b"reset":
        client.reset()
        sock.send(b"ok")
        continue
    with np.load(io.BytesIO(data), allow_pickle=False) as obs:
        model_input = build_gr00t_policy_observations(
            [obs["rgb"]], obs["joints"], cfg.language_instruction, cfg, sj, pj, modalities
        )
    prediction, _ = client.get_action(model_input)
    action = build_gr00t_action_np(
        prediction, TaskMode(cfg.task_mode_name), pj, aj, cfg.embodiment_tag
    )
    buffer = io.BytesIO()
    np.save(buffer, action, allow_pickle=False)
    sock.send(buffer.getvalue())
    print("returned", action.shape, flush=True)
