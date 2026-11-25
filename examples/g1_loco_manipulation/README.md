# Unitree G1 全身移动与操作的闭环仿真

2026-11-20 实际运行了两条策略。两段视频都来自策略读取当前仿真观测、预测动作并由全身控制器执行的闭环。下表记录的是交付视频对应的单次结果。

| 组合 | 任务 | 本次记录 |
|---|---|---|
| GR00T N1.6 + Decoupled WBC | 从货架双手取下纸箱，转身搬到蓝筐 | seed 42，945 步，50 Hz，18.90 秒。G1 水平净位移约 1.20 米。Arena 的物体与目标筐接近度判定成功。 |
| GR00T N1.7 + SONIC | 走到桌旁，右手抓香蕉并放入篮子 | 固定场景 `task1_00009`，572 步，50 Hz，11.44 秒。香蕉最大抬升 12.19 厘米，末帧位置、抬升和速度检查均通过。 |

## 运行环境与录像位置

| 任务 | 主机 | 远端工作根目录 | 本次分工 |
|---|---|---|---|
| N1.6 + Decoupled WBC 搬纸箱 | L40S 主机 | `$WORK_ROOT` | 2 张 L40S。策略服务用物理 GPU 1，Isaac Sim 容器使用 GPU 渲染，物理仿真选择 CPU。 |
| N1.7 + SONIC 放香蕉 | A6000 主机 | `$WORK_ROOT` | 1 张 RTX A6000。策略推理和仿真渲染共用 GPU 0。 |

两台主机各有一份工作根目录，下文统一写作 `$WORK_ROOT`，以下路径均相对于它。搬纸箱用的 Isaac Sim 容器写作 `$SIM_CONTAINER`。

| 文件或服务 | L40S 主机 · 搬纸箱 | A6000 主机 · 放香蕉 |
|---|---|---|
| 仿真源码 | `arena/` | `zero/` |
| VLA 源码 | `arena-remote/submodules/Isaac-GR00T/` | `zero/integrations/groot/` |
| 任务权重 | `checkpoint-box/` | `zero/checkpoints/task1/` |
| 推理入口 | `policy16/bin/python`，回环端口 `16556` | `policy/bin/python`，回环端口 `16557` |
| 仿真入口 | 容器 `$SIM_CONTAINER`，工作根目录挂载到 `/demo` | `sim/bin/python` |
| 已验证结果 | `box-runs-v2/box-0.json`、`box-runs-v2/box-0.mp4` | `zero/outputs/banana09/sim/sim_eval/` 下的 `episodes.csv` 和 `videos/task1_00009/` |

## 本地源码版本

| 目录 | 与本次运行的关系 |
|---|---|
| [Isaac-GR00T](../../README.md) | NVIDIA 官方主仓库。本机整理时的提交为 `708617f5716c`。 |
| [third_party/IsaacLab-Arena](../../third_party/IsaacLab-Arena/) | 保留 N1.6 策略适配器版本 `8b4a3a47fc53`（release/0.2.1）。远端仿真本体另外使用 `f479817431f6`（release/0.1.1）。N1.6 策略服务使用官方 GR00T 提交 `e29d8fc50b0e`。 |
| [box](box/) | 从 L40S 主机上成功运行的工作根目录取回策略桥接和仿真录像脚本。 |
| [third_party/Zero-HLM-Bench](../../third_party/Zero-HLM-Bench/) | 香蕉任务使用提交 `b481409f2cd6`，其中 [N1.7 策略实现](../../third_party/Zero-HLM-Bench/integrations/groot/) 和仿真评测器来自同一版本。 |

两个子项目由 Git 子模块固定版本。本地工作树使用 sparse checkout，仅展开开发所需的源码、配置与文档。

## 纸箱 · GR00T N1.6 + Decoupled WBC

策略权重为 [NVIDIA 的 G1 搬箱 checkpoint](https://huggingface.co/nvidia/GN1x-Tuned-Arena-G1-Loco-Manipulation/tree/gn1_6) 的 `gn1_6` 修订版。策略读取头部相机 RGB 和关节位置，每次预测 50 步动作。Decoupled WBC 执行上身关节目标和下身移动指令。[Arena 闭环评测说明](https://isaac-sim.github.io/IsaacLab-Arena/release/0.2.1/pages/example_workflows/locomanipulation/step_5_evaluation.html)

以下命令复用 L40S 主机上保留的环境和容器。在同一台机器开三个终端，先分别进入工作根目录：

```bash
cd "$WORK_ROOT"
```

**终端一 · N1.6 策略服务**

```bash
root=$PWD
cd arena-remote/submodules/Isaac-GR00T
HF_HOME="$root/hf" CUDA_VISIBLE_DEVICES=1 "$root/policy16/bin/python" -u gr00t/eval/run_gr00t_server.py \
  --modality-config-path ../../isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py \
  --model-path "$root/checkpoint-box" --embodiment-tag NEW_EMBODIMENT \
  --host 127.0.0.1 --port 16556 --device cuda
```

**终端二 · 观测和动作转换**

```bash
PYTHONPATH="$PWD/arena-remote:$PWD/arena-remote/submodules/Isaac-GR00T" \
  policy16/bin/python -u box_bridge.py
```

桥接器输出 `bridge ready` 后会在工作根目录创建 `box.sock`。容器通过同一挂载目录读写这个 Unix socket。

**终端三 · 仿真与录像**

```bash
docker start "$SIM_CONTAINER"
run_dir="box-rerun-$(date +%Y%m%d-%H%M%S)"
docker exec -e PYTHONPATH=/demo/arena -e RUN_DIR="$run_dir" "$SIM_CONTAINER" bash -o pipefail -c '
  cd /demo/arena &&
  sed "s@/demo/box-runs-v2@/demo/$RUN_DIR@g" /demo/box_run.py |
  /isaac-sim/python.sh /dev/stdin --headless --enable_cameras --device cpu \
    --kit_args="--/persistent/isaac/asset_root/cloud=/demo/s3" \
    --policy_type zero_action galileo_g1_locomanip_pick_and_place \
    --object brown_box --embodiment g1_wbc_joint'
```

这里的 `--policy_type zero_action` 用于通过旧版 Arena 的参数解析。实际传给 `env.step` 的动作来自 [box_run.py](box/box_run.py) 经 socket 获取的 N1.6 预测。上面的命令只替换录像目录，以保留原始结果。用 `box-*.json` 的 `terminated`、`truncated` 和 `metrics` 核对每一段。原始成功记录在 `box-runs-v2/`，较早的 `box-runs/` 包含一次失败尝试。

## 香蕉 · GR00T N1.7 + SONIC

使用 [Zero-HLM 的 Task 1 专用 checkpoint](https://huggingface.co/YDY0427/Zero-HLM-Bench-GR00T)，对应 `unitree_g1_sonic`。该版本的动作输出为 64 维 SONIC 运动表示和左右手各一个抓取指令。每次从 40 步预测中执行前 16 步，再读取新观测。

以下两个终端均在 A6000 主机上执行，先进入：

```bash
cd "$WORK_ROOT/zero"
```

**终端一 · N1.7 策略服务**

```bash
HF_HOME="$PWD/../hf" ../policy/bin/python scripts/start_server.py \
  --checkpoint checkpoints/task1 --host 127.0.0.1 --port 16557
```

**终端二 · 重跑同一场景**

```bash
run_dir="outputs/banana09-$(date +%Y%m%d-%H%M%S)"
OMNI_KIT_ACCEPT_EULA=YES HF_HOME="$PWD/../hf" ../sim/bin/python scripts/start_client.py \
  --task task1 --row-start 9 --row-count 1 --host 127.0.0.1 --port 16557 --output "$run_dir"
```

在 `$run_dir/sim/sim_eval/episodes.csv` 查看 `terminal_reason`、`max_object_lift` 和 `video_frames`。同目录下的 `videos/task1_00009/third_person.mp4` 与 `ego.mp4` 是同步录像，交付版将两者左右并排。原始运行目录为 `outputs/banana09/`。

## 在 A6000 主机上生成 Kimodo G1 参考动作

本地源码在 [third_party/kimodo](../../third_party/kimodo/)。A6000 主机工作根目录下的同名目录使用相同提交 `e1c219c99b4b`。Kimodo 使用独立环境 `envs/kimodo/`，模型缓存入口为 `models/hub/`。两台机器的安装包在各自的 `install/`，运行和安装日志在 `logs/`。

登录 A6000 主机后运行以下命令，生成一段三秒的 G1 走路参考动作。每次生成使用独立的输出目录。

```bash
cd "$WORK_ROOT"
source envs/kimodo/bin/activate
export HF_HUB_CACHE="$PWD/models/hub" HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=local
mkdir -p outputs
run_dir=$(mktemp -d "$PWD/outputs/kimodo-walk-XXXXXXXX")
CUDA_VISIBLE_DEVICES=0 kimodo_gen "A person walks forward." \
  --model Kimodo-G1-RP-v1 --duration 3 --seed 42 --output "$run_dir/walk"
```

### Kimodo + SONIC 三栏视频

[交付索引](../../outputs/kimodo/README.md)列出基础动作、动作衔接和厨房操作三部合集。SOMA 与 G1 的参考取自同一控制采样时刻。重定向实现和三栏渲染入口分别是 [soma_view.py](kimodo_sonic/soma_view.py) 和 [comparison.py](kimodo_sonic/comparison.py)。

在 A6000 主机的 Kimodo 环境中，从项目工作根目录复跑。以下命令输出到新目录。厨房另依赖工作根目录中的 D4RL 提交 `89141a689b03` 的 `d4rl/kitchen/adept_models/`，以及相邻 `../robosuite` 提交 `51cc01785bab` 的食品网格和纹理。

```bash
src=examples/g1_loco_manipulation/kimodo_sonic
robot=../GR00T-WholeBodyControl/gear_sonic_deploy/g1/g1_29dof.xml
out=$(mktemp -d "$PWD/outputs/kimodo-three-panel-XXXXXXXX")
export HF_HUB_CACHE="$PWD/models/hub" HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=local MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
motions=(stand walk backward sidestep turn march run squat reach wave clap)
python "$src/generate.py" "$out/motions" "${motions[@]}"
for seq in sequence greeting mobility; do
    python "$src/stitch.py" "$out/motions" --sequence "$seq"
done
for task in "${motions[@]}" sequence greeting mobility; do
    python "$src/track.py" "$out/motions/$task" --zero zero --robot "$robot"
    python "$src/comparison.py" "$out/motions/$task" --robot "$robot"
done
for task in soap microwave groceries; do
    PYTHONPATH="$src" python -c 'import sys; from pathlib import Path; from kitchen_task import prepare; prepare(Path(sys.argv[1]),sys.argv[2])' "$out" "$task"
    case "$task" in soap) offset=.39 ;; microwave) offset=.64 ;; groceries) offset=.57 ;; esac
    python "$src/track.py" "$out/$task" --zero zero --robot "$robot" --kitchen "$task" --start-offset "$offset"
    python "$src/comparison.py" "$out/$task" --robot "$robot"
done
```


## 长时移动操作

厨房餐后收尾和工帽配发出库的三栏录像见[成片与实测结果](../../outputs/long-horizon/README.md)。厨房主流程的模块与数据接口见[长程控制架构](../../docs/long-horizon.md)，环境和命令见[复跑说明](long_horizon/README.md)。

[骨盆路点实验](long_horizon/README.md#骨盆路点版厨房)包含按泵任务的 8 个配对种子，以及同一厨房收尾工序的 Kimodo + SONIC 路点版本。
