# G1 长时移动操作

[家庭与工厂成片](../../../outputs/long-horizon/README.md)

厨房正式交付使用固定路径 `outputs/long-horizon/kitchen-close-down/kitchen-close-down.mp4`。更新时在临时目录完成打包和验收，然后覆盖主视频、预览和对应的 `evidence/`，并同步时长和工序时间表。仓库只提交 `outputs/` 下的说明和证据文件，视频作为 [media-v1 Release](https://github.com/openhe-hub/groot-locomani/releases/tag/media-v1) 的附件发布，更新视频后需重新上传同名附件。

## 从原始回合重建三栏成片

在 L40S 主机物理 GPU 1 的长程任务容器中，从 `/work` 运行。下文用 `$MISSION_CONTAINER` 表示该容器，`$WORK_ROOT` 表示远端工作根目录。打包器直接读取原始相机流、机器人关节、载物轨迹和工序事件。它将 G1 的 29 个身体关节按名称映射到 MuJoCo 模型，再用正向运动学驱动 SOMA 人形。SOMA 是执行姿态的可视化。原始策略使用机器人头部图像与关节状态。

```bash
src=examples/g1_loco_manipulation/long_horizon
home_run=experiments/pelvis-input
factory_run=experiments/mixed-missions
out=$(mktemp -d "$PWD/review-XXXXXXXX")
station-env/bin/python "$src/missions/package.py" "$home_run/kitchen-08" \
    --checkpoint "$home_run/checkpoint.json" --output "$out" --encoder h264_nvenc
station-env/bin/python "$src/missions/package.py" "$factory_run/workwear-gate" \
    --checkpoint "$factory_run/factory-server-k.json" --output "$out" --encoder h264_nvenc
```

两路相机和 SOMA 共用控制步索引 `1, 3, 5, …`。路线按骨盆水平位置累计，物体高度取世界坐标 Z 与起始值之差，擦拭和读签计数取已发生的事件。渲染器复用 `kimodo_sonic/soma_view.py` 和 Kimodo 子模块的 `skin_standard.npz`。远端工作目录为 `$WORK_ROOT/long-horizon-work`。

## 多工序家庭与工厂任务

入口仍为 `record_task.py`，任务适配器与房间扩展在 `missions/`。厨房使用 task3 的篮子与柜门模型。工厂使用 task2 的工帽，策略头部相机看到的是带货架、料箱、工具台和接收车的完整车间。新工单从一次环境复位开始，G1、搬运物和操作机构的状态跨阶段保留。

GR00T 在线读取头部 RGB、机器人观测与指令，预测取物和起运动作。检测到合手指令与手指接触后，轻载物体使用有限力的弹簧阻尼抓持辅助，反力施加在腕部。厨房由 Kimodo 生成载篮和两段空手步态，载篮时另用已记录的腰臂姿态。工厂由 SONIC 跟踪此前 GR00T 产生的载物步态，回程参考来自实际去程。放手时机依据物体和接收面的相对位置，停靠偏差由额外步伐校正。手部操作使用场景状态、G1 运动学和关节驱动。

 面包盒滑盖和放行手柄采用手指进入操作区后、由实测手位驱动的 PhysX 关节，验收读取实际行程或角度。沥水板按抹布所在的板面接触区清除湿痕。料签采用短距感应模型，启动键按指尖到达按钮的状态触发。转运车为动力学刚体，由限力电机沿直线关节运行；工帽通过与托盘的接触随车移动。厨房柜门沿用原任务的连杆进入触发区后闭合机制。

复跑时， 在运行目录启动 `docker start "$MISSION_CONTAINER"`，用 `docker exec -it "$MISSION_CONTAINER" bash` 进入容器的 `/work`。下表的 `mission`、`task` 和权重逐项对应。将 `assets` 指向所选任务的工作站资产；交付回合的精确路径已保存在 `launch.json` 中。

| mission | task | 载货参考 | 主要操作 |
|---|---|---|---|
| `home` | `task3` | `pelvis-routes.npz` | 送餐篮、关柜、合盖、擦板、关灯 |
| `factory` | `task2` | `factory-route.npz` | 工帽装车、转动选择器、读料签、发车、到站核验 |

```bash
mission=home  # 工厂改为 factory
port=16753
src=examples/g1_loco_manipulation/long_horizon
receipt=()
case "$mission" in
    home)    task=task3; assets=experiments/pelvis-input/room-reach ;;
    factory) task=task2; assets=experiments/mixed-missions/dock-room; receipt=(--gate-receipt) ;;
esac
run=$(mktemp -d "$PWD/mixed-repro-XXXXXXXX")
checkpoint_record="$run/checkpoint.json"
# 启动服务后等待就绪。
PYTHONUNBUFFERED=1 policy/bin/python zero/scripts/start_server.py --checkpoint "zero/checkpoints/$task" \
  --host 127.0.0.1 --port "$port" --metadata-output "$checkpoint_record" >"$run/policy.log" 2>&1 &
server_pid=$!
until grep -q 'Server is ready' "$run/policy.log"; do
    kill -0 "$server_pid" 2>/dev/null || { cat "$run/policy.log"; exit 1; }
    sleep 2
done
sim/bin/python "$src/record_task.py" --zero zero --mission "$mission" --task "$task" \
  --station-assets "$assets" --state-only "${receipt[@]}" --row-start 9 --port "$port" --output "$run/episode"
kill "$server_pid"
sim/bin/python "$src/missions/replay.py" "$run/episode" --zero zero
station-env/bin/python "$src/missions/package.py" "$run/episode" \
  --checkpoint "$checkpoint_record" --output "$run/review" --encoder h264_nvenc
```

`--state-only` 保留实时策略相机以及 50 Hz 原始状态。录像用这些状态在原房间以 25 fps 重渲染。全景展示动线，另一路在搬运时展示头部视角，在操作时展示机构近景。`queries.json` 记录策略每次查询的控制步和输入图像摘要。`events.json` 给出各工序边界，`rollout.npz` 含机器人关节、物体与承载面状态，`station-trace.npz` 含指尖、目标和机构读数。

工厂交付版的客户端额外使用 `--gate-receipt`。出库读头位于导轨终点，连续 40 个控制步同时满足实际读码距离、车速、工帽承重及 G1 在场条件后完成交接。家庭交付版在关灯后复查篮子与面包盒。厨房的交付记录位于 `experiments/pelvis-input/kitchen-08`，工厂位于 `experiments/mixed-missions/workwear-gate`。这两条均从第一帧连续记录至实测终点。

工厂原始运行在成功读到终点编号后写汇总 JSON 时遇到了 NumPy 布尔值序列化问题。`missions/verify_dock.py` 从在线读码事件和末段原始状态独立重算回执，生成 `verified-terminal.json`。打包器同时核对被搬物的取放位移、关节行程、两个步行阶段、图像查询计数和录像帧数。

重建工作站可在远端工作目录中运行以下命令。工厂使用交付目录中的 `evidence/carry-reference.npz`，在新资产目录中命名为 `factory-route.npz`。厨房使用下文的骨盆路点及工位姿态。参考文件记录了来源运行和帧区间；重新采样 GR00T 可能得到不同轨迹。

```bash
PYTHONPATH=../usd/lib/python3.10/site-packages station-env/bin/python "$src/missions/build_stations.py" \
  --hall experiments/inspection-room/aisle-scene/inspection.usdc \
  --cabinet zero/assets/task2/usd/assets/B1/task2_guizi_rigid.usdc --output "$assets"
```

## 骨盆路点版厨房

[视频与结果](../../../outputs/long-horizon/kitchen-close-down/README.md) · [单变量按泵对照](../../../outputs/kimodo/pelvis-waypoints/README.md)

从输入重建三条运动参考使用与按泵试验相同的 L40S 主机模型环境。在 `long-horizon-work` 运行：

```bash
p=experiments/pelvis-input
out=$(mktemp -d "$PWD/pelvis-motion-XXXXXXXX")
export CUDA_VISIBLE_DEVICES=1 HF_HUB_CACHE="$PWD/$p/models/hub" HF_HUB_OFFLINE=1 TEXT_ENCODER_MODE=local
PYTHONPATH="$PWD/third_party/kimodo" "$p/runtime/venv/bin/python" \
    examples/g1_loco_manipulation/kimodo_sonic/pelvis_experiment.py "$out" \
    --study kitchen --station-headings --sink-offset .16
```

`pelvis-routes.npz` 保存生成的 50 Hz 参考及 30 Hz 输入路点帧号。准备新的工作站资产目录时，复制原家庭房间的 `serving-bench.usda`、`home-secure.usda`、`home-panel.usda`，再加入新生成的 `pelvis-routes.npz` 和交付证据中的 `station-postures.npz`。该姿态文件的 `source` 字段记录了采样时的交付路径；对应原始轨迹保存在远端工作目录的 `experiments/mixed-missions/run-home-i/rollout.npz`。姿态文件保存该原始回合控制步 705–1194 的腰臂姿态，以及控制步 1400、3030、4790 的工位姿态。将证据中的 `waypoint-plan.json` 与使用的运动参考配套保存。
