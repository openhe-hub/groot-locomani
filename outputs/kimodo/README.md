# Kimodo + SONIC · 三栏验收

三部对比片均为 1600 × 720，25 fps。从左到右依次为 SOMA 77 关节人体网格、G1 参考姿态和仿真机器人的物理执行。顶部是当前提示词。人体使用同一时刻的 G1 参考骨段方向和骨盆姿态进行重定向，再按 SOMA 官方网格的蒙皮权重变形。人体身材与 G1 的比例不同，脚和手的绝对位置会有差异。

## 动作合集

- [11 个基础动作，45 秒](https://github.com/openhe-hub/groot-locomani/releases/download/media-v1/01_basic_actions.mp4)。按时间依次为 0 秒站立、4 秒前行、8 秒后退、12 秒侧步、16 秒转身、20 秒原地踏步、24 秒跑步、28 秒下蹲起身、33 秒双手上举、37 秒挥手、41 秒拍手。
- [三条衔接动作，33.24 秒](https://github.com/openhe-hub/groot-locomani/releases/download/media-v1/02_transitions.mp4)。第一条（0–10.68 秒）是站立、前行、下蹲起身、站稳。第二条（10.68–21.44 秒）是前行、挥手、转身、站稳。第三条（21.44–33.24 秒）是侧步、伸手、退步、拍手。每条均将短参考做水平位置、朝向和姿态过渡，再由 SONIC 在单次连续物理回合中执行。合集在三条回合之间切镜。

## 骨盆路点实验

[配对按泵视频与 8 个种子的实测结果](pelvis-waypoints/README.md)

## 厨房操作

[三项厨房任务合集](https://github.com/openhe-hub/groot-locomani/releases/download/media-v1/03_kitchen_tasks.mp4) 共 27.44 秒。依次展示按两次洗洁精泵（0:00）、关微波炉门（0:10.48）、推麦片盒（0:18.96）。每段保留三栏对比和手部交互特写。

三项任务在固定的 D4RL/Adept Kitchen 仿真布局下运行，食品外观来自 Robosuite。第一项在岛台上增加了带弹簧滑轨的洗洁精泵。Kimodo 提供移动和站立参考，SONIC 跟踪身体，右臂根据仿真物体位姿进行逆运动学修正。下表记录所选起点的一次执行结果。

| 视频 | 验收结果 | 对照 |
|---|---|---|
| [走到料理台，按两次洗洁精泵](https://github.com/openhe-hub/groot-locomani/releases/download/media-v1/soap.mp4) | G1 前进 1.42 米，两次按压分别为 38.35 和 38.73 毫米。泵头分别在 5.56 秒和 7.26 秒回位，机器人最后撤手站稳。验收针对机械行程。 | 同一起点关闭手臂任务控制后，泵头接触和完整按压均为 0 次。 |
| [把麦片盒推到绿垫](https://github.com/openhe-hub/groot-locomani/releases/download/media-v1/groceries.mp4) | G1 前进 1.37 米，右手接触 41 个控制周期，盒子沿台面前移 27.75 厘米。撤手后盒子直立并停在垫上。 | 相同起点和基础动作，关闭手臂任务控制后，盒子沿推进方向位移小于 0.01 厘米。 |
| [关微波炉门](https://github.com/openhe-hub/groot-locomani/releases/download/media-v1/microwave.mp4) | G1 前进 1.28 米，将门从 60.73° 推至 1.44° 后撤手。接触记录包含右腕、右肘和右肩连杆。 | 相同起点关闭手臂任务控制后，门最终仍开着 60.60°。 |

每个任务目录保留提示词、关节参考、实际状态、测量文件和视频。厨房任务的 `approach_only.json` 是关闭手臂任务控制的独立复跑。逐项动作及三条连续衔接的原始记录位于 `motions/`。

复跑命令见[项目指南](../../examples/g1_loco_manipulation/README.md)。运行环境为 Python 3.11、PyTorch 2.7.1+cu128、MuJoCo 3.3.7 和 ONNX Runtime 1.23.2。Kimodo 提交为 `e1c219c99b4b`，Zero-HLM-Bench 为 `b481409f2cd6`，G1 机器人模型来自 GR00T-WholeBodyControl 的 `b8ff31d64c2e`。
