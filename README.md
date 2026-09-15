# BB Mimic NT：Unitree G1 跳跃投篮 Teacher Policy

本项目在 Isaac Lab 2.3 / Isaac Sim 5.1 中实现 G1–篮球–篮筐的 manager-based PPO 模仿环境。v0.3.0 只包含 Teacher policy：100 条定点跳跃投篮轨迹均匀混训，不包含 AMP、Student 蒸馏、XGen 或真机部署。

## 环境与数据

- 训练环境：`BbMimicNT-G1-Shoot-v0`
- 确定性播放环境：`BbMimicNT-G1-Shoot-Play-v0`
- 仿真 / 策略频率：200 Hz / 100 Hz（decimation 2）
- 动作：29 维 reference-relative PD residual；Gaussian policy 输出经 `tanh` 平滑约束，反馈 residual 使用 40 ms 一阶低通，目标限制在机械关节限位内，并按 reference/执行器速度上限限制每步变化；reference pose 本身不经过低通
- Teacher observation：固定 473 维，包含当前状态、reference pose/velocity/contact、历史、phase 与 reference speed
- 原始轨迹：`source/bb_mimic_nt/assets/trajectory/shoot_batch_0910.pkl`
- 预处理缓存：同目录 `shoot_batch_0910_processed.pt`（自动生成并被 Git 忽略）
- 场景资产配置：`source/bb_mimic_nt/bb_mimic_nt/objects/`；环境直接复用其中的篮球、floating hoop 和地面配置

轨迹被解释为每个并行环境自己的 local-world 坐标；写入仿真时才叠加 env origin。缓存包含 schema、源文件与 URDF 哈希、有效长度、padding mask、29 DoF、四路 contact、FK link / hand anchor 及差分速度。哈希或 schema 过期时，训练会自动重建。

## 安装与预处理

请先安装 Isaac Lab，并在其 Python 环境中执行：

```bash
python -m pip install -e source/bb_mimic_nt
python scripts/preprocess_trajectory.py
```

强制重建或指定输入输出：

```bash
python scripts/preprocess_trajectory.py \
  --input source/bb_mimic_nt/assets/trajectory/shoot_batch_0910.pkl \
  --output source/bb_mimic_nt/assets/trajectory/shoot_batch_0910_processed.pt \
  --force
```

## 训练

默认使用 4096 个环境、每次 rollout 24 步、3000 次 PPO iteration。RSI 概率按共享配置在前 30% policy steps 从 0.8 线性降到 0。

v0.3.0 改变了 action 语义和 observation 维数，不能加载 v0.2.0 的 checkpoint。新训练与自动播放使用独立的 `logs/rsl_rl/g1_shoot_teacher_v2`，必须重新训练。

```bash
python scripts/rsl_rl/train.py \
  --task BbMimicNT-G1-Shoot-v0 \
  --headless
```

短程冒烟示例：

```bash
python scripts/rsl_rl/train.py \
  --task BbMimicNT-G1-Shoot-v0 \
  --num_envs 8 --max_iterations 2 --headless
```

`training.py` 是 PPO rollout/iteration 与 RSI decay steps 的单一配置源。若要永久修改训练长度，请同步修改其中的 `PPO_MAX_ITERATIONS`；CLI 的临时 `--max_iterations` 不会重写环境构造时已确定的 RSI 日程。

TensorBoard 除 reward/termination 外，还记录 `Metrics/motion/error/*` 的原始 reference tracking error，以及 `Metrics/motion/control/*` 的 action 变化、PD target 变化、action 饱和率、关节速度和力矩占比。这些指标带有物理单位，适合排查 reward 上升但动作抖动的问题。

## 播放、交互与评估

标准播放脚本保持兼容：

```bash
python scripts/rsl_rl/play.py \
  --task BbMimicNT-G1-Shoot-Play-v0 \
  --checkpoint /path/to/model.pt
```

交互播放：

```bash
python scripts/rsl_rl/play_interactive.py \
  --checkpoint /path/to/model.pt --clip-id 0 --num-envs 4

python scripts/rsl_rl/play_interactive.py \
  --checkpoint /path/to/model.pt --all-clips

# 省略 --checkpoint 时，自动选择最新 run 中编号最大的 model_*.pt
python scripts/rsl_rl/play_interactive.py --clip-id 0
```

按键：`V` 显示/隐藏四个 reference body 点、篮球点与篮圈中心，`,` / `.` 切换 clip，`R` 重置。普通多环境模式中所有 env 使用同一 clip；`--all-clips` 创建 100 个 env 并一一对应所有 clips。交互模式关闭 RSI、push、随机 clip 和 observation corruption，并保留与训练一致的提前失败 termination；确定性 adaptive speed 保持启用。

若只想完整检查 reference（即使策略已经失败或机器人已经摔倒），可添加 `--ignore-failures`。该模式关闭提前失败 reset，因此不应把倒地后的画面当作策略的正常 episode 表现：

```bash
python scripts/rsl_rl/play_interactive.py --clip-id 0 --ignore-failures
```

Floating hoop 使用 WXYZ 四元数 `(0, 0, 0, 1)` 绕 Z 轴旋转 180°。USD 中篮圈中心相对刚体根节点的偏移为 `(2.17577, 0, 3.02)`；reset 时会反算刚体根位置，保证物理篮圈中心与轨迹的 `hoop_pos_w` 对齐。

批量评估：

```bash
python scripts/rsl_rl/evaluate_shooting.py \
  --checkpoint /path/to/model.pt --headless
```

结果默认写到 `outputs/shooting_evaluation.json`。命中判定仅作为指标：球从篮圈上方向下穿过水平面，且水平偏差不超过 0.20 m。

## 测试

不启动 Isaac Sim 的 CPU 测试：

```bash
PYTHONPATH=source/bb_mimic_nt pytest -q tests
```
