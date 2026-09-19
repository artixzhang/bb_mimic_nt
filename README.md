# BB Mimic NT：Unitree G1 跳跃投篮 Teacher Policy

本项目在 Isaac Lab 2.3 / Isaac Sim 5.1 中实现 G1–篮球–篮筐的 manager-based PPO 模仿环境。当前只包含 Teacher policy：100 条定点跳跃投篮轨迹均匀混训，不包含 AMP、Student 蒸馏、XGen 或真机部署。

## 环境与数据

- 训练环境：`BbMimicNT-G1-Shoot-v0`
- 确定性播放环境：`BbMimicNT-G1-Shoot-Play-v0`
- 仿真 / 策略频率：200 Hz / 100 Hz（decimation 2）
- 动作：29 维 reference-relative PD residual；Gaussian policy 输出经 `tanh` 约束，最终 PD 目标限制在机械关节限位内，再按每环境 0–4 个策略步延迟执行
- Teacher observation：固定 479 维，包含当前状态、reference pose/velocity/contact、历史、phase、reference speed 和 6 维 DR 特权参数
- 原始轨迹：`source/bb_mimic_nt/assets/trajectory/shoot_batch_0918.pkl`
- 预处理缓存：同目录 `shoot_batch_0918_processed.pt`（自动生成并被 Git 忽略）
- 场景资产配置：`source/bb_mimic_nt/bb_mimic_nt/objects/`；环境直接复用其中的篮球、floating hoop 和地面配置

轨迹被解释为每个并行环境自己的 local-world 坐标；写入仿真时才叠加 env origin。缓存包含 schema、源文件与 URDF 哈希、有效长度、padding mask、29 DoF、四路 contact、逐帧 `push_available`、FK link / hand anchor 及差分速度。哈希或 schema 过期时，训练会自动重建。

## 安装与预处理

请先安装 Isaac Lab，并在其 Python 环境中执行：

```bash
python -m pip install -e source/bb_mimic_nt
python scripts/preprocess_trajectory.py
```

强制重建或指定输入输出：

```bash
python scripts/preprocess_trajectory.py \
  --input source/bb_mimic_nt/assets/trajectory/shoot_batch_0918.pkl \
  --output source/bb_mimic_nt/assets/trajectory/shoot_batch_0918_processed.pt \
  --force
```

## 训练

默认使用 4096 个环境、每次 rollout 96 步、5000 次 PPO iteration。RSI 概率在前 30% policy steps 从 0.8 线性降到 0。

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

`training.py` 提供 PPO rollout 与 iteration 的默认值。训练入口把实际 `--max_iterations` 和 `num_steps_per_env` 写入 DR 与 RSI 课程；恢复训练时从 checkpoint iteration 接续。

## Domain Randomization

每次环境 reset 独立采样，直到下次 reset 保持不变。`env_cfg.dr` 为七项 DR 分别提供 `enabled`、`start_fraction` 和 `end_fraction`，并集中配置标称值、采样幅度和推力持续时间；默认在训练进度 20% 前使用标称值，20%–60% 线性扩大范围。六维特权观测依次为延迟、篮球质量、PD 比例、手部摩擦、脚部摩擦、腿部及骨盆 link 质量比例，按完整范围映射到 [-1, 1]，标称值均为 0。

| 项目 | 标称值 | 完整采样范围 |
| --- | ---: | ---: |
| 最终 PD 目标执行延迟 | 2 个 100 Hz 步 | 0–4 步，观测为 `(steps-2)/2` |
| 篮球质量 | USD 默认质量 | 默认值 ±5% |
| 29 个关节的 PD 刚度 | 机器人默认值 | `Kp` 比例 ±10%，`Kd` 比例为其平方根 |
| 双手静、动摩擦 | 0.8 | 0.6–1.0 |
| 双脚静、动摩擦 | 0.9 | 0.6–1.2 |
| 骨盆及八个髋/膝 link 质量 | 各 link 默认质量 | 共用比例 ±10%，惯量同比例变化 |
| `torso_link` 外力和力矩 | 0 | 世界坐标 X/Y 各 ±200 N、Z ±50 N，力矩各轴 ±3 Nm；最长 200 ms |

外力每次 rollout 最多触发一次，起点从当前 RSI 帧之后的 `push_available=1` 参考帧均匀选取。标记变为 0 时立即停止脉冲。播放和评估关闭随机化及外力，保留固定 2 步执行延迟。

TensorBoard 除 reward/termination 外，还记录 `Metrics/motion/error/*` 的原始 reference tracking error，以及 `Metrics/motion/control/*` 的 action 变化、PD target 变化、action 饱和率、关节速度和力矩占比。这些指标带有物理单位，适合排查 reward 上升但动作抖动的问题。

`reg/joint_jerk` 根据连续三个 100 Hz 策略步的关节速度计算加速度变化率，reset 后前两步不计罚。默认权重为 `2e-10`，训练时可通过 `Reward/reg/joint_jerk` 监控原始代价。

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
