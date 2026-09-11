# BB Mimic NT：Unitree G1 跳跃投篮 Teacher Policy

本项目在 Isaac Lab 2.3 / Isaac Sim 5.1 中实现 G1–篮球–篮筐的 manager-based PPO 模仿环境。v0.5.0 只包含 Teacher policy：100 条定点跳跃投篮轨迹均匀混训，不包含 AMP、Student 蒸馏、XGen 或真机部署。

## 环境与数据

- 训练环境：`BbMimicNT-G1-Shoot-v0`
- 确定性播放环境：`BbMimicNT-G1-Shoot-Play-v0`
- 仿真 / 策略频率：500 Hz / 100 Hz（decimation 5）
- 动作：29 维 reference-relative PD residual；Gaussian policy 输出经 `tanh` 平滑约束，反馈 residual 使用 80 ms 一阶低通和最高 2 rad/s 的独立限速，目标限制在机械关节限位内；reference pose 本身不经过低通或 residual 限速
- Actor observation：固定 473 维，包含当前状态、reference pose/velocity/contact、历史、phase 与 reference speed
- Critic observation：473 维 actor observation 加上原始轨迹在未来 0.25 / 0.50 / 1.00 s 的 165 维 reference 路点；这些 privileged observation 不进入 actor
- 原始轨迹：`source/bb_mimic_nt/assets/trajectory/shoot_batch_0910.pkl`
- 预处理缓存：同目录 `shoot_batch_0910_processed.pt`（自动生成并被 Git 忽略）
- 场景资产配置：`source/bb_mimic_nt/bb_mimic_nt/objects/`；环境直接复用其中的篮球、floating hoop 和地面配置

GPU PhysX 不支持 CCD；因此环境不再请求一个会被静默忽略的 CCD 开关，而是使用 500 Hz 物理步进降低高速篮球穿透风险。Hoop 是否参与碰撞完全由引用的 USD 决定。

轨迹被解释为每个并行环境自己的 local-world 坐标；写入仿真时才叠加 env origin。缓存包含 schema、源文件与 URDF 哈希、有效长度、padding mask、29 DoF、四路 contact、FK link / hand anchor 及通用差分速度。篮球位置始终逐帧保留输入数据，不进行弹道拟合、出手速度推断、轨迹改写或外推。哈希或 schema 过期时，训练会自动重建。

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

默认使用 4096 个环境、每次 rollout 48 步、3000 次 PPO iteration。PPO 使用 `gamma=0.995`、`lambda=0.975`；RSI 概率按共享配置在前 30% policy steps 从 0.8 线性降到 0，并从每条轨迹的全部有效帧均匀采样。

v0.3.0 改变了 action 语义和 actor observation 维数，不能加载 v0.2.0 的 checkpoint。v0.5.0 新增 critic-only observation 并回退拟合球轨迹，因此应从头训练，训练与自动播放使用 `logs/rsl_rl/g1_shoot_teacher_v3`。

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

TensorBoard 除 reward/termination 外，还记录 `Metrics/motion/error/*` 的原始 reference tracking error，以及 `Metrics/motion/control/*` 的 raw/filtered action 变化、PD target 变化、action 饱和率、关节速度和力矩占比。这些指标带有物理单位，适合排查 reward 上升但动作抖动的问题。篮球在 active reference 中偏离目标 1 m 会触发 `object_tracking_error` termination。

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

Floating hoop 使用 WXYZ 四元数 `(0, 0, 0, 1)` 绕 Z 轴旋转 180°。USD 中篮圈中心相对资产根节点的偏移为 `(2.17577, 0, 3.02)`；reset 时会反算 Xform 根位置，保证篮圈中心与轨迹的 `hoop_pos_w` 对齐。Hoop 在框架中是纯 `AssetBaseCfg` 场景 Xform，不要求 USD 带 `RigidBodyAPI`，也不会探测、添加或修改 rigid body/collider；所有 physics schema 都由 USD 自己决定。训练默认引用不含 collider 的 `bb_hoop_floating.usd`。

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

Isaac Sim 场景与两次 PPO iteration 冒烟需要可工作的 NVIDIA 驱动。
