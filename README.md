# BB Mimic NT：Unitree G1 跳跃投篮 Teacher 与 Student

本项目在 Isaac Lab 2.3 / Isaac Sim 5.1 中实现 G1–篮球–篮筐的 manager-based PPO 模仿环境，以及独立的 Student DAgger 蒸馏、评估和部署导出流程。Teacher 使用 100 条定点跳跃投篮轨迹均匀混训；Student 只拟合冻结 Teacher，不包含 task reward 或后续 finetune。

## 环境与数据

- 训练环境：`BbMimicNT-G1-Shoot-v0`
- 确定性播放环境：`BbMimicNT-G1-Shoot-Play-v0`
- 仿真 / 策略频率：200 Hz / 100 Hz（decimation 2）
- 动作：29 维 reference-relative PD residual；Gaussian policy 输出经 `tanh` 约束，最终 PD 目标限制在机械关节限位内，再按每环境 0–2 个策略步延迟执行
- Teacher observation：固定 674 维，包含当前状态、reference pose/velocity/contact、object-control state、历史、phase、reference speed 和 6 维 DR 特权参数
- 原始轨迹：`source/bb_mimic_nt/assets/trajectory/batch_setshot_1009.pkl`
- 预处理缓存：同目录 `setshot_1008_processed.pt`（自动生成并被 Git 忽略）
- 场景资产配置：`source/bb_mimic_nt/bb_mimic_nt/objects/`；环境直接复用其中的篮球、floating hoop 和地面配置

轨迹被解释为每个并行环境自己的 local-world 坐标；写入仿真时才叠加 env origin。缓存包含 schema、源文件与 URDF 哈希、有效长度、padding mask、29 DoF、四路 contact、逐帧 `push_available`、`release`、FK link / hand anchor 及差分速度。哈希或 schema 过期时，训练会自动重建。

每个原始 clip 可提供长度为帧数的二值 `release`：`0` 表示持球/控制阶段，`1` 表示自由/已释放；`0 -> 1` 是出手/释放事件，`1 -> 0` 是拿球事件。它与瞬时接触标签相互独立，因此可以表达拍球、再次接球和传球等多段交互。旧轨迹若没有该字段，预处理器会以“无左右手 contact”生成兼容值；新数据建议显式标注。

## 安装与预处理

请先安装 Isaac Lab，并在其 Python 环境中执行：

```bash
python -m pip install -e source/bb_mimic_nt
python scripts/preprocess_trajectory.py
```

强制重建或指定输入输出：

```bash
python scripts/preprocess_trajectory.py \
  --input source/bb_mimic_nt/assets/trajectory/batch_setshot.pkl \
  --output source/bb_mimic_nt/assets/trajectory/batch_setshot_processed.pt \
  --force
```

## 训练

默认使用 16384 个环境、每次 rollout 48 步、4000 次 PPO iteration。Teacher 和 Student 的默认环境数统一由 `source/bb_mimic_nt/bb_mimic_nt/training.py` 中的 `NUM_ENVS` 控制（多 GPU 时为每个进程的环境数）；`PPO_STEPS_PER_ENV` 控制 Teacher rollout，Student 的 `DAGGER_STEPS_PER_ENV` 默认跟随它。RSI 概率在前 30% policy steps 从 0.8 线性降到 0，当前为 57600 步。命令行和 Hydra 参数仍可覆盖默认配置。

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

## 持球时间与采样比例

在 `bb_mimic_nt_env_cfg.py` 的 `CommandsCfg.motion` 中设置：

```python
pre_hold_s=1.0,
post_hold_s=1.0,
hold_sampling_weights=(0.15, 0.15),
```

两个权重分别表示投篮前、投篮后的预期训练时间相对投篮动作时间的预算，不是 episode 概率。每次 reset 根据剩余参考动作的标称 1x 时长 `T`，分别以 `min(1, weight*T/hold_s)` 的概率加入完整持球段；未选中的阶段直接跳过，投篮动作照常执行。持球段选中后保留全部时长，因此将时间改成 5 秒或 10 秒仍能训练连续站立，同时降低被选中的次数，避免冲淡投篮样本。零时间或零权重跳过对应持球段；`None` 表示每次执行完整前后持球时间。短持球段无法达到预算时全部保留，不额外延长。

默认预算在没有提前失败、关闭 RSI、动作按 1x 推进且持球时长足够时，对应约 76.9% 投篮、11.5% pre-hold、11.5% post-hold。提前失败和 adaptive speed 会改变实际占比；RSI 回合从动作中间开始，不添加 pre-hold，post-hold 根据剩余动作时长采样。日志 `sampling/{pre_hold,active,post_hold}/fraction` 汇报重置环境实际累计的阶段时间占比，不能当作严格全局配额。长 hold 会减少每单位训练步数内的重置次数，但预期投篮时间占比受上述预算控制。

Teacher/Student 的 Play 配置关闭持球采样，完整执行设定时长。Student 蒸馏从 Teacher 归档环境继承持球时间和采样预算；旧 checkpoint 没有此字段时保持旧的完整持球逻辑。导出 JSON 同时记录训练预算和完整持球时间，部署无需复现训练的随机跳过机制。

先微调 Teacher，让 phase=0 的初始姿势持球稳定，再蒸馏 Student。单独给旧 Teacher/Student 增加等待时间不能保证它已学会持球；有限 pre-hold 的训练也不等于无限等待已验证。

## Object 奖励调节

奖励参数在 `mdp/rewards.py` 的 `UnifiedRewardCfg`。当前 object 的顶层权重为 `global_weights[1]=0.25`，持球阶段门控为 1，ACTIVE 的自由球阶段由 `object_free_phase_weight=0.15` 降权，POST_HOLD/DONE 则关闭 object 和 relative 奖励。它不是全部关闭。

需要更强的出手轨迹约束，可以手动将 `object_free_phase_weight` 从 `0.15` 调至 `0.3` 或 `0.5`，观察动作和投篮成功率后再决定是否设为 `1.0`（自由球阶段不降权）。也可在 Teacher 训练启动时传入 Hydra 覆盖：

```bash
python scripts/rsl_rl/train.py --task BbMimicNT-G1-Shoot-v0 --headless \
  env.rewards.unified_mimic.params.settings.object_free_phase_weight=0.5
```

保留 POST_HOLD 门控：末帧 reference 球位置不再推进，继续奖励静止末帧球位置会错误约束仍在飞行的真实球。该奖励约束参考球的位置、方向和速度，不是直接的命中奖励；用 `shot_success` 检查投篮精度。提高门控权重改变 Teacher 的 PPO 奖励，Student DAgger 没有任务奖励，需要再蒸馏更新后的 Teacher。本次实现没有修改 object 奖励配置。

## Domain Randomization

每次环境 reset 独立采样，直到下次 reset 保持不变。`env_cfg.dr` 为各项 DR 分别提供 `enabled`、`start_fraction` 和 `end_fraction`，并集中配置标称值、采样幅度和推力持续时间；默认在训练进度 20% 前使用标称值，20%–60% 通过 cubic smoothstep S 曲线平滑扩大范围，课程起止点斜率均为零。六维特权观测依次为延迟、篮球质量、PD 比例、手部摩擦、脚部摩擦、腿部及骨盆 link 质量比例，按完整范围映射到 [-1, 1]，标称值均为 0。

| 项目 | 标称值 | 完整采样范围 |
| --- | ---: | ---: |
| 最终 PD 目标执行延迟 | 0 个 100 Hz 步 | 0–2 步，观测为 `steps/2` |
| 篮球质量 | USD 默认质量 | 默认值 ±5% |
| 29 个关节的 PD 刚度 | 机器人默认值 | `Kp` 比例 ±10%，`Kd` 比例为其平方根 |
| 双手静、动摩擦 | 0.8 | 0.6–1.0 |
| 双脚静、动摩擦 | 0.9 | 0.6–1.2 |
| 骨盆及八个髋/膝 link 质量 | 各 link 默认质量 | 共用比例 ±10%，惯量同比例变化 |
| 三类 Push 外力和力矩（各自独立配置） | 0 | 默认世界坐标 X/Y 幅值 8–20 N、Z 幅值 5–10 N，随机正负；力矩各轴 ±5 Nm；最长 800 ms |

延迟由 `env_cfg.dr.delay_nominal_steps`（默认 0）和 `delay_max_offset_steps`（默认 2）控制。课程半径为 `floor(delay_max_offset_steps * strength)`，reset 时在 `[max(0, nominal-radius), nominal+radius]` 内均匀采样整数步，负下界先截为 0；不会先采样负数再将样本裁成 0。延迟课程在训练进度 30%–70% 扩大范围：固定 0 → 0–1 → 0–2 步。标称值是课程起点和播放值，完整均匀分布的中位数为 1 步。要改为标称 1、采样 0–2 步，设为 `1, 1`；固定零延迟设为 `0, 0`。延迟特权观测通式为 `(steps-nominal)/max(offset, 1)`。

三类 Push 每个 episode 各独立抽取一次是否触发（默认概率 0.8），各最多触发一个脉冲，可以同时发生。`dr.push.enabled`、`dr.global_push.enabled`、`dr.object_push.enabled` 分别控制开关，训练默认均开启。Teacher 三者默认沿用 Push 的 15%–80% smoothstep 课程；Student 三者沿用 Student Push 的 10%–35% 课程。力、力矩在 reset 时按课程强度缩放，概率不随课程缩放。

- `push`：由逐帧 `push_available` 控制机器人推力。
- `global_push`：忽略 `push_available`，从本次剩余的 pre-hold、动作段、post-hold 全时间段采样一次。
- `object_push`：由逐帧 `obj_push` 控制，自动作用于轨迹物体 `ball`。

采样按参考时间均匀进行，动作段自适应速度会改变实际触发时刻；RSI 之前的时间和未选中的 hold 不参与采样。pre-hold/post-hold 沿用首帧/末帧的可推标记。门控关闭或进入 DONE 时立即结束对应脉冲，结束后不再重触发。80% 是 episode 的独立抽取概率；没有可用时段、课程强度为零或 episode 提前结束时，实际遇到 Push 的比例可能更低。

每类用 `<类型>_force_min_n`、`<类型>_force_max_n`、`<类型>_torque_max_nm`、`<类型>_duration_s`、`<类型>_probability` 单独配置，例如 `global_push_force_max_n`。机器人两类用 `push_link_weights` / `global_push_link_weights` 指定精确 link 名和相对权重，例如 `{"torso_link": 3.0, "pelvis": 1.0}`；每类每次只选一个 link，默认 `{"torso_link": 1.0}`。同一 link 上的两类力和力矩相加。三类共用 `EventsCfg.push` 的逐策略步事件，以一次合成写入避免覆盖；独立开关放在 `dr`，无需增加多个互相覆盖的事件。

缓存 schema 已升级至 7，预处理要求 `obj_push` 为逐帧 bool/0/1；旧缓存会自动判为过期，需由更新后的原始数据重建。录制脚本也逐帧保留此字段。播放和评估默认关闭三类 Push。


TensorBoard 除 reward/termination 外，还记录 `Metrics/motion/error/*` 的原始 reference tracking error，以及 `Metrics/motion/control/*` 的 action 变化、PD target 变化、action 饱和率、关节速度和力矩占比。`Metrics/motion/phase/{controlled,free}/*` 按 `release` 分段汇总，`error/joint/*` 与 `control/joint/*` 提供逐关节误差和动作幅度。这些指标带有物理单位，适合排查 reward 上升但动作抖动的问题。

`reg/action_accel` 使用策略直接输出的 action（`env.action_manager.action`，未经 tanh、残差缩放或执行延迟），按连续三个策略步计算二阶差分平方均值 `mean((a_t - 2*a_{t-1} + a_{t-2})²)`。与 `action_rate` 一致，该项衡量逐步变化，不做 dt 归一化。该项不依赖物理引擎的关节速度，reset 后前两步不计罚。默认 `action_accel_weight` 为 `1e-11`，训练时可通过 `Reward/reg/action_accel` 监控原始代价；权重可独立调整。

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

## Student DAgger

Student 训练和播放任务分别为 `BbMimicNT-G1-Shoot-Student-DAgger-v0` 与 `BbMimicNT-G1-Shoot-Student-Play-v0`。Teacher 保持 674 维无噪声、无观测延迟输入；Student 默认使用 469 维显式缩放输入：phase、五帧重力投影、pelvis 角速度、nominal-relative 关节位置、关节速度、实际执行动作，以及 reset 时固定的轨迹篮筐柱坐标 `hoop_pos_cyl`（`[r, theta, z]`）。机器人传感量训练时加入可调高斯噪声与每回合采样一次的 0–3 步延迟，推理固定为标称 1 步。`student_observation.history_length=5` 在 100 Hz 下覆盖 40 ms，输入维度为 `93 * history_length + 4`。历史、导出和网络输入维度共用这一配置。

Student 输出 29 维 nominal-relative 关节位置（rad），直接形成 PD target，不经过滤波。它与 Teacher 共用 PD 参数、动作执行延迟以及全部动力学/外力 DR；动作延迟参数从 Teacher checkpoint 对应的环境配置读取，新默认为标称 0、采样 0–2 步，已有 checkpoint 保留原配置。Student 观测延迟由 `student_observation.delay_nominal_steps=1`、`delay_min_steps=0`、`delay_max_steps=3` 独立控制；只把标称改为 0 时，完整采样范围仍是 0–3 步，推理固定 0 步。三个值须满足 `0 <= min <= nominal <= max`；固定零延迟设为 `0, 0, 0`。观测噪声和延迟范围在前 20% 训练内按 smoothstep 增强，初始延迟固定在标称值；设置 `student_observation.full_strength=True` 可直接使用完整观测扰动。Student 使用独立的 `StudentDomainRandomizationCfg` 控制课程，Teacher checkpoint 只同步扰动范围，不覆盖课程及播放开关，设置对应 `dr.<项>.full_strength=True` 可使用完整范围，reference 始终以 1× 推进。DAgger 在前 10% 完全由 Teacher 执行，10%–30% 通过 smoothstep 概率逐环境混合，从 30% 起完全由 Student 执行；所有访问状态始终由冻结 Teacher 标注。

默认训练 1000 轮，Student 课程按本次总迭代数计算，可在 `bb_mimic_nt_student_env_cfg.py` 独立调整：

| 项目 | 开始扩展 | 达到完整范围 | 默认迭代数 |
| --- | ---: | ---: | ---: |
| 观测噪声 / 观测延迟 | 0% | 20% | 0–200 |
| 质量 / PD / 摩擦 DR | 5% | 25% | 50–250 |
| 动作执行延迟 | 10% | 30% | 100–300 |
| Push | 10% | 35% | 100–350 |
| Student 接管概率 | 10% | 30%（完全接管） | 100–300 |

350 轮后，Student 独立执行且全部扰动达到完整范围，剩余约 650 轮用于完整扰动训练。`--max-iterations` 会按比例缩放这些时间点；Teacher 的训练课程保持原配置。播放关闭随机化的设置也不会被加载 Teacher 配置重新开启。

```bash
python scripts/rsl_rl/train_dagger.py \
  --teacher-checkpoint logs/rsl_rl/g1_shoot_teacher_v2/<run>/model_4999.pt \
  --headless
```

Teacher 的网络结构、动作缩放、DR 范围、参考轨迹、前后 HOLD 和 reset 关节噪声从所选 checkpoint 的 `params/` 读取。Student 保持关闭 RSI 和自适应速度。训练前即写出 `params/env.yaml`、`params/agent.yaml` 和 `params/student_config.json`；每个 `model_*.pt` 都包含独立导出所需的模型结构与部署 metadata。默认每轮收集 48 步并更新 32 次，batch size 跟随 `NUM_ENVS`（当前 16384）；回放容量由 `training.py` 按 `2 * NUM_ENVS * max(32, DAGGER_STEPS_PER_ENV)` 推导，当前为 1572864，一半保存最新 FIFO 状态、一半作为会持续更新的全历史 reservoir。16384 个环境时 recent 可容纳完整 48 步；469 维输入加 29 维标签时两分区合计约占 3.13 GB（2.92 GiB）显存，每个 batch 默认 75% 采样最新状态，保留最新闭环状态与历史覆盖。这些是默认配置的联动计算；若通过 CLI/Hydra 单独覆盖环境数或 rollout，请相应检查 `--replay-capacity` 和 `--batch-size`。回放数据常驻训练 device，避免 rollout 期间 CPU/GPU 往返；如需在恢复训练时连同回放集恢复，可添加 `--save-replay`。

新默认配置需要从头训练，旧的 283 维 checkpoint 不能续训为 469 维。旧导出模型仍可播放，加载时会从 metadata 恢复其 3 帧历史和原延迟。DAgger 日志增加每关节目标 MAE、Student/Teacher 目标逐步变化 RMS、目标变化差、失败回合比例与回合步数；动作变化指标排除跨 reset 的跳变，均以 rad 为单位。

```bash
python scripts/rsl_rl/play_student.py \
  --student-weights logs/rsl_rl/g1_shoot_student_dagger_v1/<run>/exported/student_weights.pt \
  --clip-id 0

python scripts/rsl_rl/play_student.py \
  --student-weights logs/rsl_rl/g1_shoot_student_dagger_v1/<run>/model_400.pt \
  --clip-id 0

python scripts/rsl_rl/play_student.py \
  --student-weights /path/to/student_weights.pt --all-clips

python scripts/rsl_rl/evaluate_student.py \
  --teacher-checkpoint /path/to/teacher_model.pt \
  --student-weights /path/to/student_weights.pt --headless

python scripts/rsl_rl/export_student.py \
  --checkpoint /path/to/model_250.pt \
  --output outputs/student_export
```

交互播放支持 `V` reference 点、`,` / `.` 切换 clip、`P` 或空格暂停/恢复、`R` 重启、`--all-clips`、`--ignore-failures` 和 `--no-timeout`。导出目录包含 `student_weights.pt`、TorchScript、ONNX 与 `student_config.json`；配置记录观测顺序/缩放、关节顺序、nominal pose、安全限位、PD 参数和固定标称延迟。成对评估对相同 clip 分别运行 Teacher 与 Student，输出篮球首次触地相对篮筐的水平落点和最高高度。

Student Play 默认关闭 reset 关节位置/速度扰动和域随机化，同一 clip 每次从参考初始状态重启。需要测试扰动时，在 `play_student.py` 命令中添加 `--random-init`（开启训练配置中的 reset 关节噪声）或 `--domain-randomization`（开启完整强度的物理参数随机化、动作延迟采样和推力），两者可独立使用或同时开启。观测噪声仍关闭，观测延迟仍使用导出配置的固定值；RSI 和随机 clip 仍关闭。

Student 真机导出流程、phase/初始化契约、参考观测构造器及 READY 验证见 [部署说明](docs/student_deployment.md)。

### 从 Teacher rollout 导出训练 motion

```bash
python scripts/rsl_rl/record_teacher_motion.py \
  --checkpoint logs/rsl_rl/g1_shoot_teacher_v2/<run>/model_6000.pt \
  --pre_hold 1.0 --post_hold 3.0 \
  --num_envs 32 --export_dir outputs/teacher_motion --headless
```

使用 Isaac Lab 的 Python 环境运行。默认使用最多 32 个环境并行录制，每个环境一次录制一个 clip，
每批结束后开始下一批；可用 `--num_envs 64` 调整并行数，`--num_envs 1` 使用串行模式，
实际环境数不超过所选 clip 数。可用 `--clip_ids 0 2` 指定片段，输出始终保持源数据顺序
（指定 `--clip_ids` 时保持指定顺序，包括重复 ID），不受各片段结束先后影响。
并行模式保持相同录制规则，但多环境物理仿真可能产生浮点差异，不保证与串行结果逐位相同。
导出文件为 `<export_dir>/teacher_motion.pkl`，
与训练源 pkl 一样通过 `joblib.load` 读取为 `list[dict]`，保留所有字段、dtype、
关节及接触通道顺序。采样率为 100 Hz，包含初始帧与自动 reset 前的终止帧。
`root_pos/root_rot/dof/obj_pos/obj_rot` 取仿真实测值，位置减去环境原点，
`contact` 初始帧沿用源轨迹第一帧 label（reset 后尚无新物理步的接触力），后续帧取仿真实测值。
四元数为 WXYZ；`hoop_pos_w` 为实际篮圈中心，`release/push_available` 沿用当前参考阶段的语义标注。
前后 hold 期间仍由 teacher 控制并录制，时长按控制步量化；关闭 RSI、随机 hold 和自适应速度。
若 rollout 因摔倒等条件提前终止，会报错且不导出该批次。
