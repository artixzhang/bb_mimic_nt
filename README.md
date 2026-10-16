# BB Mimic NT：Unitree G1 跳跃投篮 Teacher 与 Student

本项目在 Isaac Lab 2.3 / Isaac Sim 5.1 中实现 G1–篮球–篮筐的 manager-based PPO 模仿环境，以及独立的 Student DAgger 蒸馏、评估和部署导出流程。Teacher 按 `skills.yaml` 的 episode 权重混合训练多个独立技能；Student 只拟合冻结 Teacher，不包含 task reward 或后续 finetune。

## 环境与数据

- 训练环境：`BbMimicNT-G1-Shoot-v0`
- 确定性播放环境：`BbMimicNT-G1-Shoot-Play-v0`
- 仿真 / 策略频率：200 Hz / 100 Hz（decimation 2）
- 动作：Teacher/Student 共用 29 维无量纲 direct action；输出 clip 到 ±100，目标为 `nominal + action × (0.25 × effort_limit / Kp)`，不做目标角度裁剪，再按每环境 0–2 个策略步延迟执行。scale 使用固定标称参数，详见 [动作接口](source/bb_mimic_nt/docs/actions.md)
- Teacher observation：默认 676 维（`673 + command.dimension`），包含当前状态、reference pose/velocity/contact、object-control state、历史、技能 one-hot command、reference speed 和 6 维 DR 特权参数
- 技能配置：`source/bb_mimic_nt/assets/trajectory/skills.yaml`
- 原始轨迹：各技能的 `motion_file`；每个文件对应一个同目录 `*_processed.pt` 缓存（自动生成并被 Git 忽略）
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

默认使用 8192 个环境、每次 rollout 48 步、7000 次 PPO iteration。Teacher 和 Student 的默认环境数统一由 `source/bb_mimic_nt/bb_mimic_nt/training.py` 中的 `NUM_ENVS` 控制（多 GPU 时为每个进程的环境数）；`PPO_STEPS_PER_ENV` 控制 Teacher rollout，Student 的 `DAGGER_STEPS_PER_ENV` 默认跟随它。Teacher/Student 共用 `RSI_INITIAL_PROBABILITY=0.3` 和 `RSI_DECAY_FRACTION=0.15`，RSI 从训练开始线性衰减，在各自预算的前 15% 完全退出；Teacher 默认退出期限为 50400 策略步（1050 轮）。命令行和 Hydra 参数仍可覆盖默认配置。

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

`training.py` 集中配置 RSI 初始概率、退出比例，以及 PPO rollout 与 iteration 的默认值。训练入口把实际 `--max_iterations` 和 `num_steps_per_env` 写入 DR 与 RSI 课程；恢复训练时从 checkpoint iteration 接续。

### 多卡训练与 CUDA 越界排查

例如每卡 8192 个环境、4 卡训练：

```bash
/workspace/isaaclab/_isaac_sim/python.sh -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node=4 \
  scripts/rsl_rl/train.py --task BbMimicNT-G1-Shoot-v0 \
  --headless --distributed --num_envs 8192 --max_iterations 3000
```

多卡默认对前馈 PPO 使用 CPU `randperm`，再把索引同步传到该进程的 GPU，绕开 GPU 随机排列生成路径。每轮先检查 rollout 是否完整、各字段尺寸是否一致、传输后的索引是否在本地 batch 范围内；异常会在取样前报告 rank、device 和尺寸/范围，不裁剪或重试错误索引。当前参数每轮传输约 3 MiB 索引，保持每轮一次打乱、各 epoch 复用，以及 RSL-RL 的 minibatch 切分方式。该改动只作用于当前 runner 的 storage。单卡和 recurrent policy 默认保留原采样器；单卡可用 `--ppo_minibatch_indices cpu` 启用检查，多卡可用 `--ppo_minibatch_indices native` 对照原采样器。

这项规避针对一次约 146 iteration 的崩溃：最先发生 `IndexKernel.cu: index out of bounds`，随后才出现 NCCL 等待超时；异步堆栈落在 PPO `policy.act()`。索引 kernel 的规模和末尾位置与 676 维 observation minibatch 相符，因此优先排查 PPO 取批。尚未用同步 CUDA 堆栈确认具体底层故障，也未在当前无 GPU 驱动的检查环境中完成四卡复现，不能据此认定是 PyTorch `randperm` 的已知缺陷。

若仍崩溃，在上述脚本参数末尾添加 `--debug_cuda`，它会在启动模拟器前设置 `CUDA_LAUNCH_BLOCKING=1`，让错误指向实际失败的操作；该模式较慢，仅用于诊断。需要分 rank 保存日志时，在 `scripts/rsl_rl/train.py` **之前**增加 torchrun 参数 `--log-dir logs/torchrun_debug --tee 3`。多卡入口开启完整 Hydra 堆栈和 torchrun 错误记录，并在 worker 异常后立即退出，避免 CUDA 出错后的模拟器析构阻塞、让其他 rank 再等待 10 分钟。

## 多技能配置与采样

训练配置位于 `source/bb_mimic_nt/assets/trajectory/skills.yaml`，支持 `defaults` 和每个技能的局部覆盖。初始化目前只支持 `robot/ball/hoop: reference`，沿用轨迹初始化及现有 reset 关节噪声。可通过 `env.commands.motion.skills_path=/absolute/path/skills.yaml` 选择另一份配置。

每次 episode reset 先按 `sampling_weight` 归一化选择技能，再在该技能内均匀选择 clip；权重不按轨迹数量或时长调整。一次 episode 只执行一个技能，结束后按概率执行末帧保持并终止，不串接技能。长 hold 会增加该技能在实际训练步数中的占比，episode 权重不保证训练时间占比。

技能按 `command_id` 排序，one-hot 第 `command_id` 位为 1，宽度为 `command.dimension`。ID 必须唯一且落在 `[0, dimension)`。Teacher 和 Student 都接收此 command，整个 episode 保持不变。当前对应 `hold=[1,0,0]`、`shoot=[0,1,0]`、`stand_to_hold=[0,0,1]`。

`rsi.enabled` 决定该技能是否参与全局 RSI 课程；关闭的技能始终从首帧开始。`end.trigger_probability` 默认 0.3，每 episode 独立抽一次，和 RSI 独立：选中后静止跟踪末帧，reference 速度置零，完整保持 `end.duration_s` 秒；未选中则在 reference 完成后终止回合。固定秒数直接使用，`[minimum, maximum]` 每 episode 均匀采样；概率不会缩短被选中的保持段。设为 0 完全跳过 end，设为 1 每次完整执行。

单帧轨迹从初始化直接进入末帧保持，当前 hold 配置的 `trigger_probability=1.0` 保证持续持姿。单帧技能若关闭 end，会在下一次 termination 检查时结束，不进行零时间递归 reset。仿真终止与 reset 按策略步检查，跳过保持的末帧仍可能占一个终止检查步，不构成长时间保持。

Teacher、Student、Play 共用上述规则；播放时需要每次完整展示 end，可把对应技能概率设为 1。新训练归档保存解析后的完整 `skills_config`；早期已解析归档没有 `trigger_probability` 时沿用每次执行 end 的行为。

PPO 每轮 iteration 先收集 `NUM_ENVS * num_steps_per_env` 条 transition，再用这批数据做多次优化。当前单 GPU 默认为 `8192 * 48 = 393216` 条，5 个 epoch、4 个 mini-batch 共 20 次 optimizer 更新；iteration 不固定 episode 数量。跳过保持会增加同一 rollout 中的动态动作比例及 episode 重启次数。`sampling/{active,end_hold}/fraction` 记录结束回合的实际时间分配，`sampling/end_trigger_fraction` 记录触发抽样比例，`sampling/end_hold_duration_s` 记录每回合 end 时长（含跳过的零秒）；它们由刚结束的回合汇总，不是每个 rollout 的严格全局配额。hold 概率为 1、另外两个技能为 0.3 时，总体期望触发率约为 37%，而非 30%。

## Object 奖励调节

`object.interaction_enabled` 为技能全局开关：false 时整个 episode 关闭 object、relative 和手球 contact 奖励，并关闭对应的交互距离终止条件。脚地 contact 和身体跟踪继续生效。

`end.object_interaction_enabled` 默认 false；设为 true 时末帧保持期间继续训练物体交互，但仍受技能全局开关约束。当前 hold 在保持期间开启交互；shoot 在结束后关闭；stand_to_hold 整个 episode 关闭。

奖励参数在 `mdp/rewards.py` 的 `UnifiedRewardCfg`。自由球时的 object 和 relative 权重分别由 `object_free_weight`（默认 1.0）和 `relative_free_weight`（默认 0.3）控制，仍保留原有 `release` 数据语义。Student DAgger 只拟合冻结 Teacher，不包含 task reward；它使用同一技能采样、RSI 和保持逻辑。

动作和观测语义已升级为 direct action，旧 Teacher/Student checkpoint 需要重新训练。技能配置、ID 映射和 motion profile 随训练归档和 Student 导出保存，Student 蒸馏继承所选 Teacher 的配置。

## Domain Randomization

物理参数和 delay 在每次环境 reset 时独立采样，并保持到下次 reset；Push 则跨 episode 按仿真时间循环触发。配置集中在 `env_cfg.dr`。Teacher 物理参数课程在训练进度 10%–60% 经 cubic smoothstep 从标称值扩大至完整范围；Student 为 5%–25%。六维特权观测仍依次为延迟、篮球质量、PD 比例、手部摩擦、脚部摩擦、腿部及骨盆 link 质量比例，维度和编码不变。

| 项目 | 标称值 | 完整采样范围 |
| --- | ---: | ---: |
| 最终 PD 目标执行延迟 | 0 个 100 Hz 步 | 0–2 步，观测为 `steps/2` |
| 篮球质量 | USD 默认质量 | 默认值 ±5% |
| 29 个关节的 PD 刚度 | 机器人默认值 | `Kp` 比例 ±10%，`Kd` 比例为其平方根 |
| 双手静、动摩擦 | 0.8 | 0.6–1.0 |
| 双脚静、动摩擦 | 0.9 | 0.6–1.2 |
| 骨盆及八个髋/膝 link 质量 | 各 link 默认质量 | 共用比例 ±10%，惯量同比例变化 |

### 渐变 Delay

动作延迟由 `dr.delay_nominal_steps`（默认 0）和 `dr.delay_max_offset_steps`（默认 2）控制。每次 reset 先在完整范围 `[max(0, nominal-offset), nominal+offset]` 均匀采样整数 `D`，再计算 `x = nominal + strength * (D - nominal)`，以 `x` 的小数部分为概率向上取整，否则向下取整。强度为 0 时全部为 nominal，为 1 时完整范围均匀分布，中间各档延迟的概率连续变化。例如默认范围在强度 0.1 时，约 90% 为 0 步、10% 为 1 步；不会在整数半径阈值处突然引入大批延迟样本。

动作 delay 的 Teacher 课程仍为 30%–70%，Student 为 10%–30%。Student 观测 delay 使用相同采样算法，保留独立的 0%–20% 课程、nominal=1 和完整范围 0–3 步。每个 episode 内 delay 不变。动作延迟特权观测仍为 `(steps-nominal)/max(offset, 1)`；关闭随机化时固定 nominal。要配置标称 1、完整范围 0–2 步，设动作 delay 参数为 `1, 1`；固定零延迟设为 `0, 0`。

### 循环 Push

四类 Push 各自独立配置、计时和采样，可同时发生，训练默认均开启。`dr.<类型>.enabled` 控制开关，`dr.<类型>_interval_range_s` 控制间隔，默认均为连续均匀 `(1.0, 5.0)` 秒。与 HumanX 一样，计时跨 episode 保留：仅环境首次初始化时采样首次间隔，之后每次有效触发机会结束时，从当前仿真时刻重新采样下一间隔；episode reset 只清除活动脉冲和外力，保留下一触发时间以及已到期等待门控的机会。因此新 episode 可以在不足 1 秒时触发，只有环境首次启动需要等待完整首个间隔。计时不受 RSI 起始帧和自适应动作速度影响。这里保留连续采样，HumanX 的 `randint(1,5)` 则采样整数 1、2、3、4 秒。Teacher 三类外力使用各自的 15%–80% smoothstep 课程，速度 Push 为 0%–50%；Student 四者均为 10%–35%。幅度在每次命中时按当时课程强度重新采样。

| 类型 | 门控 | 每次机会命中概率 | 满强度幅度和持续时间 |
| --- | --- | ---: | --- |
| `push` | 逐帧 `push_available` | 0.8 | XYZ 力幅值 0–70 / 0–70 / 0–50 N；各轴力矩 ±5 Nm；0.1 s |
| `global_push` | 无轨迹门控 | 0.8 | 各轴力幅值 0–10 N；各轴力矩 ±2 Nm；0.3 s |
| `object_push` | 逐帧 `obj_push` | 0.6 | 各轴力幅值 0–3 N；无力矩；0.1 s |
| `velocity_push` | 无轨迹门控 | 1.0 | 各轴线速度增量 ±0.5 m/s、角速度增量 ±0.52 rad/s；瞬时叠加 |

力的符号按轴随机。门控关闭时，已到期的机会等待重新允许，不累计补发；门控允许后独立抽取一次概率，未命中也重采样间隔。末帧保持沿用末帧门控。门控关闭、开关禁用或进入 DONE 时立即清除对应的活动脉冲，保留计时。间隔是机会间隔，实际 Push 次数受概率、门控和课程强度影响；短 episode 连续 reset 不会推迟已采样的触发时间。

三类外力仍使用 `<类型>_force_min_n`、`<类型>_force_max_n`、`<类型>_torque_max_nm`、`<类型>_duration_s` 和 `<类型>_probability`。间隔下界不得小于本类脉冲时长，时长必须是策略步长的正整数倍。机器人两类外力分别用 `push_link_weights` / `global_push_link_weights` 指定目标 link 权重，默认 `{"torso_link": 0.5, "pelvis": 0.5}`，每次命中重新选择一个 link；作用于同一 link 的力与力矩相加。

速度 Push 直接向当前世界坐标根 link 速度叠加增量，不覆盖当前速度，也不改写外力缓冲。范围分别由 `velocity_push_linear_max_m_s` 和 `velocity_push_angular_max_rad_s` 控制。四类共享逐策略步驱动事件，各自保留独立状态，外力统一合成写入。

### HumanX 初始状态随机化

`dr.init_state.enabled` 是总开关，每个分量另有独立的 `.enabled`。Teacher 和 Student 均对每次 reset 从训练开始使用满强度，与 HumanX HOI 的固定 `noise_to_initial_level=1.0` 一致，不附加渐变课程。

| 分量 | 幅度配置 | 分布 |
| --- | --- | --- |
| `dof_pos` | `.std=0.1` | 关节位置高斯噪声，rad |
| `dof_vel` | `.std=0.01` | 关节速度高斯噪声，rad/s |
| `root_pos` | `.std=0.02` | 根 XYZ 位置高斯噪声，m |
| `root_rot` | `.max_angle_deg=10.0` | 随机单位轴，角度均匀采样 0–10° |
| `root_vel` | `.std=0.05` | 根 XYZ 线速度高斯噪声，m/s |
| `root_ang_vel` | `.std=0.02` | 根 XYZ 角速度高斯噪声，rad/s |
| `object_pos` | `.half_range_m=0.1` | 球 XY 各均匀采样 ±0.1 m，Z 不变 |

例如 `env_cfg.dr.init_state.root_rot.enabled = False` 只关闭根旋转随机化。扰动加在 RSI/参考帧选定后的模拟器初态副本上，参考轨迹不变；旋转噪声左乘参考姿态，关节位置保留机械限位裁剪。球的姿态和速度、篮筐状态保持参考值。动作和观测历史从最终实际初态初始化。

新配置中的旧 `commands.motion.reset_dof_pos_noise/reset_dof_vel_noise` 默认为 0。只在 `dr.init_state` 缺失或为 `None` 的旧 Teacher 兼容路径使用它们，不与新噪声叠加；关闭 `init_state.enabled` 不会恢复旧噪声。Student 继承 Teacher 的幅度和间隔，同时保留自己的课程与播放开关；旧 Teacher 未记录速度 Push 时默认关闭该项。

播放和评估默认关闭四类 Push 及全部初态随机化。课程日志新增速度 Push 强度和各初态分量的有效开关状态。缓存 schema 仍为 7，逐帧 `push_available` / `obj_push` 数据语义不变。


TensorBoard 除 reward/termination 外，还记录 `Metrics/motion/error/*` 的原始 reference tracking error，以及 `Metrics/motion/control/*` 的 action 变化、PD target 变化、action 饱和率、关节速度和力矩占比。`error/joint/*` 与 `control/joint/*` 提供逐关节误差和动作幅度。这些指标带有物理单位，适合排查 reward 上升但动作抖动的问题。

控制正则集中在 `UnifiedRewardCfg.regularization`：action rate、真实力矩平方、位置/速度/力矩软限位均按关节求和，原始系数为 `0.5 / 1e-5 / 10 / 5 / 5`；总成本乘课程强度后除以 `16.05`。该除数是明确的奖励尺度适配，不表示与 HumanX 奖励严格等价。取消动作幅值、action acceleration、reference 速度误差正则和总正则截断。前 10% 训练强度为 0.1，10%–60% 经 smoothstep 升至 1，软限位同期从 `0.95/0.96/0.95` 收紧至 `0.90/0.95/0.85`；播放与评估使用最终值。课程随实际训练预算和恢复 iteration 接续，不单独乘 dt。PPO 初始标准差为 0.8，entropy 系数为 0.01。

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

# 仅播放 shoot，逗号/句号在该技能内切换 clip
python scripts/rsl_rl/play_interactive.py \
  --checkpoint /path/to/model.pt --skill shoot

# 同时播放 shoot 的所有 clips（当前 100 个环境）
python scripts/rsl_rl/play_interactive.py \
  --checkpoint /path/to/model.pt --skill shoot --all-clips

# 省略 --checkpoint 时，自动选择最新 run 中编号最大的 model_*.pt
python scripts/rsl_rl/play_interactive.py --clip-id 0
```

按键：`V` 显示/隐藏四个 reference body 点、篮球点与篮圈中心，`,` / `.` 切换 clip，`R` 重置。`--skill hold|shoot|stand_to_hold` 限定播放技能，切换 clip 在该技能内循环；此时 `--clip-id` 是技能内索引，从 0 开始。不指定 `--skill` 时沿用全局 clip 索引。普通多环境模式中所有 env 使用同一 clip；`--all-clips` 按所选技能的 clip 数创建 env 并一一对应，不筛选时播放全部 clips（当前 300 个）。日志显示技能名、command ID 与全局 clip ID。交互模式关闭 RSI、push、随机 clip 和 observation corruption，并保留与训练一致的提前失败 termination；adaptive speed 关闭，end 是否执行仍按技能概率抽样。

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

结果默认写到 `outputs/shooting_evaluation.json`。默认均匀评估全部技能的 clips；添加 `--skill shoot` 可仅评估投篮技能。评估关闭 RSI、外力、动力学随机化、reset 关节噪声和观测噪声，并使用确定性策略。

`summary.hit_rate` 只以具有 reference release 事件的可投篮回合为分母，`shot_episodes` 和 `shot_hits` 给出分母与命中数；非投篮技能的命中率为 `null`。`by_skill` 分别报告每个技能的命中率、轨迹误差与 termination。`all_episode_hit_fraction` 保留全部技能回合的命中比例，不能作为投篮准确率；`summary.mean_ball_rmse` 也仍是全部技能的平均，应查看 `by_skill.shoot.mean_ball_rmse` 判断投篮球轨迹误差。训练中的 `Metrics/motion/shot_success` 仍按当次结束的全部技能回合统计，并受 episode 采样权重和训练扰动影响，不能直接与评估命中率比较。

命中判定仅作为指标：球从篮圈上方向下穿过水平面，且水平偏差不超过 0.20 m。

## 测试

不启动 Isaac Sim 的 CPU 测试：

```bash
PYTHONPATH=source/bb_mimic_nt pytest -q tests
```

## Student DAgger

Student 训练和播放任务分别为 `BbMimicNT-G1-Shoot-Student-DAgger-v0` 与 `BbMimicNT-G1-Shoot-Student-Play-v0`。Teacher 使用默认 676 维无噪声、无观测延迟输入；Student 默认使用 471 维显式缩放输入：技能 one-hot command、五帧重力投影、pelvis 角速度、nominal-relative 关节位置、关节速度、裁剪后且执行延迟前的无量纲请求动作，以及 reset 时固定的轨迹篮筐柱坐标 `hoop_pos_cyl`（`[r, theta, z]`）。机器人传感量训练时加入可调高斯噪声与每回合采样一次的 0–3 步延迟，推理固定为标称 1 步。`student_observation.history_length=5` 在 100 Hz 下覆盖 40 ms，输入维度为 `93 * history_length + 3 + command.dimension`。历史、导出和网络输入维度共用这一配置。

Student 输出与 Teacher 相同的 29 维无量纲动作，经共享 clip/scale/nominal 映射形成 PD target。它与 Teacher 共用 PD 参数、动作执行延迟以及全部动力学/外力 DR；动作延迟参数从 Teacher checkpoint 对应的环境配置读取，新默认为标称 0、采样 0–2 步，已有 checkpoint 保留原配置。Student 观测延迟由 `student_observation.delay_nominal_steps=1`、`delay_min_steps=0`、`delay_max_steps=3` 独立控制；只把标称改为 0 时，完整采样范围仍是 0–3 步，推理固定 0 步。三个值须满足 `0 <= min <= nominal <= max`；固定零延迟设为 `0, 0, 0`。观测噪声和延迟范围在前 20% 训练内按 smoothstep 增强，初始延迟固定在标称值；设置 `student_observation.full_strength=True` 可直接使用完整观测扰动。Student 使用独立的 `StudentDomainRandomizationCfg` 控制课程，Teacher checkpoint 只同步扰动范围，不覆盖课程及播放开关，设置对应 `dr.<项>.full_strength=True` 可使用完整范围，reference 始终以 1× 推进。DAgger 在前 10% 完全由 Teacher 执行，10%–30% 通过 smoothstep 概率逐环境混合，从 30% 起完全由 Student 执行；所有访问状态始终由冻结 Teacher 标注。

默认训练 1000 轮，Student 课程按本次总迭代数计算，可在 `bb_mimic_nt_student_env_cfg.py` 独立调整：

| 项目 | 开始扩展 | 达到完整范围 | 默认迭代数 |
| --- | ---: | ---: | ---: |
| RSI（从 0.3 线性衰减） | 0% | 15%（完全退出） | 0–150 |
| 观测噪声 / 观测延迟 | 0% | 20% | 0–200 |
| 质量 / PD / 摩擦 DR | 5% | 25% | 50–250 |
| 动作执行延迟 | 10% | 30% | 100–300 |
| Push | 10% | 35% | 100–350 |
| Student 接管概率 | 10% | 30%（完全接管） | 100–300 |

Student RSI 退出期限按实际 `--max-iterations * --num-steps-per-env * RSI_DECAY_FRACTION` 计算，初始概率默认跟随 `training.py` 的 `RSI_INITIAL_PROBABILITY`，可通过 `env.commands.motion.rsi_initial_probability` 覆盖。`--resume-checkpoint` 从保存的 iteration 接续；Student 的 `--max-iterations` 表示目标总轮数，RSI 不从零重新开始。`Curriculum/rsi/probability` 记录实际概率，播放仍关闭 RSI。

350 轮后，Student 独立执行且全部扰动达到完整范围，剩余约 650 轮用于完整扰动训练。`--max-iterations` 会按比例缩放这些时间点；Teacher 的训练课程保持原配置。播放关闭随机化的设置也不会被加载 Teacher 配置重新开启。

```bash
python scripts/rsl_rl/train_dagger.py \
  --teacher-checkpoint logs/rsl_rl/g1_shoot_teacher_v4/<run>/model_4999.pt \
  --headless
```

Teacher 的网络结构、动作契约与逐关节缩放、DR 范围、技能配置与参考轨迹、末帧保持和初态随机化配置从所选 checkpoint 的 `params/` 读取。Student 使用逐技能 RSI 开关，关闭自适应速度。训练前即写出 `params/env.yaml`、`params/agent.yaml` 和 `params/student_config.json`；每个 `model_*.pt` 都包含独立导出所需的模型结构与部署 metadata。默认每轮收集 48 步并更新 32 次，batch size 跟随 `NUM_ENVS`（当前 8192）；回放容量由 `training.py` 按 `2 * NUM_ENVS * max(32, DAGGER_STEPS_PER_ENV)` 推导，当前为 786432，一半保存最新 FIFO 状态、一半作为会持续更新的全历史 reservoir。8192 个环境时 recent 可容纳完整 48 步；471 维输入加 29 维标签时两分区合计约占 1.57 GB（1.46 GiB）显存，每个 batch 默认 75% 采样最新状态，保留最新闭环状态与历史覆盖。这些是默认配置的联动计算；若通过 CLI/Hydra 单独覆盖环境数或 rollout，请相应检查 `--replay-capacity` 和 `--batch-size`。回放数据常驻训练 device，避免 rollout 期间 CPU/GPU 往返；如需在恢复训练时连同回放集恢复，可添加 `--save-replay`。

Direct action 需要从头训练：旧 residual Teacher、旧弧度 Student、旧 replay 和导出包均被契约校验拒绝，即使观测/动作维数相同。Teacher 归档必须保留 `params/policy_contract.json`；新 Student checkpoint 内嵌契约，导出包使用 deployment v4 和 `joint_action` 输出。DAgger 日志增加每关节目标 MAE、Student/Teacher 目标逐步变化 RMS、目标变化差、失败回合比例与回合步数；动作变化指标排除跨 reset 的跳变，均以 rad 为单位。

```bash
python scripts/rsl_rl/play_student.py \
  --student-weights logs/rsl_rl/g1_shoot_student_dagger_v4/<run>/exported/student_weights.pt \
  --clip-id 0

python scripts/rsl_rl/play_student.py \
  --student-weights logs/rsl_rl/g1_shoot_student_dagger_v4/<run>/model_400.pt \
  --clip-id 0

python scripts/rsl_rl/play_student.py \
  --student-weights /path/to/student_weights.pt --all-clips

# 仅播放 shoot，--clip-id 为该技能内索引，逗号/句号在该技能内切换
python scripts/rsl_rl/play_student.py \
  --student-weights /path/to/student_weights.pt --skill shoot --clip-id 0

# 同时播放单个技能的全部 clips；可替换为 hold 或 stand_to_hold
python scripts/rsl_rl/play_student.py \
  --student-weights /path/to/student_weights.pt --skill shoot --all-clips

python scripts/rsl_rl/evaluate_student.py \
  --teacher-checkpoint /path/to/teacher_model.pt \
  --student-weights /path/to/student_weights.pt --headless

python scripts/rsl_rl/export_student.py \
  --checkpoint /path/to/model_250.pt \
  --output outputs/student_export
```

交互播放支持 `V` reference 点、`,` / `.` 切换 clip、`P` 或空格暂停/恢复、`R` 重启、`--skill`、`--all-clips`、`--ignore-failures` 和 `--no-timeout`。技能选择与 Teacher 一致：`--skill hold|shoot|stand_to_hold` 将播放和按键切换限制在该技能内，`--clip-id` 使用技能内索引；不指定技能时使用全局索引。普通多环境模式所有 env 使用同一 clip；`--skill <name> --all-clips` 按该技能 clip 数创建 env 并一一对应，单独 `--all-clips` 则播放全部技能的所有 clips。日志显示技能名、command ID 与全局 clip ID。

部署交付只需 `student_config.json` 与 `student_policy.onnx`；配置无版本标记，记录 ONNX 输入输出、观测顺序/缩放、关节顺序、nominal pose、安全限位、PD 参数、固定标称延迟和 `export_source`。多技能通过 `skills.command` 定义 one-hot 编码，通过 `skills.definitions` 明确技能名、command ID、可选 motion profiles、初始化及 end 规则；每个 profile 包含篮筐初始柱坐标，切换技能需要重新初始化并重置历史。训练采样权重和 RSI 不进入部署配置。导出目录还保留 `student_weights.pt`、TorchScript、校验向量和 manifest，用于本项目重放与验收。成对评估对相同 clip 分别运行 Teacher 与 Student，输出篮球首次触地相对篮筐的水平落点和最高高度。

Student Play 默认关闭全部初态扰动和域随机化，同一 clip 每次从参考初始状态重启。需要测试扰动时，在 `play_student.py` 命令中添加 `--random-init`（开启训练配置中的全部 HumanX 初态随机化）或 `--domain-randomization`（开启完整强度的物理参数随机化、动作延迟采样、三类外力和独立速度 Push），两者可独立使用或同时开启。观测噪声仍关闭，观测延迟仍使用导出配置的固定值；RSI 和随机 clip 仍关闭。

Student 真机导出流程、技能 command/初始化契约和参考观测构造器见 [部署说明](docs/student_deployment.md)。

### 从 Teacher rollout 导出训练 motion

```bash
python scripts/rsl_rl/record_teacher_motion.py \
  --checkpoint logs/rsl_rl/g1_shoot_teacher_v4/<run>/model_6000.pt \
  --end-duration-s 3.0 \
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
