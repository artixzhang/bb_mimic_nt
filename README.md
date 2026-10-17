# BB Mimic NT：Unitree G1 跳跃投篮 Teacher 与 Student

本项目在 Isaac Lab 2.3 / Isaac Sim 5.1 中实现 G1–篮球–篮筐的 manager-based PPO 模仿环境，以及独立的 Student PPO + BC 训练、评估和部署导出流程。Teacher 按 `skills.yaml` 的 episode 权重混合训练多个独立技能；Student 使用受限部署观测，同时优化任务奖励和冻结 Teacher 的动作监督。

## 环境与数据

- 训练环境：`BbMimicNT-G1-Shoot-v0`
- 确定性播放环境：`BbMimicNT-G1-Shoot-Play-v0`
- 仿真 / 策略频率：200 Hz / 100 Hz（decimation 2）
- 动作：Teacher/Student 共用 29 维无量纲 direct action；输出 clip 到 ±100，目标为 `nominal + action × (0.25 × effort_limit / Kp)`，不做目标角度裁剪。Teacher 的动作 DR 可采样 0–2 个策略步延迟，Student 默认关闭人工延迟。scale 使用固定标称参数，详见 [动作接口](source/bb_mimic_nt/docs/actions.md)
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

默认使用 8192 个环境、每次 rollout 48 步、8000 次 Teacher PPO iteration。Teacher 和 Student 的默认环境数统一由 `source/bb_mimic_nt/bb_mimic_nt/training.py` 中的 `NUM_ENVS` 控制（多 GPU 时为每个进程的环境数）；`PPO_STEPS_PER_ENV` 控制 Teacher rollout，Student 的 `STUDENT_STEPS_PER_ENV` 默认跟随它。Teacher/Student 共用 `RSI_INITIAL_PROBABILITY=0.3` 和 `RSI_DECAY_FRACTION=0.15`，RSI 从训练开始线性衰减，在各自预算的前 15% 完全退出；Teacher 默认退出期限为 57600 策略步（1200 轮）。命令行和 Hydra 参数仍可覆盖默认配置。

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

奖励参数在 `mdp/rewards.py` 的 `UnifiedRewardCfg`。自由球时的 object 和 relative 权重分别由 `object_free_weight`（默认 1.0）和 `relative_free_weight`（默认 0.3）控制，仍保留原有 `release` 数据语义。Student PPO + BC 使用同一任务奖励、技能采样、RSI 和保持逻辑，并加入冻结 Teacher 的动作监督。

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

动作 delay 的 Teacher 课程仍为 30%–70%；Student 默认关闭该 DR，显式开启后课程为 10%–30%。Student 观测 delay 默认 nominal=0、randomize_delay=False，保留可选完整范围 0–3 步及 0%–20% 课程，显式开启后使用相同采样算法。每个 episode 内 delay 不变。动作延迟特权观测仍为 `(steps-nominal)/max(offset, 1)`；关闭随机化时固定 nominal。要配置标称 1、完整范围 0–2 步，设动作 delay 参数为 `1, 1`；固定零延迟设为 `0, 0`。

### 循环 Push

四类 Push 各自独立配置、计时和采样，可同时发生，训练默认均开启。`dr.<类型>.enabled` 控制开关，`dr.<类型>_interval_range_s` 控制间隔，默认均为连续均匀 `(1.0, 5.0)` 秒。与 HumanX 一样，计时跨 episode 保留：仅环境首次初始化时采样首次间隔，之后每次有效触发机会结束时，从当前仿真时刻重新采样下一间隔；episode reset 只清除活动脉冲和外力，保留下一触发时间以及已到期等待门控的机会。因此新 episode 可以在不足 1 秒时触发，只有环境首次启动需要等待完整首个间隔。计时不受 RSI 起始帧和自适应动作速度影响。这里保留连续采样，HumanX 的 `randint(1,5)` 则采样整数 1、2、3、4 秒。Teacher 三类外力使用各自的 15%–80% smoothstep 课程，速度 Push 为 0%–50%；Student 四者均为 10%–35%。幅度在每次命中时按当时课程强度重新采样。

| 类型 | 门控 | 每次机会命中概率 | 满强度幅度和持续时间 |
| --- | --- | ---: | --- |
| `push` | 逐帧 `push_available` | 0.8 | XYZ 力幅值 0–70 / 0–70 / 0–50 N；各轴力矩 ±5 Nm；0.1 s |
| `global_push` | 无轨迹门控 | 0.8 | 各轴力幅值 0–10 N；各轴力矩 ±2 Nm；0.3 s |
| `object_push` | 逐帧 `obj_push` | 0.6 | 各轴力幅值 0–3 N；无力矩；0.1 s |
| `velocity_push` | `push_available` 为 true | 1.0 | 各轴线速度增量 ±0.5 m/s、角速度增量 ±0.52 rad/s；瞬时叠加 |

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

## Student PPO + BC

`train_dagger.py` 和注册任务 `BbMimicNT-G1-Shoot-Student-DAgger-v0` 保留名称，训练算法改为 **PPO + 冻结 Teacher 的动作 MSE**。每个 rollout 都由 Student 的高斯策略采样动作执行；Teacher 只为同一状态提供确定性、裁剪后的 action label。没有 Teacher 接管、动作混合或跨轮 replay。Student critic 使用训练时可用的 Teacher 观测，并从 Teacher critic 及其 normalization 初始化；部署只保留 Student actor 的确定性均值。

Student 保持部署观测限制，不加入球状态或 reference phase。默认输入为 **936 维**：技能 one-hot command、十帧重力投影、pelvis 角速度、nominal-relative 关节位置、关节速度、裁剪后且执行延迟前的请求动作，以及 reset 时固定的轨迹篮筐柱坐标 `hoop_pos_cyl`（`[r, theta, z]`）。输入维度为 `93 * history_length + 3 + command.dimension`；默认 `history_length=10`、command 维度 3，100 Hz 下从最老到最新帧覆盖 90 ms。Teacher 仍使用原来的 676 维输入和五帧历史。增加 Student 历史不改变部署可用的传感量。

Student 输出与 Teacher 相同的 29 维无量纲动作，经共享 clip/scale/nominal 映射形成 PD target。默认 **动作延迟和观测延迟均为零**：动作 DR 开关 `env.dr.delay.enabled=False`，Student 观测 `delay_nominal_steps=0`、`randomize_delay=False`。保留可选延迟实现及范围，显式开启 `env.dr.delay.enabled=True` 或 `env.student_observation.randomize_delay=True` 才启用相应随机延迟。加载 Teacher 时不会覆盖 Student 的标称延迟；物理 DR 范围和 PD/action 契约仍继承 Teacher。导出和播放不加入人工延迟。

默认每轮收集 `8192 * 48 = 393216` 个状态，以 `batch_size=8192` 完整遍历当前 rollout **5 个 epochs，共 240 次优化**。每个 epoch 重新洗牌，尾批也参与训练。总损失包括 PPO clipped surrogate、value loss、entropy 和 `bc_coef * MSE(student_mean, teacher_action)`，默认 `bc_coef=1`、初始 std 0.8、entropy 系数 0.01；奖励使用与 Teacher 同一套 `UnifiedMimicReward`，球飞行统计返回零奖励并继续执行。Student actor 不使用 running normalization。

Actor（含可学习的动作 std）和 Critic 分别使用独立 Adam，并分别裁剪梯度，避免 Critic 梯度大小直接改变 Actor 的裁剪系数。`agent.algorithm.actor_learning_rate` / `critic_learning_rate` 与 `actor_max_grad_norm` / `critic_max_grad_norm` 默认均为 `null`，分别回退到现有 `learning_rate=3e-4` 和 `max_grad_norm=1.0`；原有 Hydra 覆盖仍对两边生效，也可分别设置。两边使用同一 Actor KL 信号，分别调整各自当前学习率；BC、奖励、DR 和 std 配置保持原值。

TensorBoard 的 `StudentPPO/actor_grad_norm`、`StudentPPO/critic_grad_norm` 记录各侧裁剪前梯度范数，`StudentPPO/actor_clip_fraction`、`StudentPPO/critic_clip_fraction` 记录发生裁剪的优化步比例；`Loss/actor_learning_rate`、`Loss/critic_learning_rate` 分别记录两侧学习率。

```bash
python scripts/rsl_rl/train_dagger.py \
  --teacher-checkpoint logs/rsl_rl/g1_shoot_teacher_v4/<run>/model_4999.pt \
  --num-learning-epochs 5 --batch-size 8192 --bc-coef 1.0 \
  --headless
```

默认目标为 1000 轮。`--num-envs`、`--num-steps-per-env`、`--batch-size` 和 `--num-learning-epochs` 决定每轮实际优化次数：`epochs * ceil(num_envs * rollout_steps / batch_size)`。旧 `--replay-capacity`、`--replay-recent-sample-fraction`、`--updates-per-iteration`、`--save-replay` 已移除。

Student 仍保留初态随机化、传感器噪声、质量/PD/摩擦 DR 和 Push；这些范围没有因更换算法而全部关闭。噪声在前 20% 训练内升到全强度，物理 DR 在 5%–25% 扩展，Push 在 10%–35% 扩展；RSI 初始概率 0.3，前 15% 训练内退出，是否启用仍按技能配置。可选动作延迟课程在显式开启后使用 10%–30% 时间段。`--max-iterations` 和实际 rollout 长度控制这些时间点，恢复训练接续已完成轮数。

训练增加 **按技能的 termination curriculum**：root/dof/interaction tracking 阈值按各技能的归一化存活比例调整，初始放宽到原阈值的 1.5 倍，范围 1–2 倍。每个 episode 的计划时长由 reset 后剩余 reference 时长加该次实际采样的 end hold 计算，阈值倍数在该 episode 内固定；完成和失败信息在自动 reset 前保存。每轮最多更新一次，样本不足时保持。fall、机械限位和 non-finite 判据不放宽。课程参数集中在 `student_termination`；这些是待实验验证的初始参数，不代表已验证的成功率目标。`StudentTermination/<skill>/...` 记录归一化存活、完成率、失败原因及实际阈值。评估关闭该课程并恢复原始阈值。

新训练目录为 `logs/rsl_rl/g1_shoot_student_ppo_bc_v1`，checkpoint 顶层格式为 `bb_mimic_nt_student_ppo_bc_v1`，包含 actor/critic、两份独立 Adam 状态及学习率、已完成迭代数、Teacher SHA256、部署 metadata 和 termination curriculum 状态。`--resume-checkpoint` 接受同一 Teacher、兼容观测接口的 PPO+BC checkpoint，包括此前使用共同 Adam 的版本：恢复时按参数拆分旧 Adam 状态，保留各参数的动量和更新步数，并让两侧接续已保存的学习率，不重置为配置初值。正在运行的训练进程不会热更新，使用独立优化器需重启训练或从 checkpoint 恢复。

旧纯 DAgger checkpoint 不能恢复为 PPO+BC 训练，需要重新开始；旧 direct-action Student checkpoint/导出包仍可播放、评估和重导出，其历史维度按归档配置读取。旧 residual action 模型仍不兼容。

```bash
python scripts/rsl_rl/play_student.py \
  --student-weights logs/rsl_rl/g1_shoot_student_ppo_bc_v1/<run>/model_400.pt \
  --skill shoot --all-clips

python scripts/rsl_rl/export_student.py \
  --checkpoint logs/rsl_rl/g1_shoot_student_ppo_bc_v1/<run>/model_400.pt \
  --output outputs/student_export
```

交互播放支持 `V` reference 点、`,` / `.` 切换 clip、`P` 或空格暂停/恢复、`R` 重启、`--skill`、`--all-clips`、`--ignore-failures` 和 `--no-timeout`。选择技能后 `--clip-id` 是技能内索引，否则是全局索引。普通模式所有 env 使用同一 clip；`--all-clips` 为每个选中 clip 创建一个 env。播放默认关闭初态扰动和域随机化、RSI、观测噪声和人工延迟；`--random-init` 和 `--domain-randomization` 可独立开启对应扰动。

### Student 成对评估

原来的飞行评估保持为默认 `--mode flight`，在关闭提前终止的情况下记录同一 clip 的球最高高度和首次触地相对篮筐落点；这类指标不表示机器人闭环存活能力：

```bash
python scripts/rsl_rl/evaluate_student.py \
  --teacher-checkpoint /path/to/teacher_model.pt \
  --student-weights /path/to/student_weights.pt --headless
```

闭环模式必须显式选择技能，对 Teacher 与 Student 使用同一组 clips、原始终止阈值、确定性均值动作、无 DR/RSI/初态噪声。每个 episode 强制固定的末帧保持时间，默认 5 秒，可用 `--end-hold-s` 改变；两种策略的 end trigger 都固定为 1。只统计各环境第一个 episode，不把提前失败后自动重开的回合混入结果。

```bash
python scripts/rsl_rl/evaluate_student.py \
  --mode closed-loop --skill shoot --end-hold-s 5 \
  --teacher-checkpoint /path/to/teacher_model.pt \
  --student-weights /path/to/student_weights.pt \
  --output outputs/student_closed_loop_shoot.json --headless
```

可分别运行 `--skill hold` 和 `--skill stand_to_hold`。`summary.teacher/student` 包含 `completion_fraction`、`failure_fraction`、`mean_survival_fraction`、实际时长和每个终止原因计数；`clips` 保存配对明细。同一步同时出现 reference_finished 和失败时按失败计。归一化存活比例按每个 episode 的计划时长计算，成功完成置为 1；它与完整完成率不同。课程快照在自动 reset 前保存，评估随后立即复制，防止下一个 episode 覆盖。

真机交付只需同次导出的 `student_config.json` 与 `student_policy.onnx`。配置记录观测布局/缩放、关节顺序、nominal、PD 参数、技能命令和单一 initial_pose，不带参考轨迹。运行时技能切换和篮筐参数更新保留连续历史。详细接口、兼容性和验证方法见 [部署说明](docs/student_deployment.md)。

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
