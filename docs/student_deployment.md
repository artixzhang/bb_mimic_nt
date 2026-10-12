# Student 导出与部署接口

```bash
PYTHONPATH=source/bb_mimic_nt python scripts/rsl_rl/export_student.py
```

默认选择最新有非空 checkpoint 的时间戳 run，再选择编号最大的 `model_N.pt`，导出到 `outputs/student_<run-time>`。可用 `--checkpoint`、`--student-weights`、`--student-config`、`--output` 覆盖。自动训练导出与独立导出使用同一逻辑。训练进程不会重新加载已编辑的 Python 源码；正在运行的进程结束后可用上述命令重新导出。

## 交付文件

下游部署只需要同一次导出的 `student_config.json` 和 `student_policy.onnx`。JSON 自包含观测、动作、技能选择和初始状态；不需要读取训练目录、Teacher、motion cache 或 manifest。其余文件用于本项目播放、重导出和离线验收。

| 文件 | 用途 |
| --- | --- |
| student_policy.pt / student_policy.onnx | 完整推理网络，输入 `[batch, observation_dim]`，输出 `[batch, action_dim]` |
| student_weights.pt | 原始 Student 权重；网络结构由配置的 hidden dims 和 activation 重建 |
| student_config.json | 唯一运行配置：关节、观测、动作、技能、初始化，以及 export_source 源文件信息 |
| verification_vectors.npz | 输入观测及预期未经裁剪的网络输出 |
| manifest.json | 文件 SHA256、导出来源、Teacher/motion/URDF 追溯及通过的验证项目 |

部署 JSON 和 manifest 不包含版本标记。输入包含技能 one-hot command，没有 phase 或时间进度输入。训练采样权重、RSI、课程、噪声和 Teacher residual 参数保留在训练 checkpoint/params 中；部署 JSON 保存显式技能选择、初始化 profile 和 end 规则。Teacher/motion/URDF 追溯信息保存在 manifest.provenance。

`student_config.json` 的 `export_source` 记录本次导出输入文件的 `path`（导出时的绝对路径）、`file`（文件名）、`size_bytes` 和 `sha256`；从 checkpoint 导出时还记录 `iteration`。训练结束自动导出使用最后保存的 checkpoint，`--student-weights` 重导出使用实际输入的权重文件，不沿用旧 checkpoint 的哈希。相同信息同步写入 `manifest.provenance.export_source`。路径仅供追溯，部署时无需访问训练机器或复制 checkpoint；导出产物本身的 SHA256 仍由 `manifest.artifacts_sha256` 记录。底层 Python API 未传入源文件时只保留已有来源信息，不推测缺失的路径或哈希。

从 checkpoint 导出时，读取其 `deployment_contract` 保存的技能定义与初始状态，并使用各技能归档的 `motion_file` 提取篮筐柱坐标；技能内索引与参考时长在导出阶段计算。不会读取本地最新 skills.yaml，也没有旧 metadata 的恢复分支。当前多技能 checkpoint 无需重新训练即可导出；导出时需要其技能轨迹源文件。重导出当前产物无需访问 Teacher 或轨迹源文件。

## JSON 字段与单位

`model.file` 为 ONNX 文件名；`model.input` / `model.output` 定义输入输出的 name、dtype 和 shape。shape 的首项 `null` 表示动态 batch。下游按配置中的 name 绑定 ONNX 输入输出。

顶层 `policy_hz` 是策略频率；所有 delay_steps 的单位都是策略 tick。周期由 `1 / policy_hz` 得到。`observation_dim`、`action_dim` 是单个样本的维数；`history_length` 是每种历史观测的帧数。

所有 29 元素关节数组都严格遵循 `joint_names` 的顺序：

- `nominal_joint_position`：offset 的坐标原点，rad；不是投篮准备姿态。
- `safe_lower_joint_position` / `safe_upper_joint_position`：最终 position target 的安全边界，rad，已包含训练使用的位置 margin。
- `pd_stiffness`：Kp，N·m/rad；`pd_damping`：Kd，N·m·s/rad。仿真使用 implicit position actuator，硬件须匹配位置控制与 gain 约定。
- `student_hidden_dims` / `activation`：用于从 student_weights.pt 重建网络；TorchScript/ONNX 内已包含结构。

`observation.fields` 是输入的唯一布局定义。每项含 `name`、`start`、`stop`、`frame_size`、`history_length`、`unit`、`scale`、`source`。`frame_size` 是每帧元素数，字段自己的 `history_length` 对 command 和 hoop 为 1、对历史字段为顶层 history_length。`[start:stop]` 为零基、stop 不含在内的切片；unit 是乘 scale **之前**的物理单位，实际输入为 `raw_value * scale`。下游据此拼接输入；导出不提供配置修复或兼容层。

| 字段 | 单帧维数 | 原始值 |
| --- | --- | --- |
| skill_command | command_dimension | one_hot(command_id, command_dimension)，无量纲，整个 episode 固定 |
| projected_gravity_history | 3 | pelvis/root_link 坐标系的重力方向，xyz，无量纲；`R.T @ [0,0,-1]` |
| pelvis_angular_velocity_history | 3 | pelvis/root_link 坐标系 gyro，xyz，rad/s |
| joint_position_offset_history | 29 | 实测 q − nominal，rad |
| joint_velocity_history | 29 | 实测 qdot，rad/s |
| applied_action_offset_history | 29 | 已实际生效的 PD target − nominal，rad |
| hoop_pos_cyl | 3 | 轨迹中的篮筐柱坐标 [r, theta, z]，单位 [m, rad, m]；reset 时读取，随后冻结 |

输入采用 field-major：先拼完某一种 history 的所有帧，再拼下一种 history。每种 history 为 oldest-to-newest，每帧内部是 xyz 或 joint_names 顺序。history_length=5 时维数为 471（`93 * history_length + 3 + command_dimension`）。只有 sensor histories（gravity/gyro/q/qdot）应用 `observation.sensor_delay_steps`；skill command 和 action history 不额外应用传感器延迟。action history 中的 target 本身已经经过 action delay。

篮筐观测按 `[r_scale, theta_scale, z_scale]` 分量缩放：训练配置 `student_observation.scales.hoop_position=0.25` 用于 r、z，`student_observation.scales.hoop_theta=2.0` 单独用于弧度制 theta（不是额外乘数）。导出配置的 `hoop_pos_cyl` 字段 `scale` 保存为 `[0.25, 2.0, 0.25]`，部署端逐分量相乘。

没有 running normalization、输入 clip 或部署人工噪声。机器人 IMU 需转换到 pelvis/root_link 坐标系；重力输入是方向，不能直接输入 9.81 m/s² 加速度。

## Action 与历史初始化

Student 网络直接输出 29 维 nominal-relative position offset，单位 rad。没有输出 tanh、额外 scale、Teacher reference 相加或 smoothing。执行转换在网络外：

```python
target = clip(nominal_joint_position + network_output,
              safe_lower_joint_position, safe_upper_joint_position)
```

之后按照 `action.delay_steps` 处理总执行延迟。position clip 不根据 effort/velocity limit 计算；这些限制由执行器单独处理。下一 tick 的 action history 填实际已生效的 PD target，不填最新网络预测或实测关节角。延迟队列在启动时应填实际初始 PD target。

初始化时由硬件启动控制器进入所选 motion profile 的姿态和持球条件。名义初始 PD target 可由 `clip(initial_joint_position_rad, lower, upper)` 得到，无需另存重复的 target 或 offset 数组。

随后读取真实首帧 gravity/gyro/q/qdot 和实际初始 PD target；传感器历史及其 delay padding 全部重复这一首帧样本，action history 全部重复 `实际初始target - nominal`。首帧 qdot 也使用实测值，不统一填零。训练 reset 的 q/qdot 可以带扰动，测量 q 与 PD target 不要求相同。

使用所选 profile 的 `initial_hoop_pos_cyl = [r, theta, z]`（单位 [m, rad, m]），不进行 pelvis 坐标变换；每个 episode reset 时读取一次。若下游改变篮筐条件，应传入相同坐标约定下的新值。

## 技能 command 和运行顺序

`skills.command` 定义统一的技能输入：`dimension` 与顶层 `command_dimension` 相同，`encoding=one_hot`、`selection=explicit_command_id`、`lifetime=episode`、`switching=reset_required`。将所选技能的 `command_id` 位置设为 1，其余位置为 0；ID 是 one-hot 下标，不是技能列表索引，也不是 clip ID。ID 可以不连续，未配置的下标保持为 0，不能作为可选技能。当前配置的映射为：

| 技能名 | command_id | one-hot |
| --- | --- | --- |
| hold | 0 | [1, 0, 0] |
| shoot | 1 | [0, 1, 0] |
| stand_to_hold | 2 | [0, 0, 1] |

`skills.definitions` 保存每个技能的 `name`、`command_id`、`motion_profile_ids`、`initialization`、`object`、`end` 和 `motion_file`。下游通过技能名或 command_id 选择 definition，再从该 definition 的 motion_profile_ids 选择启动 profile。`motion_file` 是仿真重放使用的轨迹来源路径，仅供本地播放和追溯；ONNX 部署无需访问它。训练混合权重不用于部署自动调度，也不进入 JSON。仿真播放从这些定义构建均等权重、关闭 RSI 的配置，并按显式选定的 clip 播放。

`skills.motion_profiles` 保存每个 clip 的全局 `clip_id`、技能内 `skill_clip_id`、`command_id`、`frame_count`、`reference_fps`、`reference_duration_s` 和初始关节/root/ball/hoop 条件。`motion_profile_ids` 引用全局 clip_id；同一技能可以有多个 profile，它们使用同一个 command_id。`reference_duration_s = (frame_count - 1) / reference_fps`，支持参考时长为 0 的单帧持姿技能。root/ball 位置以训练场景原点为参考，root quaternion 使用 WXYZ；hoop 使用原轨迹中的柱坐标。

`skills.execution` 描述运行规则。参考运动速度为 1；每 episode 按技能的 `end.trigger_probability` 独立进行 Bernoulli 抽样，选中后从 `end.duration_s=[min,max]` 均匀采样并保持最后参考帧，未选中在参考运动完成后结束。执行计时由外部控制器管理。`object.interaction_enabled` 和 `end.object_interaction_enabled` 是训练中的奖励及终止条件掩码：全局开关为 false 时整个 episode 关闭对应交互条件；全局开启时，end 开关决定末帧保持期间是否继续启用。它们不是球抓取/释放的执行器命令，初始持球条件由 profile.initial_ball_controlled 描述。

部署初始化时选定技能与 profile，并调用 `state.reset()`。每个策略 tick 传入同一个 command_id，构造观测、推理并提交 target。输入没有时间进度；reference 帧数和末帧保持时间用于仿真 episode 管理。部署运行时长由外部控制器管理，不在策略输入中添加计时量。当前训练不串接技能；更换技能时需要重新初始化并 reset 历史。

```python
import json
import onnxruntime as ort
from bb_mimic_nt.student_deployment import StudentObservationState

cfg = json.load(open('student_config.json'))
model = ort.InferenceSession(cfg['model']['file'], providers=['CPUExecutionProvider'])
skill = next(s for s in cfg['skills']['definitions'] if s['name'] == 'shoot')
profile_id = skill['motion_profile_ids'][0]
profile = next(p for p in cfg['skills']['motion_profiles'] if p['clip_id'] == profile_id)
command_id = skill['command_id']
state = StudentObservationState(cfg)
state.reset(gravity, gyro, q, qdot, applied_pd_target, profile['initial_hoop_pos_cyl'])
# 每一 tick，adapter 提供校准后的传感器和实际已生效的命令：
obs = state.observe(tick, command_id, gravity, gyro, q, qdot, applied_pd_target)
output = model.run([cfg['model']['output']['name']],
                   {cfg['model']['input']['name']: obs.numpy()})[0]
target = state.action_target(output[0])
# 硬件 adapter 提交 target，并记录/读取实际生效命令。
```

参考 builder 假设接收无延迟原始采样；若硬件采样已有延迟，通过 `sensor_delay_steps` 参数选择需要补充的延迟，使总延迟匹配配置。action.delay_steps 同样是总延迟假设，硬件已有的延迟不要重复施加。joint index/sign/zero、IMU 安装旋转、实际命令延迟及 motor limits 由硬件 adapter 标定；启动、watchdog、fallback 和急停属于硬件控制器。

## 验收

导出检查 ONNX 结构、TorchScript/ONNX ReferenceEvaluator 与 PyTorch 输出一致性；安装 onnxruntime 后还检查该后端。manifest 中明确记录实际通过的项目。

```bash
python scripts/verify_student_bundle.py outputs/student_2026-09-30_13-23-24 --backend torchscript
python scripts/verify_student_bundle.py outputs/student_2026-09-30_13-23-24 --backend onnxruntime
```

校验脚本检查包哈希、输入输出维数及实际后端的 verification_vectors；验收输入覆盖所有已配置技能的有效 one-hot。离线验收不等于真机稳定性验证。仿真持球评估应选择 hold 技能对应的 clip。
