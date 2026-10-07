# Student 导出与部署接口

```bash
PYTHONPATH=source/bb_mimic_nt python scripts/rsl_rl/export_student.py
```

默认选择最新有非空 checkpoint 的时间戳 run，再选择编号最大的 `model_N.pt`，导出到 `outputs/student_<run-time>`。可用 `--checkpoint`、`--student-weights`、`--student-config`、`--output` 覆盖。自动训练导出与独立导出使用同一逻辑。训练进程不会重新加载已编辑的 Python 源码；正在运行的旧进程结束后可用上述命令重新导出。

## 包内文件

| 文件 | 用途 |
| --- | --- |
| student_policy.pt / student_policy.onnx | 完整推理网络，输入 `[batch, observation_dim]`，输出 `[batch, action_dim]` |
| student_weights.pt | 原始 Student 权重；网络结构由配置的 hidden dims 和 activation 重建 |
| student_config.json | 唯一运行配置：关节、观测、动作、phase、初始化，以及 export_source 源文件信息 |
| verification_vectors.npz | 输入观测及预期未经裁剪的网络输出 |
| manifest.json | 文件 SHA256、导出来源、Teacher/motion/URDF 追溯及通过的验证项目 |

新导出使用 `format=bb_mimic_nt_student_v2`。只保留这一个配置版本标记，没有嵌套 schema。旧 v1 配置和训练 checkpoint 仍可读取、播放并重新导出。训练课程、噪声、Teacher residual 参数保留在训练 checkpoint/params 中，不进入部署 JSON。Teacher 路径/哈希和 motion/URDF 哈希保留在 manifest.provenance；整包一起交付。

`student_config.json` 的 `export_source` 记录本次导出输入文件的 `path`（导出时的绝对路径）、`file`（文件名）、`size_bytes` 和 `sha256`；从 checkpoint 导出时还记录 `iteration`。训练结束自动导出使用最后保存的 checkpoint，`--student-weights` 重导出使用实际输入的权重文件，不沿用旧 checkpoint 的哈希。相同信息同步写入 `manifest.provenance.export_source`。路径仅供追溯，部署时无需访问训练机器或复制 checkpoint；导出产物本身的 SHA256 仍由 `manifest.artifacts_sha256` 记录。底层 Python API 未传入源文件时只保留已有来源信息，不推测缺失的路径或哈希。

旧 checkpoint 没有 motion profile 时，导出器从归档 Teacher 配置及 motion cache 恢复；依赖缺失时报错。恢复信息进入 manifest；旧模型未保存 motion 哈希时，仍需确认恢复的 cache 是训练版本。

## JSON 字段与单位

顶层 `policy_hz` 是策略频率；所有 delay_steps 的单位都是策略 tick。周期由 `1 / policy_hz` 得到。`observation_dim`、`action_dim` 是单个样本的维数；`history_length` 是每种历史观测的帧数。

所有 29 元素关节数组都严格遵循 `joint_names` 的顺序：

- `nominal_joint_position`：offset 的坐标原点，rad；不是投篮准备姿态。
- `safe_lower_joint_position` / `safe_upper_joint_position`：最终 position target 的安全边界，rad，已包含训练使用的位置 margin。
- `pd_stiffness`：Kp，N·m/rad；`pd_damping`：Kd，N·m·s/rad。仿真使用 implicit position actuator，硬件须匹配位置控制与 gain 约定。
- `student_hidden_dims` / `activation`：用于从 student_weights.pt 重建网络；TorchScript/ONNX 内已包含结构。

`observation.fields` 是输入的唯一布局定义。每项含 `name`、`start`、`stop`、`unit`、`scale`、`source`。`[start:stop]` 为零基、stop 不含在内的切片；unit 是乘 scale **之前**的物理单位，实际输入为 `raw_value * scale`。配置验证器会检查字段顺序、连续切片、单位、维数和有限 scale。

| 字段 | 单帧维数 | 原始值 |
| --- | --- | --- |
| phase | 1 | frame / (frame_count - 1)，无量纲 |
| projected_gravity_history | 3 | pelvis/root_link 坐标系的重力方向，xyz，无量纲；`R.T @ [0,0,-1]` |
| pelvis_angular_velocity_history | 3 | pelvis/root_link 坐标系 gyro，xyz，rad/s |
| joint_position_offset_history | 29 | 实测 q − nominal，rad |
| joint_velocity_history | 29 | 实测 qdot，rad/s |
| applied_action_offset_history | 29 | 已实际生效的 PD target − nominal，rad |
| hoop_pos_cyl | 3 | 轨迹中的篮筐柱坐标 [r, theta, z]，单位 [m, rad, m]；reset 时读取，随后冻结 |

输入采用 field-major：先拼完某一种 history 的所有帧，再拼下一种 history。每种 history 为 oldest-to-newest，每帧内部是 xyz 或 joint_names 顺序。history_length=5 时维数为 469。只有 sensor histories（gravity/gyro/q/qdot）应用 `observation.sensor_delay_steps`；phase 和 action history 不额外应用传感器延迟。action history 中的 target 本身已经经过 action delay。

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

直接提供与训练轨迹一致的 `hoop_pos_cyl = [r, theta, z]`（单位 [m, rad, m]），不进行 pelvis 坐标变换；reset 时读取一次，READY -> ACTIVE 不重新读取。

## Phase 和运行顺序

`phase.motion_profiles` 包含 clip_id、frame_count、reference_fps 和初始关节/root/ball 条件；root/ball 位置以训练场景原点为参考，root quaternion 使用 WXYZ。硬件启动坐标系须与该 profile 对齐。动作时长为 `(frame_count - 1) / reference_fps`，每 tick phase 增量为 `reference_fps / policy_hz / (frame_count - 1)`，均不重复保存。

`phase.pre_hold_s` / `post_hold_s` 是训练配置的完整 hold 时长，仿真播放采用这些时长；训练是否逐 episode 采样 hold 由训练归档记录。部署参考 clock 在 READY 保持 phase=0，触发后进入 ACTIVE，不自动倒计时触发。准备控制的稳定性需要针对相应模型和 hold 时长实际评估；不从配置时长推断验证已通过。

每个策略 tick 先采样、构造当前 phase 的观测、推理、提交 target，最后推进 phase。ACTIVE 到终帧后保持 phase=1，经过 post_hold_s 后交给硬件控制器。

```python
import json
import torch
from bb_mimic_nt.student_deployment import StudentObservationState, StudentPhaseClock

cfg = json.load(open('student_config.json'))
model = torch.jit.load('student_policy.pt').eval()
profile = cfg['phase']['motion_profiles'][0]
clock = StudentPhaseClock(profile, cfg['policy_hz'], cfg['phase']['post_hold_s'])
state = StudentObservationState(cfg)
state.reset(gravity, gyro, q, qdot, applied_pd_target, hoop_pos_cyl)
# 每一 tick，adapter 提供校准后的传感器和实际已生效的命令：
obs = state.observe(tick, clock.phase, gravity, gyro, q, qdot, applied_pd_target)
with torch.inference_mode():
    target = state.action_target(model(obs))
# 硬件 adapter 提交 target，并记录/读取实际生效命令。
clock.after_policy_tick()
# 投篮触发时 clock.trigger()；不再次 state.reset()。
```

参考 builder 假设接收无延迟原始采样；若硬件采样已有延迟，通过 `sensor_delay_steps` 参数选择需要补充的延迟，使总延迟匹配配置。action.delay_steps 同样是总延迟假设，硬件已有的延迟不要重复施加。joint index/sign/zero、IMU 安装旋转、实际命令延迟及 motor limits 由硬件 adapter 标定；启动、watchdog、fallback 和急停属于硬件控制器。

## 验收

导出检查 ONNX 结构、TorchScript/ONNX ReferenceEvaluator 与 PyTorch 输出一致性；安装 onnxruntime 后还检查该后端。manifest 中明确记录实际通过的项目。

```bash
python scripts/verify_student_bundle.py outputs/student_2026-09-30_13-23-24 --backend torchscript
python scripts/verify_student_bundle.py outputs/student_2026-09-30_13-23-24 --backend onnxruntime
```

校验脚本检查包哈希、新配置接口及实际后端的 verification_vectors。离线验收不等于真机稳定性验证。仿真播放可用 `--ready-hold-s 3` 进行 phase-zero 持球评估，该参数只改变本次播放时长，不改变模型训练分布。
