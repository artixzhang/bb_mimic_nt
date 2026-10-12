# Student 导出与部署接口

```bash
PYTHONPATH=source/bb_mimic_nt python scripts/rsl_rl/export_student.py
```

默认选择最新有非空 checkpoint 的时间戳 run，再选择编号最大的 `model_N.pt`，导出到 `outputs/student_<run-time>`。可用 `--checkpoint`、`--student-weights`、`--student-config`、`--output` 覆盖。自动训练导出与独立导出使用同一逻辑。

## 交付文件

真机部署只需要同一次导出的 **student_policy.onnx + student_config.json**。JSON 不再导出 motion profiles、profile IDs、各 clip 的 root/ball/hoop 状态、参考时长或轨迹路径。部署不需要 Teacher、reference motion、skills.yaml 或 manifest。

| 文件 | 用途 |
| --- | --- |
| student_policy.onnx | 输入 `[batch, observation_dim]`，输出 `[batch, action_dim]` |
| student_config.json | 观测与动作接口、关节参数、技能命令、default_skill 和单个 initial_pose |
| student_policy.pt / student_weights.pt | 本项目 TorchScript 播放和权重重导出 |
| verification_vectors.npz | 覆盖所有技能 command 的输入与预期网络输出 |
| manifest.json | 文件 SHA256、来源追溯和离线验证结果 |

`export_source` 记录导出输入文件的 path、file、size_bytes、sha256，以及 checkpoint 的 iteration。路径只用于追溯，部署无需访问它。Teacher、motion 和 URDF 哈希保存在 manifest.provenance。

从 checkpoint 导出时，仅从其归档元数据读取技能和启动关节角；不打开轨迹文件，也不读取当前 skills.yaml。早期多技能 checkpoint 即使缺少 hoop 初值、参考时长等字段也能导出。旧 phase 输入模型不能通过修改 JSON 转成 command 输入模型，必须使用多技能训练权重。

## 默认技能和启动姿态

- `default_skill`：启动时执行的技能名称，默认 `hold`；没有 hold 时选择配置中的第一个技能。
- `initial_pose`：唯一一组启动关节角，29 个 rad 数值，严格按 `joint_names` 排列。默认取 default_skill 的第一个 clip 的第一帧。
- `nominal_joint_position`：网络输入/输出 offset 的坐标原点，与 initial_pose 含义不同。

硬件启动控制器先把关节移到 initial_pose，再开始持续运行 policy，初始 command 选择 default_skill。移动到启动姿态的轨迹及速度由硬件控制器负责。不要把网络输出直接加到 initial_pose。

可以覆盖默认技能、技能内 clip 索引，或直接指定姿态：

```bash
# 取 shoot 技能第 2 个 clip 的首帧；索引从 0 开始
PYTHONPATH=source/bb_mimic_nt python scripts/rsl_rl/export_student.py \
  --default-skill shoot --initial-clip-index 1

# pose.json 是按 joint_names 顺序排列的 29 元素弧度制 JSON 数组
PYTHONPATH=source/bb_mimic_nt python scripts/rsl_rl/export_student.py \
  --default-skill hold --initial-pose pose.json
```

两个姿态覆盖选项互斥。重导出精简后的权重包时默认保留其 default_skill 和 initial_pose；因为包中已经没有其他 clip 的姿态，更换默认技能时需同时传入 `--initial-pose`，或改用训练 checkpoint 导出。初始姿态只是启动目标，不在运行中自动反复应用。

## 观测接口

`model.input` / `model.output` 定义 ONNX 的 name、dtype 和 shape；首项 `null` 表示动态 batch。顶层 `policy_hz` 是策略频率，delay_steps 的单位均为策略 tick。

`observation.fields` 定义输入布局。每项的 `[start:stop]` 是零基且右开切片；`frame_size` 是每帧元素数，`history_length` 是该字段的帧数；`unit` 是缩放前的单位，输入为 `raw_value * scale`。

| 字段 | 单帧维数 | 原始值 |
| --- | --- | --- |
| skill_command | command_dimension | one_hot(command_id, command_dimension)，由运行端选择 |
| projected_gravity_history | 3 | pelvis/root_link 系重力方向 xyz，`R.T @ [0,0,-1]` |
| pelvis_angular_velocity_history | 3 | pelvis/root_link 系角速度 xyz，rad/s |
| joint_position_offset_history | 29 | 实测 q − nominal，rad |
| joint_velocity_history | 29 | 实测 qdot，rad/s |
| applied_action_offset_history | 29 | 已实际生效的 PD target − nominal，rad |
| hoop_pos_cyl | 3 | 外部篮筐参数 `[r, theta, z]`，单位 `[m, rad, m]` |

历史按 field-major 拼接：先拼完一种观测的所有帧，再拼下一种。每种历史为 oldest-to-newest，每帧按 xyz 或 joint_names 顺序。输入维数为 `93 * history_length + 3 + command_dimension`；5 帧历史、3 维 command 时为 471。输入没有 phase、参考帧或时间进度。

只有 gravity/gyro/q/qdot 的 sensor histories 应用 `observation.sensor_delay_steps`；command、hoop 和 action history 不额外应用传感器延迟。action history 中的 target 本身已经经过 action delay。

hoop 是外部观测参数，不是 ONNX 常量，也不在 JSON 中保存固定坐标。按与训练轨迹 hoop_pos_cyl 相同的坐标约定传入，不额外进行 pelvis 坐标变换。各分量缩放来自字段 scale，例如 `[0.25, 2.0, 0.25]`，theta 的 2.0 是单独的弧度缩放，不再乘 0.25。

没有 running normalization、输入 clip 或部署噪声。IMU 数据需转换到 pelvis/root_link 坐标系；重力输入是方向，不能直接使用 9.81 m/s² 加速度。

## 动作和历史

Student 输出 29 维 nominal-relative position offset，单位 rad。没有额外 tanh、scale、reference 相加或 smoothing。网络外转换为：

```python
target = clip(nominal_joint_position + network_output,
              safe_lower_joint_position, safe_upper_joint_position)
```

所有关节数组遵循 joint_names。pd_stiffness 为 Kp（N·m/rad），pd_damping 为 Kd（N·m·s/rad）；位置限位已包含训练使用的 margin。

转换后匹配 `action.delay_steps` 的执行延迟。下一 tick 的 action history 填实际已生效的 PD target，不填尚未生效的预测或实测关节角。延迟队列初值是实际启动 PD target。

启动时只 reset 一次：传感器历史和延迟 padding 重复第一帧实测 gravity/gyro/q-minus-nominal/qdot；动作历史重复实际启动 target-minus-nominal。qdot 不强制清零。切换技能和更新篮筐参数不 reset 历史，也不停止网络推理。

## 持续运行与多次投篮

`skills.command.labels` 按 one-hot 下标记录技能名，空缺下标为 null；valid_ids 是可用 ID。每个 definition 提供 name、command_id 和 command_one_hot。command ID 是 one-hot 下标，不是列表索引或 clip ID。参考 builder 拒绝未知 ID。

definitions 还保留 skills.yaml 中的 object 和 end 设置，便于对照训练。这些交互掩码、末帧保持时长和触发概率不会在部署端自动触发动作，不作为抓球、释放、切换技能或计时的控制指令。

运行端可以持续执行 `hold → shoot → hold → 更新篮筐 → shoot`。技能切换和篮筐更新在下一次新 tick 的观测中生效，历史连续保留。投篮期间保持篮筐参数不变；何时恢复 hold、何时开始下一次 shoot 由外部控制器决定。

下面只展示观测和推理调用；实际机器人测量、启动动作及 target 提交由硬件 adapter 实现：

```python
import json
import onnxruntime as ort
from bb_mimic_nt.student_deployment import StudentObservationState

cfg = json.load(open('student_config.json'))
model = ort.InferenceSession(cfg['model']['file'], providers=['CPUExecutionProvider'])
command_ids = {s['name']: s['command_id'] for s in cfg['skills']['definitions']}
state = StudentObservationState(cfg)
# 启动控制器先到 cfg['initial_pose']，读取传感器和实际生效 target。
state.reset(gravity, gyro, q, qdot, applied_pd_target, hoop_pos_cyl)

# 第一个 tick 传 None，使用 cfg['default_skill']；之后 None 保持上一 command。
command_id = None
for tick in range(num_ticks):
    # 外部控制器可在 tick 边界修改 command_id 和 hoop_pos_cyl。
    # 如 command_id = command_ids['shoot']；两次投篮之间修改 hoop_pos_cyl。
    # 每个 tick 都先获取新的传感器测量和实际已生效 target。
    obs = state.observe(tick, command_id, gravity, gyro, q, qdot,
                        applied_pd_target, hoop_pos_cyl=hoop_pos_cyl)
    output = model.run([cfg['model']['output']['name']],
                       {cfg['model']['input']['name']: obs.numpy()})[0]
    target = state.action_target(output[0])
    # 提交 target，并在后续 tick 读取实际已生效的命令。
```

`observe(..., hoop_pos_cyl=None)` 保持上次篮筐参数。同一 tick 重复调用返回缓存，不再推进历史或应用新的参数；更新应在下一个 tick 提交。

接口允许持续切换，但现有训练仍按 episode 固定 command 和 hoop，没有连续多次投篮的验证结论。投篮间更新避免了投篮中条件突变，但 hold/shoot 切换后的状态分布和连续执行表现仍需单独验证；本次调整没有修改训练观测或网络权重。

硬件本身已有的观测/动作延迟不应重复施加。传感器延迟可通过 StudentObservationState 的 sensor_delay_steps 参数覆盖，匹配训练使用的总延迟。

## 验收

导出检查 ONNX 结构、TorchScript 和 ONNX ReferenceEvaluator 与 PyTorch 的数值一致性；安装 onnxruntime 后也会验证该后端。manifest 记录实际通过的项目。

```bash
python scripts/verify_student_bundle.py outputs/student_2026-10-12_11-32-11 --backend torchscript
python scripts/verify_student_bundle.py outputs/student_2026-10-12_11-32-11 --backend onnxruntime
```

仿真播放需要参考轨迹，由本地环境的 skills_config 或 skills_path 提供；技能名和 command ID 必须与导出包匹配。仿真使用这些本地轨迹进行 reset 和评估，不把这些需求带入真机配置。
