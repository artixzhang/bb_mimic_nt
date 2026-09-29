# Student 导出与真机接入

每次自动训练导出和 `scripts/rsl_rl/export_student.py` 都生成同一套部署包。部署约定由训练环境中的实际 motion cache 和配置生成，不要求手工维护 frame 数、初始姿态或 JSON。旧 checkpoint 缺少这些信息时，导出器从它保存的 Teacher 配置和 motion cache 恢复；依赖缺失时明确报错，不猜动作时长。恢复信息记录在 provenance 中；旧模型未保存 motion 哈希，因此部署前仍需确认该 cache 确实是训练使用的版本。

```bash
PYTHONPATH=source/bb_mimic_nt python scripts/rsl_rl/export_student.py
```

不传参数时，自动选择最新时间戳 run 中编号最大的非空 model_N.pt；没有 checkpoint 的新 run 会跳过。输出自动为 outputs/student_<run-time>。显式 --checkpoint、--output 仍可覆盖；训练未结束时选中的是中间 checkpoint，实际选择会打印并写入 export_source。验证 manifest 后再将整包交给部署程序。历史导出的模型仍兼容；重新导出会补全部署契约。尚在运行的训练不会重新加载 Python 源码，结束后可用上述命令重新导出。

## 包内文件

| 文件 | 用途 |
| --- | --- |
| student_policy.pt / student_policy.onnx | TorchScript / ONNX 推理模型 |
| student_weights.pt | PyTorch 权重 |
| student_config.json | 唯一配置：模型、观测、初始化与时序契约，同时兼容仿真播放 |
| verification_vectors.npz | 输入观测与预期未裁剪动作输出，用于核对部署推理 |
| manifest.json | 文件 SHA256 和实际通过的验证项目 |

配置只保留一份；deployment_contract 是部署时序与初始化规范。training_observation_settings、Teacher 来源和 action_adapter 用于追溯、旧模型重新导出及仿真播放，真机观测构造器不会使用这些训练字段。推理仅使用显式 scales，不更新 running normalization。真机参考实现和仿真共用 student_config.json。

导出自动检查 ONNX 结构，以及 TorchScript、ONNX ReferenceEvaluator 与 PyTorch 的数值一致性。安装 onnxruntime 时还会检查 ONNX 输出；未安装时 manifest 明确标为未验证。部署程序应核对文件哈希，并在实际推理后端运行 verification_vectors，不能把这些离线检查视为真机稳定性验证。

## 控制与观测顺序

参考实现位于 `bb_mimic_nt.student_deployment`，不依赖 Isaac Sim：`StudentPhaseClock` 管理 phase，`StudentObservationState` 构造观测、初始化历史并转换动作。

1. 按 joint_names 映射电机顺序、方向及零位，校准 IMU 到 pelvis/root_link 的旋转。关节位置/速度分别用 rad 和 rad/s，gyro 用 pelvis 坐标系 rad/s；重力输入为 `R.T @ [0,0,-1]` 单位向量，不是 9.81 m/s² 加速度。
2. 用硬件启动控制器进入所选 clip 的 initial_joint_position_rad，球也应处于训练初始条件。nominal_joint_position 只是观测/输出的坐标原点，不是投篮准备姿态。
3. 读取真实首帧传感器和实际生效的 PD target。用首帧填满所有 sensor history；用 `(实际PD target - nominal)` 填满 action history，不全填零。训练中初始 PD target 是安全限位内的第一帧 reference。
4. 捕获篮筐中心相对 pelvis 的位置，并转换到 reset 时的 pelvis 坐标系；随后冻结，READY -> ACTIVE 时也不重新捕获。
5. 每个策略 tick 先采样、构造当前 phase 的输入、推理，再计算 `target = clip(nominal + output, lower, upper)`、提交目标，最后调用 clock.after_policy_tick()。不要再加 tanh、Teacher residual scale 或 reference pose。
6. 下一 tick 的 action history 填实际生效、已经过 clipping/执行延迟的 PD target，而不是最新网络输出或实测关节角。传感历史应用观测延迟，phase/action history 不应用该传感延迟。

```python
import json
import torch
from bb_mimic_nt.student_deployment import StudentObservationState, StudentPhaseClock

cfg = json.load(open('student_config.json'))
model = torch.jit.load('student_policy.pt').eval()
profile = cfg['deployment_contract']['phase']['motion_profiles'][0]
clock = StudentPhaseClock(profile, cfg['policy_hz'],
                          cfg['deployment_contract']['phase']['trained_post_hold_s'])
state = StudentObservationState(cfg)
# adapter 提供校准后的 gravity/gyro/q/qdot、真实 applied_pd_target、initial_hoop_b。
state.reset(gravity, gyro, q, qdot, applied_pd_target, initial_hoop_b)
# 每一 tick:
obs = state.observe(tick, clock.phase, gravity, gyro, q, qdot, applied_pd_target)
with torch.inference_mode():
    target = state.action_target(model(obs))
# 硬件 adapter 提交 target；记录/读取真实生效命令。
clock.after_policy_tick()
# 收到投篮触发时调用 clock.trigger()，不要 state.reset()。
```

硬件已经提供的传感/执行延迟不要重复人工施加。参考 builder 假设接收无延迟原始采样；有实际传感延迟时，通过 sensor_delay_steps 显式选择需要补充的延迟，使总延迟匹配训练分布。部署时须测量延迟、确认 PD gain/torque conventions，并接入 watchdog、fallback、急停与启动控制；这部分取决于实际硬件接口，本仓库不会启用电机。

## phase 与准备控制

每个 clip 自动导出 frame_count、reference_fps 和步长。当前 166 帧、100 Hz reference/100 Hz policy 时，ACTIVE 每 tick 增加 `1/165`，165 tick 后达到 1；POST_HOLD 保持 1，时长取训练配置。更换 clip/策略频率时不得沿用硬编码分母。

READY 可以执行策略并保持 phase=0，传感/动作历史继续更新。触发投篮只启动 phase clock，保持历史和 frozen hoop 向量。这是可实现的运行方式，但当前 PRE_HOLD=0，模型没有经过长期 phase-zero 持球训练。Student 也不观察球位置/接触，不能据此保证球一定拿得住。

可先在仿真中延长准备状态进行独立评估：

```bash
python scripts/rsl_rl/play_student.py --student-weights <model_or_export> --ready-hold-s 3
```

该参数只是播放实验，不修改训练分布。要将 READY 作为长期真机控制状态，应先训练/验证扩展 PRE_HOLD 下的 Teacher，再用相同准备状态和扰动蒸馏 Student；不能仅修改部署 phase。结束 POST_HOLD 后由硬件状态机接管，不能把 DONE 等同于直接断开电机力矩。

部署机器可以独立验收整包与实际推理后端：

```bash
python scripts/verify_student_bundle.py outputs/student_2026-09-29_17-26-30 --backend torchscript
# 使用 ONNX Runtime 的部署端安装相应运行库后：
python scripts/verify_student_bundle.py outputs/student_2026-09-29_17-26-30 --backend onnxruntime
```

phase=0 不是“禁止投篮”的指令。已有模型在持续 phase=0 下投篮或跌倒时，应暂停将它作为 READY 控制器，使用单独的准备控制器；先训练 Teacher 的 PRE_HOLD（例如 1–2 秒）并验证，再蒸馏对应 Student。增加 pre_hold 仅改变播放或导出配置，不会让现有权重学会持球稳定。
