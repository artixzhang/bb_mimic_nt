# Teacher / Student 的 Direct Action

两种策略都输出 29 维**无量纲**动作，按 `G1Constants.G1_ACTUATED_JOINT_NAMES` 排列。DAgger 标签为 Teacher 裁剪后的同一动作，无需 reference residual 或弧度偏移转换。

```python
requested_action = clip(network_output, -100.0, 100.0)
action_scale = 0.25 * nominal_effort_limit / nominal_kp
requested_target = nominal_joint_position + action_scale * requested_action
applied_target = apply_action_delay(requested_target)
```

`action_scale` 的单位为 rad / action，使用当前机器人标称力矩限值和 Kp，在初始化时按关节计算并固定。PD domain randomization 不改变它。网络输出不做 `tanh`，最终目标不做关节角度裁剪；机器人自身的力矩/机械限位及现有终止条件继续生效。零动作请求 nominal pose，reference 仍用于观测、初始化和跟踪奖励，但不叠加到执行目标。

实现由 `action_math.py` 的纯张量函数统一维护，仿真由 `DirectJointPositionAction` 管理延迟。配置中的 `action_scale_factor=0.25`、`action_clip=100.0` 是默认值；导出包保存最终逐关节 `action_scale`，部署使用保存的数组。

## 状态与时序

| 名称 | 含义 | 单位 |
| --- | --- | --- |
| `raw_actions` | 网络原始输出 | 无量纲 |
| `requested_actions` | clip 后、scale 前、执行延迟前请求 | 无量纲 |
| `requested_targets` | 当前请求的绝对 PD 目标 | rad |
| `applied_targets` / `processed_actions` | 延迟队列选出的实际 PD 目标 | rad |

Teacher 的动作观测、Teacher/Student 动作历史和 action-rate 正则统一使用 `requested_actions`。Teacher `pd_error = requested_targets - measured_q`；日志另行记录实际执行目标误差。关节位置和参考关节位置均减去 nominal，reference−actual 误差不变。当前根线速度、根角速度、关节速度观测分别缩放为 `2.0`、`0.25`、`0.05`；参考速度保留原有缩放。

reset 后请求动作和动作历史为零，requested target 为 nominal；执行目标和延迟缓存填实际 reset q，包含 RSI 及关节位置噪声。command 写入 reset 状态后再次调用 `reset_to_current_state(env_ids)`。局部 reset 不清除其他环境的历史。

Student 的 `previous_action_history` 没有额外传感器延迟：obs₀ 用零动作，obs₁ 包含 clip(a₀)，依次推进。关节传感器仍按既有噪声和观测延迟设置处理。技能切换不会清空部署历史。

## 控制正则

`UnifiedRewardCfg.regularization` 集中配置五项 HumanX 控制正则：

| 项目 | 每环境代价（按关节求和） | 系数 |
| --- | --- | ---: |
| action rate | `(requested_action - previous_requested_action)²` | 0.5 |
| torque | `applied_torque²`，Nm² | 1e-5 |
| position limit | 超出软位置范围的角度距离 | 10 |
| velocity limit | `clip(abs(qdot) - factor * velocity_limit, 0, 1)` | 5 |
| torque limit | `clip(abs(torque) - factor * effort_limit, 0, 1)` | 5 |

总成本为 `strength * weighted_sum / 16.05`。16.05 是对 HumanX 正向奖励系数和的迁移尺度约定，不表示两套任务奖励严格等价；不做总成本截断，也不额外乘 dt。现有 mimic、腾空惩罚和 termination penalty 保留。动作幅值、二阶动作变化和 reference 关节速度误差不再作为控制正则。

前 10% 训练强度为 0.1，10%–60% 经 smoothstep 升至 1。位置、速度、力矩软限位因子同期分别由 `0.95/0.96/0.95` 收紧至 `0.90/0.95/0.85`。课程使用实际训练预算和恢复 iteration；播放/评估使用最终值。`enabled=False` 表示使用最终参数，不是关闭正则；关闭单项可将其 weight 设为 0。

HumanX 对照来源为 `xmimic/humanoidverse/config/robot/g1/g1_29dof_model16.yaml`、`config/rewards/humanx/hoi.yaml`、`envs/humanx/hoi.py` 和 `envs/legged_base_task/legged_robot_base.py`。本项目保留自身机器人 PD 与 implicit actuator，采用训练进度课程而非 HumanX 的 reset 批次课程。PPO 初始标准差为 0.8，entropy 系数为 0.01。

## 模型兼容性

Teacher run 的 `params/policy_contract.json` 保存解析后的动作映射、PD、物理限值和 Teacher 观测布局。所有 Teacher 加载入口验证契约；Student training v4 内嵌同一契约，deployment v4 保存独立部署所需参数，ONNX 输出名为 `joint_action`。

旧 residual Teacher、旧 nominal-relative rad Student 以及旧 replay 不能直接使用，需要重新训练。输入/输出维数仍可能是 676/29 或 471/29，因此不能以维数判断兼容。新实验目录为 `g1_shoot_teacher_v4` 和 `g1_shoot_student_dagger_v4`。
