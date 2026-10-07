Teacher 网络输出的是 29 维原始 residual

```Python
bounded_residual = tanh(teacher_output)
target = reference_joint_position + bounded_residual * residual_scale
target = clip(target, safe_lower, safe_upper)
applied_target = apply_action_delay(target)
```

`residual_scale` 是每个关节允许围绕当前 reference 修正的角度幅度, 单位为 rad.

Student 的执行过程为:

```Python
offset = student_policy(observation) # 29维，单位 rad
target = clip(nominal + offset, safe_lower, safe_upper)
applied_target = apply_action_delay(target)
```

学生策略输出的相对 nominal pose 的 offset.