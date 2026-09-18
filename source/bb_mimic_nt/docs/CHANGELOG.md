# Changelog

## v0.3.0 — 2026-09-11

- 将整段关节限位的绝对 action 改为 reference-relative residual PD target；policy 输出使用平滑 `tanh` 边界，并只对反馈 residual 使用 40 ms 一阶低通。
- 按每关节 reference 峰值速度和机械速度上限实施 PD target slew-rate limit，并在 motion reset 后同步 action target。
- Teacher observation 从 431 维扩展到 473 维，增加 root/DoF/篮球 reference velocity 与 reference contact。
- velocity、object 和 relative tracking 使用远距离仍有分辨率的 rational kernel；显著加强 action/target 连续性正则，并改为 reference-relative joint velocity cost。
- adaptive speed 使用 tracking-error EMA 且最多每 0.1 s 切档，避免 reference 时钟逐帧抖动。
- Reward TensorBoard 日志改为按实际 episode 时长求均值；硬关节限位增加 PhysX 数值容差，RSI 飞行段不再产生虚假的投篮命中。
- 新 action/observation 与旧 checkpoint 不兼容，实验目录改为 `g1_shoot_teacher_v2`。

## v0.2.0 — 2026-09-11

- 将 cart-pole 模板替换为 G1–篮球–kinematic 篮筐 manager-based Teacher 环境。
- 新增 100 条变长轨迹的 MotionBatchV1 校验、Pinocchio FK、速度差分、padding 与哈希缓存。
- 新增 29 维非对称绝对 PD action、431 维 Teacher observation、RSI、hold、自适应 reference speed 和离手锁速。
- 新增单一有状态 `UnifiedMimicReward`、完整 failure termination、投篮命中与跟踪 RMSE 指标。
- 注册 `BbMimicNT-G1-Shoot-v0` 与 `BbMimicNT-G1-Shoot-Play-v0`。
- 新增轨迹预处理、交互播放和批量投篮评估脚本。
- 新增 CPU 单元测试；Isaac Sim/PPO 冒烟保留为 GPU 驱动可用时的运行验收。
- 环境改为复用 `objects/` 资产配置，修正 floating hoop 的 180° 姿态和篮圈中心局部偏移。
- 修复交互播放 reference marker 与 `R` reset，并支持省略参数时自动加载最新 checkpoint。
- 将 policy 前向与可变环境状态更新分离，修复手动 reset 时的 PyTorch inference-tensor 崩溃。
- 交互 marker 增加篮球参考点并逐策略帧显式刷新；默认保留训练用失败 termination，另提供 `--ignore-failures` 完整检查 reference。
- 修复 reset 时向 kinematic 篮筐写入线速度和角速度导致的 PhysX 报错。
- TensorBoard 增加 root、DoF、link、篮球、手球相对位置和 contact 的原始 tracking error，以及 action/PD target/关节速度/力矩抖动诊断指标。

## 26-09-15_23-00

- `rsl_rl_ppo_cfg.py` 降低噪声探索奖励
    - init_noise_std: 0.8 -> 0.4
    - entropy_coef: 0.005 -> 0.0001
- `rewards.py` 加大死亡惩罚, 加大正则化, 压低噪声
    - regularization_clip: 0.35 -> 1.0
    - termination_penalty: 5.0 -> 50.0
- `bb_mimic_nt_env_cfg.py` 提升仿真速度
    - self.decimation: 5 -> 2
    - self.sim.dt: 1.0/500.0 -> 1.0/200.0
- `actions.py` 提升residual权限, 加大residual作用范围
    - residual_scale_fraction: 0.20 -> 0.50
    - maximum_residual_scale: 0.50 -> 0.80

## 26-09-17_14-00

- `push_linear_velocity` 0.5 -> 1.0
- `push_yaw_velocity` 0.3 -> 0.6
- `PPO_STEPS_PER_ENV` 24 -> 48
- `root_position_reward` 奖励函数模型变化, 重点奖励高度, 水平方向提供弱拉力
- `object_reward` 区分 object 速度方向奖励与速度模长奖励.
- `action_magnitude_weight` 0.005 -> 0.05
- `action_rate_weight` 0.05 -> 0.10
- `root_tracking_error` `max_error` 0.75 -> 1.5

## 26-09-18_14-17
- remove residual action filter. policy 的 `tanh` 输出直接用于 reference-relative PD target.
- remove 关节目标的历史速度限制, 以及独立的 `processed_actions` 状态, `desired_actions` 经位置限制后直接作为 PD target.
- add root-local link 旋转追踪. 加入 `torso` 管控; 增加对应数据处理缓存.
- `PPO_STEPS_PER_ENV` 24 -> 96 显著提升 reward.
- `init_noise_std` 0.4 -> 0.25 降低初始噪声.

## 26-09-18_16-13
- 修改 G1 PD 参数, 验证阻尼比, 相位裕度.