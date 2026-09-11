# Changelog

## v0.5.0 — 2026-09-11

- 回退所有弹道拟合与 launch-specific 逻辑：Motion cache schema v3 原样保存篮球轨迹，仅使用通用有限差分生成速度；RSI 恢复为全部有效帧均匀采样。
- 删除 `release_reached` 和 `release_velocity_error_mps` TensorBoard / 评估指标，reference observation 直接使用当前原始球速度。
- PPO rollout 从 24 增至 48 步，`gamma` / `lambda` 调整为 `0.995` / `0.975`。
- 为 critic 增加未来 0.25 / 0.50 / 1.00 s 的原始 reference 路点，actor observation 仍保持 473 维；新实验目录为 `g1_shoot_teacher_v3`。
- 保留 reference-relative residual 平滑、通用 object reward / 1 m termination，以及与 rigid-body API 无关的 hoop Xform。

## v0.4.0 — 2026-09-11

- 将 hoop 从 `RigidObject` 解耦为通用 `AssetBaseCfg` / Xform 场景元素；框架不要求也不自动添加 rigid body 或 collider，并继续支持逐环境篮筐位姿。
- Motion cache schema 升级为 v2：按每条 clip 的离手后轨迹拟合重力一致的抛物线和独立出手速度，消除第 80 帧中心差分造成的公共错误速度。
- 出手前 observation 提供 clip-specific launch velocity；object reward 增加线速度分量并提高顶层占比，RSI 仅从可控的离手前帧起步，并记录出手速度误差事件指标。
- 新增 active object position error 大于 1 m 的 termination。
- residual-only 低通从 40 ms 加强到 80 ms；reference feed-forward 完全直通，policy residual 单独限制在最高 2 rad/s，并按 demonstrated velocity scale 和 hold 阶段加强平滑正则。
- PPO 明确声明 policy/critic observation group，消除未来版本兼容警告；修复交互播放第一次按 `V` 未真正隐藏 marker 的状态错误。

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
