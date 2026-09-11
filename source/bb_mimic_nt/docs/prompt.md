## 概述
我要完成的任务是 Unitree G1 定点跳跃投篮模仿强化学习, 目前只支持投篮这一个动作.

我提供给你了 HumanX 论文供你参考, 并不需要严格按照它的实现方式. 我想实现的框架不需要那么重型, 只需要比较精简高效好用的功能, 具体功能为:
- 提供一个机器人投篮的运动数据 batch 作为 reference, 让机器人参考轨迹中的动作, 完成投篮. 投篮轨迹中包含篮球的运动轨迹, 让篮球轨迹尽可能接近期望的轨迹.
- 轨迹 batch 中包含多条数据, 需要混合训练, 这些数据目前是从同一个 robot 动作基础上, 通过直接改变篮球的轨迹增强得到的. 目标是让机器人能够对任意位置的篮筐都能投进.
- 框架需要有数据处理能力, 要将轨迹数据实现计算打包, 例如使用差分法计算速度, 使用 fk 计算 link_body_pos 等关键数据, 防止 RL 过程中现场计算.
- 首先聚焦于 Teacher policy, 观测可以不局限于现实可行.

## 轨迹解析

轨迹的结构可以参考 `source/bb_mimic_nt/assets/trajectory/shoot.pkl` .

使用 joblib 解析轨迹:

```Python
Python 3.10.20 (main, Jun 11 2026, 15:17:37) [GCC 14.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import joblib
>>> data = joblib.load('shoot_dataset.pkl')
>>> len(data)
100
>>> data[0].keys()
dict_keys(['root_pos', 'root_rot', 'dof', 'obj_pos', 'obj_rot', 'contact', 'contact_names', 'fps', 'dof_names', 'hoop_pos_w'])
```

提供内聚的数据处理脚本, 为 observation 和 command 的计算所需要的项目提供数据, 例如 fk body 位置, 差分速度等. 计算在训练开始前一次性完成, 避免训练过程中计算损失时间.

## reward的项目以及分组方式.

最终的全局奖励设计为:

$$
r_t = r_t^{mimic} - r_t^{reg} - r_t^{term}
$$

其中, $r_t^{mimic}$ 是总分为 1 的正项奖励, $r_t^{reg}$ 是很小的正则化惩罚, $r_t^{term}$ 是很大的结束惩罚.

每个 reward 的计算要有单独的一个函数, 然后在总的reward计算中调用, 使用 unified reward 设计, 不使用系统的 reward 叠加. 使用层级结构, 为各自对应参数配置预留统一修改的位置. reward 的各个子项目使用在 tensorboard 中按层级打印监控.

### $r_t^{mimic}$ 项奖励

$r_t^{mimic}$ 是根据内部项权重自动分配系数, 是总分保持为 1 的奖励项. 一共分为 4 个大项, 每个大项中拆分若干小项(不要整体合并连加, 而是需要按组), 4 个大项使用连加的形式:

$$
r_t^{mimic} = \alpha_{body}r_t^{body} + \alpha_{obj}r_t^{obj} + \alpha_{rel}r_t^{rel} + \alpha_{c}r_t^{cg}
$$

其中各项各自的理论总分均为 1, 各项分别含义为:

$r_t^{body} = \beta_{root}r^{root} + \beta_{joint}r^{joint} + \beta_{link}r^{link}$
- $r^{root} = w_{root\_p}r^{root\_p} + w_{root\_r}r^{root\_r} + w_{root\_pv}r^{root\_pv} + w_{root\_rv}r^{root\_rv}$ </br> 分别是根节点在世界坐标系下的位置, 旋转, 线速度, 角速度. 每一项使用高斯核, 并有独立的参数, 如: $r^{root\_p} = \exp(-\sigma_{root\_p}||p_{root} - \hat{p}_{root}||^2)$ . $r^{root}$ 的总分也为 1. 
- $r^{joint} = w_{joint\_p}r^{joint\_p} + w_{joint\_pv}r^{joint\_pv}$ </br> 分别是关节的位置, 以及关节速度. 每一项使用高斯核, 并有独立的参数, 如: $r^{joint\_p} = \exp(-\sigma_{joint\_p}||p_{joint} - \hat{p}_{joint}||^2)$ . $r^{joint}$ 总分为 1. 所有关节平分权重.
- $r^{link} = w_{link\_p}r^{link\_p} + w_{link\_r}r^{link\_r}$ </br> 约束 link 的位姿, 使用 root-local 参考系. 提供一个 list, 只有在 list 中的 link 才计算 reward, 其余 link 忽略, 例如只追踪 EE 的 root-local link pos, rot reward. 使用高斯核函数, 拥有独立的参数. $r^{link}$ 总分也为 1. 所有追踪的 link 平分权重.

$r_t^{obj} = \beta_{obj\_p}r^{obj\_p} + \beta_{obj\_r}r^{obj\_r}$ 分别衡量 object 的位置和旋转, $r_t^{obj}$ 的理论总分为 1. 使用高斯核函数, 并有独立的参数. 旋转项系数先设置为 0, 即对于篮球任务先不追踪球的旋转. 注意 object 的门控逻辑, 当轨迹结束后, 虽然环境继续 step, 但是 object 奖励项应该关闭. object 项的 reward 关闭后, 总奖励按其余项的系数 $\alpha$ 自动重新分配, 使总分依然维持在 1. object 项奖励参考的坐标系是 global 系, 而不是身体的 root-loacl 坐标系.

$r_t^{rel} = \beta_{rel\_p}r^{rel\_p} + \beta_{rel\_r}r^{rel\_r}$ 衡量物体和指定的 link 或 prim 的相对位置和旋转, $r_t^{rel}$ 的理论总分为 1. 使用高斯核函数, 并有独立的参数. 旋转项系数先设置为 0, 即对于篮球任务先不追踪球的旋转. 维护一个 list, 追踪在 list 内的 link 和 object 的相对奖励. 所有追踪的 link-object 平分权重. 注意 rel 项的门控逻辑, 当轨迹结束后, 环境继续 step, 但 rel 项奖励应该关闭, 关闭后总奖励按其余项的系数 $\alpha$ 自动重新分配, 使总分依然维持在 1.

$r_t^{cg}$ 衡量 link 和 object 以及地面的接触. 根据动作数据中的接触图, 得到标准接触向量 $\boldsymbol{s}_t^{cg} \in \{0,1\}^{J}$ , 其中 $J$ 是预期控制的 contact edge 的数量, 在本项目中, 只控制 '左手-球', '右手-球', '左脚-地面', '右脚-地面' 4 个接触边, 并以固定编码的形式记录在动作轨迹中. 计算公式为 $r_t^{cg} = \exp(-\sum_{j=1}^{J}\beta_j\cdot \boldsymbol{e}_t^{cg}[j])$ , 其中, $\boldsymbol{e} = |\boldsymbol{s}_t^{cg} - \hat{\boldsymbol{s}}_t^{cg}|$ 为实际和参考向量的差值.

### $r_t^{reg}$ 项惩罚

正则化惩罚使用连加的形式, 其计算公式为:

$$
r_t^{reg} = \min\left(c_{\max}, \sum_i w_i\cdot c_i\right)
$$

有专门的一个参数, 设置 clip threshold. 正则化项包括:

- $c^{action\_rate}$ 动作率
- $c^{torque}$ 扭矩
- $c^{limit}$ 关节限位
- $c^{joint\_pv}$ 关节速度

### termination

termination项目包括:
- fall/tilt: 倾角过大或高度过低
- 机械极限: 任意关节穿透限位
- 数值溢出: 当数值存在 `NaN/Inf`
- root 追踪: 当 root 偏移过大
- reference 追踪: 当 dof 与 reference 区别过大
- interactive 追踪: 当contact graph 要求手部与object接触, 但实际二者距离超出阈值. 需要做正确的距离解析. 在 USD 中, 预留了对应的 prim: `left_hand/left_hand/center_of_ball` 以及 `right_hand/right_hand/center_of_ball` . 阈值适当放大, 防止 termination 过于严格以及 contact graph 本身的物理不可达. 只约束 '手-球' 接触, 不约束 '脚-地' 接触.

## Action 格式

没有固定的 action scale. 读取机器人关节配置, 使用 nominal + 绝对 PD 位置的形式控制关节. 将整个关节的可运动范围通过 action 输出的 (-1,1) 和对于每个关节的 action scale 进行生成.

## 资产手部处理

关闭 fixed_joint 的 rigid body 的合并, 防止 contact 无法正确解析. reference 观测不要观测 wrist, 要观测hand.

## 末帧维持

在动作 reference 结束后, 不要立刻开始下一 episode, 而是按最后一帧维持站立. 

## RSI 调整

通过 Curriculum Learning, 逐渐改变 RSI 的初始状态比例. 只在训练前期加入 80% RSI, 随训练进行逐渐降低 RSI 的概率. 训练中后期要求均从完整轨迹开始进行.

## DR

加入推力域随机化, 对于单帧循环以及最后一帧维持的时候加入随机推力, 使用门控, 对于动作正在进行中时, 关闭随机推力的域随机化.

## Adaptive speed

加入自适应参考速度, 由框架根据实时状态, 动态调整参考轨迹的播放速度. 保持前向不可回退, 自适应速度在一定速度范围内有几档可选, 防止轨迹不可达.

## observation

初步设计的 observation 项, 包含如下项:
- gravity_vec_b [3]
- root_pos_z_w [1]
- root_quat_w [4]
- root_lin_vel_b [3]
- root_ang_vel_b [3]
- dof_pos [29]
- dof_pos_vel [29]
- link_pos_b [3*4] # fk 追踪的 link (EE)
- obj_pos_b [3]
- obj_lin_vel_b [3]
- hoop_pos_b [3]
- prev_action [29]
- pd_error [29]

- delta_root_pos_b [3]
- delta_root_quat_b [4]
- delta_dof_pos [29]
- delta_link_pos_b [3*4]
- delta_obj_pos_w [3]
- ref_dof_pos [29]
- ref_link_pos_b [3*2]
- ref_obj_pos_w [3]

- history_gravity_vec_b [3*3]
- history_dof_pos [29*3]
- history_action [29*3]

- phase [1]

可以根据项目需求调整, 不一定严格按照上述项目.

## interactive play

不要修改官方模板中的play, 或者其他标准文件, 而是新建一个. `play_interactive.py`要求能够显示手脚以及篮筐中心的参考点, 按 `V` 显示或隐藏. 能够接受参数能同时显示轨迹中所有目标篮筐位置的clip在一个场景中, env熟练根据轨迹数据数量自动调整. 或者指定env数量, 然后按 `,` 和 `.` 切换上一个或下一个目标篮筐位置. 按 `R` 重置环境. play 的时候关闭所有噪声或可能影响结果的机制.

## 机器人于其余资产

资产的配置会防止在 `source/bb_mimic_nt/assets` 目录下. 也会有对应的 Unitree G1 机器人参数设置.

## 版本记录

在 doc 目录中添加一个版本记录的 md 文档, 在每次修改后, 归纳记录每次的修改.

## 代码风格

代码简洁易懂, 功能完善, 不要过度堆砌

## README

说明功能, 设计, 以及使用方法.
