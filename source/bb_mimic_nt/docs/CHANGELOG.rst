Changelog
---------

0.5.0 (2026-09-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Removed ballistic fitting and all launch-specific observations, RSI restrictions, TensorBoard metrics, and evaluation metrics; schema v3 preserves raw object trajectories and uses generic finite differences.
* Increased PPO rollouts from 24 to 48 steps and changed gamma/lambda to 0.995/0.975.
* Added raw-reference waypoints at 0.25, 0.50, and 1.00 seconds to the critic only while keeping the actor observation at 473 values.
* Kept the generic residual-action smoothing, object reward and one-metre termination, and physics-schema-independent hoop Xform.

0.4.0 (2026-09-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Decoupled the hoop into a generic ``AssetBaseCfg`` Xform with no required or automatically authored rigid-body/collision APIs.
* Upgraded the motion cache to schema v2 with per-clip, gravity-consistent ballistic fitting and clip-specific release velocities.
* Exposed and logged the launch-velocity target before release, added object velocity reward and a one-metre object tracking termination, and restricted RSI starts to causally controllable pre-release frames.
* Strengthened residual-only filtering and hold-phase smoothness regularization based on demonstrated joint-velocity scales.
* Declared RSL-RL observation groups explicitly and fixed the first interactive marker visibility toggle.

0.3.0 (2026-09-11)
~~~~~~~~~~~~~~~~~~

Changed
^^^^^^^

* Replaced full-range absolute actions with tanh-bounded, reference-relative residual PD targets, a 40 ms residual-only low-pass filter, and data-informed target slew-rate limits.
* Expanded Teacher observations to 473 values with reference velocities and contact state; v0.2.0 checkpoints are intentionally incompatible.
* Added non-saturating velocity/object/relative tracking kernels, stronger control-continuity regularization, smoothed adaptive speed, duration-correct reward logging, and robust reset/limit/hit handling.
* Moved new checkpoints to the ``g1_shoot_teacher_v2`` experiment namespace.

0.2.0 (2026-09-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Implemented the Unitree G1 jump-shot Teacher environment, MotionBatchV1 preprocessing, unified imitation reward, RSI, adaptive reference speed, deterministic playback, and evaluation tooling.
* Reused the ``objects`` scene configurations, corrected the floating hoop center transform, and repaired interactive markers, reset controls, and automatic checkpoint discovery.
* Added a moving basketball reference marker and fixed manual-reset crashes caused by mutable reward state created inside PyTorch inference mode.
* Stopped writing velocity to the kinematic hoop and made interactive playback preserve training failure terminations by default, with an explicit ``--ignore-failures`` reference-inspection mode.
* Added raw tracking-error and control-smoothness diagnostics to TensorBoard for root, joints, links, ball, relative/contact state, actions, PD targets, velocity, and torque.

0.1.0 (2026-09-11)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created an initial template for building an extension or project based on Isaac Lab
