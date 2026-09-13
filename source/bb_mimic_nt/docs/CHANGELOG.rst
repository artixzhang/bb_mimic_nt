Changelog
---------

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
