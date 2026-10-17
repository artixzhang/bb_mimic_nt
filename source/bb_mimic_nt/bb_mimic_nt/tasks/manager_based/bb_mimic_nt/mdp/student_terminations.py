"""Student-only tracking thresholds; physical safety terminations stay unchanged."""

from bb_mimic_nt.student_curriculum import get_student_termination_curriculum

from .terminations import interaction_tracking_error, reference_dof_error, root_tracking_error


def _scaled_threshold(env, value, command_name):
    motion = env.command_manager.get_term(command_name)
    controller = get_student_termination_curriculum(env, motion)
    return value if controller is None else value * controller.episode_scale


def student_root_tracking_error(env, maximum_error=1.0, command_name="motion"):
    return root_tracking_error(env, _scaled_threshold(env, maximum_error, command_name), command_name)


def student_reference_dof_error(env, maximum_rmse=1.0, command_name="motion"):
    return reference_dof_error(env, _scaled_threshold(env, maximum_rmse, command_name), command_name)


def student_interaction_tracking_error(env, maximum_distance=.35, command_name="motion"):
    # interaction_tracking_error compares one threshold to every required hand.
    threshold = _scaled_threshold(env, maximum_distance, command_name)
    if hasattr(threshold, "ndim") and threshold.ndim:
        threshold = threshold[:, None]
    return interaction_tracking_error(env, threshold, command_name)
