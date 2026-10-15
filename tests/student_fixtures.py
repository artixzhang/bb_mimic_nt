"""Valid self-contained direct-action Student metadata for CPU contract tests."""

import copy

import torch

from bb_mimic_nt.skills import load_skill_config
from bb_mimic_nt.student import student_observation_fields
from bb_mimic_nt.student_deployment import build_deployment_contract


def direct_student_metadata(history=5, nominal=.1):
    scales = [0.1 + index * .01 for index in range(29)]
    robot = dict(joint_names=[f'joint_{i}' for i in range(29)],
                 nominal_joint_position=[nominal] * 29, action_scale=scales, action_clip=100.,
                 joint_position_limits=[[-3., 3.]] * 29, pd_stiffness=[100.] * 29,
                 pd_damping=[3.] * 29, effort_limits=[value * 400. for value in scales],
                 velocity_limits=[30.] * 29, policy_hz=100.)
    contract = dict(copy.deepcopy(robot), version=1, action_semantics='direct_joint_position_v1',
                    observation_semantics='nominal_direct_observation_v1', target_clip=None,
                    teacher_history_length=5, teacher_observation_terms=[
                        dict(name='joint_position', scale=1., history_length=0),
                        dict(name='previous_action_history', scale=1., history_length=5)])
    fields = student_observation_fields(history)
    metadata = dict(copy.deepcopy(robot), command_dimension=3, format='bb_mimic_nt_student_training_v4',
                    observation_dim=sum(size for _, size in fields), action_dim=29, history_length=history,
                    observation_fields=[dict(name=name, size=size) for name, size in fields],
                    action_semantics='direct_joint_position_v1', policy_contract=contract,
                    history_order='oldest_to_newest', student_hidden_dims=[16, 8], activation='elu',
                    default_observation_delay_steps=1, default_action_delay_steps=0,
                    observation_settings=dict(scales=dict(gravity=1., angular_velocity=.25,
                        joint_position=1., joint_velocity=.05, previous_action=1., hoop_position=.25, hoop_theta=2.)))
    metadata['training_observation_settings'] = dict(
        metadata['observation_settings'], history_length=history, delay_nominal_steps=1,
        delay_min_steps=0, delay_max_steps=3, start_fraction=0., end_fraction=.3, noise_strength=1.)
    batch = dict(metadata=dict(dof_names=robot['joint_names'], fps=100., skill_config=load_skill_config(),
                              skill_clip_counts=[1, 2, 1],
                              skill_sources=[dict(source_sha256=f'motion-{i}') for i in range(3)],
                              urdf_sha256='urdf-hash'),
                 lengths=torch.tensor([1, 166, 66, 100]),
                 dof_pos=torch.arange(4.).view(4, 1, 1).expand(4, 166, 29).clone(),
                 root_pos=torch.zeros(4, 166, 3),
                 root_quat=torch.tensor([1., 0., 0., 0.]).repeat(4, 166, 1),
                 object_pos=torch.zeros(4, 166, 3), release=torch.zeros(4, 166),
                 hoop_pos_cyl=torch.tensor([4., -.7, 3.05]).repeat(4, 166, 1))
    metadata['deployment_contract'] = build_deployment_contract(batch)
    return metadata
