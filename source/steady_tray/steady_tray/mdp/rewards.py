# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Dict, List, Sequence

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat, quat_box_minus, quat_error_magnitude, wrap_to_pi, euler_xyz_from_quat
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

@torch.jit.script
def track_lin_vel_xy_yaw_frame_exp(
    root_quat_w: torch.Tensor, root_lin_vel_w: torch.Tensor, vel_command: torch.Tensor, sigma: float, weight: float
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    vel_yaw = quat_apply_inverse(yaw_quat(root_quat_w), root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(vel_command[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / sigma) * weight


@torch.jit.script
def track_ang_vel_z_world_exp(
    root_ang_vel_w: torch.Tensor, vel_command: torch.Tensor, sigma: float, weight: float
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    ang_vel_error = torch.square(vel_command[:, 2] - root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / sigma) * weight


@torch.jit.script
def track_lin_vel_xy_base_exp(root_lin_vel_b: torch.Tensor, vel_command: torch.Tensor, weight: float, sigma: float) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) base frame frame using exponential kernel."""
    
    lin_vel_error = torch.sum(
        torch.square(vel_command[:, :2] - root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / sigma) * weight


@torch.jit.script
def track_ang_vel_z_base_exp(root_ang_vel_b: torch.Tensor, vel_command: torch.Tensor, weight: float, sigma: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) base frame using exponential kernel."""

    ang_vel_error = torch.square(vel_command[:, 2] - root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / sigma) * weight


@torch.jit.script
def track_lin_vel_x_base_exp(root_lin_vel_b: torch.Tensor, vel_command: torch.Tensor, weight: float, sigma: float) -> torch.Tensor:
    """Reward tracking of linear velocity commands (x-axis) base frame using exponential kernel."""
    lin_vel_error = torch.square(vel_command[:, 0] - root_lin_vel_b[:, 0])
    return torch.exp(-lin_vel_error / sigma) * weight

@torch.jit.script
def track_lin_vel_y_base_exp(root_lin_vel_b: torch.Tensor, vel_command: torch.Tensor, weight: float, sigma: float) -> torch.Tensor:
    """Reward tracking of linear velocity commands (y-axis) base frame using exponential kernel."""
    lin_vel_error = torch.square(vel_command[:, 1] - root_lin_vel_b[:, 1])
    return torch.exp(-lin_vel_error / sigma) * weight


@torch.jit.script
def lin_vel_z_l2(root_lin_vel_b: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""

    return torch.square(root_lin_vel_b[:, 2]) * weight


@torch.jit.script
def ang_vel_xy_l2(root_ang_vel_b: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""

    return torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1) * weight

@torch.jit.script
def joint_tracking_exp(joint_pos: torch.Tensor, joint_idx: List[int], joint_pos_command: torch.Tensor, weight: float, sigma: float) -> torch.Tensor:
    """Reward tracking of joint positions using exponential kernel."""
    joint_pos_error = torch.sum(torch.square(joint_pos[:, joint_idx] - joint_pos_command), dim=1)
    return torch.exp(-joint_pos_error / sigma) * weight

@torch.jit.script
def flat_orientation_l2(projected_gravity_b: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1) * weight


@torch.jit.script
def action_rate_l2(action: torch.Tensor, prev_action: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""

    return torch.sum(torch.square(action - prev_action), dim=1) * weight


@torch.jit.script
def joint_accel_l2(joint_accel: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize the rate of change of the joint accelerations using L2 squared kernel."""

    return torch.sum(torch.square(joint_accel[:, joint_idx]), dim=1) * weight


@torch.jit.script
def joint_vel_l2(joint_vel: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize the rate of change of the joint velocities using L2 squared kernel."""

    return torch.sum(torch.square(joint_vel[:, joint_idx]), dim=1) * weight


@torch.jit.script
def joint_torque_l2(joint_torque: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize the rate of change of the joint torques using L2 squared kernel."""

    return torch.sum(torch.square(joint_torque[:, joint_idx]), dim=1) * weight

@torch.jit.script
def joint_pos_l2(joint_pos: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    return torch.sum(torch.square(joint_pos[:, joint_idx]), dim=1) * weight

@torch.jit.script
def joint_deviation_l1(joint_pos: torch.Tensor, default_joint_pos: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # compute out of limits constraints
    angle = joint_pos[:, joint_idx] - default_joint_pos[:, joint_idx]
    return torch.sum(torch.abs(angle), dim=1) * weight


@torch.jit.script
def joint_deviation_exp(joint_pos: torch.Tensor, joint_idx: List[int], joint_pos_command: torch.Tensor, weight: float, sigma: float) -> torch.Tensor:
    """Reward tracking of joint positions using exponential kernel."""
    joint_pos_error = torch.sum(torch.square(joint_pos[:, joint_idx] - joint_pos_command), dim=1)
    return torch.exp(-joint_pos_error / sigma) * weight


@torch.jit.script
def negative_knee_joint(joint_pos: torch.Tensor, joint_idx: List[int], min_threshold: float, weight: float) -> torch.Tensor:
    """Penalize negative knee joint angles (lower body only)."""
    return torch.sum((joint_pos[:, joint_idx] < min_threshold).float(), dim=1) * weight


@torch.jit.script
def alive_reward(terminated: torch.Tensor, weight: float) -> torch.Tensor:
    """Reward alive."""
    return ~terminated * weight


@torch.jit.script
def joint_pos_limits(joint_pos: torch.Tensor, soft_joint_pos_limits: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # compute out of limits constraints
    out_of_limits = -(
        joint_pos[:, joint_idx] - soft_joint_pos_limits[:, joint_idx, 0]
    ).clip(max=0.0)
    out_of_limits += (
        joint_pos[:, joint_idx] - soft_joint_pos_limits[:, joint_idx, 1]
    ).clip(min=0.0)

    return torch.sum(out_of_limits, dim=1) * weight


@torch.jit.script
def joint_vel_limits(joint_vel: torch.Tensor, soft_joint_vel_limits: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize joint velocities if they cross the soft limits.
    
    This is computed as a sum of the absolute value of the difference between the joint velocity and the soft limits.
    """
    # violation
    violation = (torch.abs(joint_vel[:, joint_idx]) - soft_joint_vel_limits[:, joint_idx]).clip(min=0.0, max=1.0)
    # compute out of limits constraints
    out_of_limits = torch.sum(violation, dim=1)
    return out_of_limits * weight


@torch.jit.script
def joint_torque_limits(joint_torque: torch.Tensor, effort_limits: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize joint torques if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint torque and the soft limits.
    """
    # violation
    violation = (torch.abs(joint_torque[:, joint_idx]) - effort_limits[:, joint_idx]).clip(min=0.0, max=1.0)
    # compute out of limits constraints
    out_of_limits = torch.sum(violation, dim=1)
    return out_of_limits * weight


@torch.jit.script
def termination_penalty(terminated: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize termination."""

    return terminated * weight

def base_height(body_pos_w: torch.Tensor, body_idx: int, height_scanner: RayCaster | None, target_height: float, weight: float) -> torch.Tensor:
    """Penalize base height."""
    if height_scanner is not None:
        adjusted_target_height = target_height + torch.mean(height_scanner.data.ray_hits_w[..., 2], dim=1)
    else:
        adjusted_target_height = target_height
    base_height = body_pos_w[:, body_idx, 2]
    height_error = torch.square(base_height - adjusted_target_height)
    return height_error * weight


def feet_air_time(vel_command: torch.Tensor, contact_sensor: ContactSensor, feet_body_indexes: List[int], step_dt: float, threshold: float, weight: float) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    first_contact = contact_sensor.compute_first_contact(step_dt)[:, feet_body_indexes]
    last_air_time = contact_sensor.data.last_air_time[:, feet_body_indexes]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(vel_command[:, :2], dim=1) > 0.1
    return reward * weight


def feet_air_time_positive_biped(vel_command: torch.Tensor, contact_sensor: ContactSensor, feet_body_indexes: List[int], threshold: float, weight: float) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """

    air_time = contact_sensor.data.current_air_time[:, feet_body_indexes]
    contact_time = contact_sensor.data.current_contact_time[:, feet_body_indexes]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(vel_command[:, :2], dim=1) > 0.1
    return reward * weight


def feet_slide(body_lin_vel_w: torch.Tensor, contact_sensor: ContactSensor, feet_body_indexes: List[int], weight: float) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contacts = contact_sensor.data.net_forces_w_history[:, :, feet_body_indexes, :].norm(dim=-1).max(dim=1)[0] > 1.0
    feet_vel = body_lin_vel_w[:, feet_body_indexes, :2]
    reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)
    return reward * weight


def feet_swing_height(body_pos_w: torch.Tensor, contact_sensor: ContactSensor, feet_body_indexes: List[int], weight: float, target_height=0.08) -> torch.Tensor:
    """Reward Feet Swing Height"""

    # get feet contact
    contact = contact_sensor.data.net_forces_w[:, feet_body_indexes, :3].norm(dim=-1) > 1.0

    # get feet swing height
    feet_pos_z = body_pos_w[:, feet_body_indexes, 2]
    pos_error = torch.square(feet_pos_z - target_height) * (~contact)
    return torch.sum(pos_error, dim=1) * weight

@torch.jit.script
def feet_orientation(body_rot_w: torch.Tensor, gravity_vec_w: torch.Tensor, feet_body_indexes: List[int], weight: float) -> torch.Tensor:
    """Reward Feet Orientation"""
    # left foot
    left_quat_w = body_rot_w[:, feet_body_indexes[0], :]
    left_foot_orientation = quat_apply_inverse(left_quat_w, gravity_vec_w)
    # right foot
    right_quat_w = body_rot_w[:, feet_body_indexes[1], :]
    right_foot_orientation = quat_apply_inverse(right_quat_w, gravity_vec_w)
    # compute orientation error
    reward = torch.sum(torch.square(left_foot_orientation[:, :2]), dim=1)**0.5 + torch.sum(torch.square(right_foot_orientation[:, :2]), dim=1)**0.5 
    # reward
    return reward * weight

@torch.jit.script
def feet_close_xy(body_pos_w: torch.Tensor, feet_body_indexes: List[int], threshold: float, weight: float) -> torch.Tensor:
    """Reward Feet Close to XY"""
    # get feet position
    left_foot_pos_xy = body_pos_w[:, feet_body_indexes[0], :2]
    right_foot_pos_xy = body_pos_w[:, feet_body_indexes[1], :2]
    # compute distance
    distance = torch.norm(left_foot_pos_xy - right_foot_pos_xy, dim=1)
    return (distance < threshold).float() * weight

@torch.jit.script
def feet_pos_l2(joint_pos: torch.Tensor, feet_body_indexes: List[int], weight: float) -> torch.Tensor:
    """Penalize feet position using L2 squared kernel."""
    left_foot_pos = joint_pos[:, feet_body_indexes[0]]
    right_foot_pos = joint_pos[:, feet_body_indexes[1]]
    return (torch.abs(left_foot_pos) + torch.abs(right_foot_pos)) * weight


def gait_phase_reward(
        env: DirectRLEnv, 
        contact_sensor: ContactSensor, 
        leg_phases: torch.Tensor,
        feet_body_indexes: List[int],
        weight: float, 
        stance_phase_threshold: float = 0.55 # Can use 0.6 if gait_period is 1.2
        ) -> torch.Tensor:
    """Reward Gait Phase"""

    contact = contact_sensor.data.net_forces_w[:, feet_body_indexes, :3].norm(dim=-1) > 1.0


    # stance phase
    stance_phase = leg_phases < stance_phase_threshold

    # reward
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(len(feet_body_indexes)):
        match = ~(contact[:, i] ^ stance_phase[:, i]) 
        reward += match.float()
    
    return reward * weight



def feet_gait(
    env: DirectRLEnv,
    contact_sensor: ContactSensor,
    feet_body_indexes: List[int],
    period: float,
    offset: List[float],
    threshold: float,
    command: torch.Tensor,
    weight: float,
    ) -> torch.Tensor:
    """Reward Feet Gait"""

    is_contact = contact_sensor.data.current_contact_time[:, feet_body_indexes] > 0.0
    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phases = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(feet_body_indexes)):
        is_stance = leg_phases[:, i] < threshold
        reward += ~(is_contact[:, i] ^ is_stance)
    
    cmd_norm = torch.norm(command, dim=1)
    reward *= cmd_norm > 0.1
    return reward * weight

@torch.jit.script
def feet_clearance(
        body_pos_w: torch.Tensor, 
        body_lin_vel_w: torch.Tensor, 
        feet_body_indexes: List[int], 
        target_feet_height: float,
        tanh_mult: float,
        sigma: float,
        weight: float) -> torch.Tensor:
    """Reward Feet Clearance"""
    feet_height_error = torch.square(body_pos_w[:, feet_body_indexes, 2] - target_feet_height)
    feet_vel_tanh = torch.tanh(tanh_mult * torch.norm(body_lin_vel_w[:, feet_body_indexes, :2], dim=2))
    reward = feet_height_error * feet_vel_tanh
    return torch.exp(-torch.sum(reward, dim=1) / sigma) * weight


@torch.jit.script
def stand_still(joint_pos: torch.Tensor, joint_idx: List[int], default_joint_pos: torch.Tensor, vel_command: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize action if zero vel command."""
    
    return torch.sum(torch.abs(joint_pos[:, joint_idx] - default_joint_pos[:, joint_idx]), dim=1) * (torch.norm(vel_command[:, :2], dim=1) < 0.1) * weight


@torch.jit.script
def body_acc_l2(body_acc_w: torch.Tensor, body_idx: int, weight: torch.Tensor) -> torch.Tensor:
    """Penalize body linear/angular acceleration using L2 squared kernel."""

    return torch.sum(torch.square(body_acc_w[:, body_idx, :]), dim=1) * weight


@torch.jit.script
def body_acc_exp(body_acc_w: torch.Tensor, body_idx: int, weight: torch.Tensor, lambda_acc: float) -> torch.Tensor:

    acc_squared_norm = torch.sum(torch.square(body_acc_w[:, body_idx, :]), dim=1)
    return torch.exp(-lambda_acc * acc_squared_norm) * weight


@torch.jit.script
def body_orientation_l2(body_rot_w: torch.Tensor, gravity_vec_w: torch.Tensor, body_idx: int, weight: float) -> torch.Tensor:
    """Penalize body orientation using L2 squared kernel."""

    body_orientation = quat_apply_inverse(body_rot_w[:, body_idx, :], gravity_vec_w)
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1) * weight


@torch.jit.script
def cup_upright_bonus_smooth(body_rot_w: torch.Tensor, gravity_vec_w: torch.Tensor, body_idx: int, weight: float):
    """Reward Cup Upright Bonus Smooth"""
    body_orientation = quat_apply_inverse(body_rot_w[:, body_idx, :], gravity_vec_w)
    
    tilt_magnitude = torch.norm(body_orientation[:, :2], dim=1)
    
    # perfect upright (< 0.05 rad ≈ 2.9°)
    perfect_upright = (tilt_magnitude < 0.05).float() * 3.0
    
    # good upright (< 0.1 rad ≈ 5.7°)  
    good_upright = ((tilt_magnitude >= 0.05) & (tilt_magnitude < 0.1)).float() * 1.5
    
    # okay upright (< 0.15 rad ≈ 8.6°)
    okay_upright = ((tilt_magnitude >= 0.1) & (tilt_magnitude < 0.15)).float() * 0.5
    
    return perfect_upright + good_upright + okay_upright

@torch.jit.script
def object_pos_deviation(object_pos_w: torch.Tensor, plate_pos_w: torch.Tensor, default_rel_pos_w: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize object position deviation from the default relative position."""
    rel_pos = object_pos_w - plate_pos_w
    return torch.sum(torch.square(rel_pos - default_rel_pos_w), dim=1) * weight

@torch.jit.script
def cup_upright_bonus_exp(body_rot_w: torch.Tensor, gravity_vec_w: torch.Tensor, body_idx: int, weight: float, sigma: float):
    """Reward Cup Upright Bonus Exponential"""
    body_orientation = quat_apply_inverse(body_rot_w[:, body_idx, :], gravity_vec_w)
    
    tilt_magnitude = torch.norm(body_orientation[:, :2], dim=1)
    
    return torch.exp(-tilt_magnitude / sigma) * weight

@torch.jit.script
def cup_upright_bonus_exp_new(projected_gravity_b: torch.Tensor, weight: float, sigma: float):
    """Reward Cup Upright Bonus Exponential"""
    
    tilt_magnitude = torch.norm(projected_gravity_b[:, :2], dim=1)
    
    return torch.exp(-tilt_magnitude / sigma) * weight

@torch.jit.script
def body_ang_vel_l2(body_ang_vel_w: torch.Tensor, body_idx: int, weight: float) -> torch.Tensor:
    """Penalize body angular velocity using L2 squared kernel."""
    return torch.sum(torch.square(body_ang_vel_w[:, body_idx, :]), dim=1) * weight

@torch.jit.script
def penalty_residual_action(residual_actions: torch.Tensor, action_dim: Dict[str, int], weight: float, lambda_delta_upper: float, lambda_delta_lower: float) -> torch.Tensor:
    """Penalize residual action."""
    upper = residual_actions[:, :action_dim["upper_body"]]
    lower = residual_actions[:, action_dim["upper_body"]:]
    cost = lambda_delta_upper * (upper**2).sum(dim=1) + lambda_delta_lower * (lower**2).sum(dim=1)
    return cost * weight

@torch.jit.script
def residual_action_l2(residual_actions: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize residual action."""
    return torch.sum(torch.square(residual_actions), dim=1) * weight

@torch.jit.script
def body_vel_l2(body_vel: torch.Tensor, body_idx: int, weight: torch.Tensor) -> torch.Tensor:
    """Penalize body linear/angular acceleration using L2 squared kernel."""

    return torch.sum(torch.square(body_vel), dim=1) * weight

@torch.jit.script
def body_vel_exp(body_vel: torch.Tensor, body_idx: int, weight: torch.Tensor, lambda_vel: float) -> torch.Tensor:

    vel_squared_norm = torch.sum(torch.square(body_vel), dim=1)
    return torch.exp(-lambda_vel * vel_squared_norm) * weight

@torch.jit.script
def compute_pose_deviation_penalty(
    current_pose: torch.Tensor,  # [batch, 7] - (x,y,z,qw,qx,qy,qz)
    target_pose: torch.Tensor,   # [batch, 7] - (x,y,z,qw,qx,qy,qz)
    pos_weight: float = -1.0,
) -> torch.Tensor:
    # position deviation - prevent sliding
    pos_current = current_pose[:, :3]
    pos_target = target_pose[:, :3]
    
    # position penalty l2
    pos_penalty = torch.sum(torch.square(pos_current - pos_target), dim=1)
    # combine penalty
    total_penalty = pos_weight * pos_penalty
    return total_penalty


def body_contacts(threshold: float, contact_sensor: ContactSensor, body_ids: List[int], weight: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1) * weight



def joint_effort_l2(joint_effort: torch.Tensor, joint_idx: Sequence[int], weight: float, clip: tuple[float, float] = None) -> torch.Tensor:
    """Penalize joint effort using L2 squared kernel."""
    effort = torch.sum(torch.square(joint_effort[:, joint_idx]), dim=1) * weight
    if clip is not None:
        effort = torch.clip(effort, clip[0], clip[1])
    return effort

@torch.jit.script
def plate_force_xy_l1(plate_wrench: torch.Tensor, weight: float) -> torch.Tensor:
    """Penalize plate force in xy plane using L1 kernel."""
    return torch.sum(torch.abs(plate_wrench[:, :2]), dim=1) * weight



def plate_tray_holder_in_contact(plate_contact_sensor: ContactSensor, force_threshold: float = 0.01) -> torch.Tensor:
    """Reward plate tray holder contact."""
    contact_force = plate_contact_sensor.data.force_matrix_w
    fz = contact_force[:, 0, :, 2]
    is_holding_plate = torch.all(fz >= force_threshold, dim=1)
    return is_holding_plate.float()

@torch.jit.script
def track_plate_pose_exp(
    plate_pos_w: torch.Tensor,
    pelvis_pos_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
    target_plate_pos_pelvis: torch.Tensor, 
    weight: float,
    sigma: float = 0.05
) -> torch.Tensor:
    """Reward plate pose deviation."""
    # transform plate pose to pelvis frame
    plate_pos_pelvis = quat_apply_inverse(pelvis_quat_w, plate_pos_w - pelvis_pos_w)
    plate_pos_error = torch.sum(torch.square(plate_pos_pelvis - target_plate_pos_pelvis), dim=1)
    return torch.exp(-plate_pos_error / sigma) * weight

@torch.jit.script
def penalty_plate_lin_vel_robot_frame(
    robot_quat_w: torch.Tensor, 
    plate_lin_vel_w: torch.Tensor, 
    robot_lin_vel_w: torch.Tensor,
    weight: float
) -> torch.Tensor:

    lin_vel_in_robot_frame = quat_apply_inverse(robot_quat_w, plate_lin_vel_w - robot_lin_vel_w)
    return torch.sum(torch.square(lin_vel_in_robot_frame), dim=1) * weight

@torch.jit.script
def penalty_plate_ang_vel_robot_frame(
    robot_quat_w: torch.Tensor,
    plate_ang_vel_w: torch.Tensor,
    robot_ang_vel_w: torch.Tensor,
    weight: float
) -> torch.Tensor:
    ang_vel_in_robot_frame = quat_apply_inverse(robot_quat_w, plate_ang_vel_w - robot_ang_vel_w)
    return torch.sum(torch.square(ang_vel_in_robot_frame), dim=1) * weight

@torch.jit.script
def penalty_body_roll_pitch_l2(
    body_root_quat_w: torch.Tensor,
    weight: float
) -> torch.Tensor:
    """Penalize body roll and pitch using L2 squared kernel."""
    body_euler_roll, body_euler_pitch, _ = euler_xyz_from_quat(body_root_quat_w)
    
    return (body_euler_roll**2 + body_euler_pitch**2) * weight


def penalty_force_l2(
    plate_contact_sensor: ContactSensor,
    plate_quat_w: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    """Penalize large opposite signs in fy/fz (squeeze or up-down), and general imbalance."""
    force_matrix_w = plate_contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        return torch.zeros(plate_quat_w.shape[0], device=plate_quat_w.device)
    
    left_f = quat_apply_inverse(plate_quat_w, force_matrix_w[:, 0, 0, :])
    right_f = quat_apply_inverse(plate_quat_w, force_matrix_w[:, 0, 1, :])
    
    return torch.sum(torch.square(left_f), dim=1) * weight + torch.sum(torch.square(right_f), dim=1) * weight
    
    
    
    
@torch.jit.script
def track_plate_pose_l2(
    plate_pos_w: torch.Tensor,
    pelvis_pos_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
    target_plate_pos_pelvis: torch.Tensor, 
    weight: float,
) -> torch.Tensor:
    """Reward plate pose deviation."""
    # transform plate pose to pelvis frame
    plate_pos_pelvis = quat_apply_inverse(pelvis_quat_w, plate_pos_w - pelvis_pos_w)
    plate_pos_error = torch.sum(torch.square(plate_pos_pelvis - target_plate_pos_pelvis), dim=1)
    return plate_pos_error * weight

@torch.jit.script
def track_plate_pose_tanh(
    plate_pos_w: torch.Tensor,
    pelvis_pos_w: torch.Tensor,
    pelvis_quat_w: torch.Tensor,
    target_plate_pos_pelvis: torch.Tensor, 
    weight: float,
    std: float = 0.5
) -> torch.Tensor:
    """Reward plate pose deviation."""
    # transform plate pose to pelvis frame
    plate_pos_pelvis = quat_apply_inverse(pelvis_quat_w, plate_pos_w - pelvis_pos_w)
    plate_pos_error = torch.sum(torch.square(plate_pos_pelvis - target_plate_pos_pelvis), dim=1)
    return 1 - torch.tanh(plate_pos_error / std) * weight

@torch.jit.script
def penalty_pos_deviation_l2(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    weight: float
) -> torch.Tensor:
    """Penalize position deviation from the default position."""
    return torch.sum(torch.square(current_pos - target_pos), dim=1) * weight

@torch.jit.script
def penalty_quat_deviation(
    current_quat: torch.Tensor,
    target_quat: torch.Tensor,
    weight: float
) -> torch.Tensor:
    """Penalize quaternion deviation from the default quaternion."""
    return quat_error_magnitude(current_quat, target_quat) * weight




def plate_friction_penalty_per_hand(
    plate_contact_sensor: ContactSensor, 
    plate_quat_w: torch.Tensor,
    mu_static_plate: torch.Tensor,
    mu_static_left_ee: torch.Tensor,
    mu_static_right_ee: torch.Tensor,
    contact_threshold: float = 0.1,
    weight: float = -0.01,
) -> torch.Tensor:
    """Penalize friction with plate by friction cone."""
    force_matrix_w = plate_contact_sensor.data.force_matrix_w
    device = plate_quat_w.device
    if force_matrix_w is None:
        return torch.zeros(plate_quat_w.shape[0], device=device)
    
    # average static friction coefficients
    mu_static_left = (mu_static_plate + mu_static_left_ee) / 2.0
    mu_static_left = mu_static_left.to(device)
    mu_static_right = (mu_static_plate + mu_static_right_ee) / 2.0
    mu_static_right = mu_static_right.to(device)

    
    # transform contact forces to plate frame
    left_contact_forces = force_matrix_w[:, 0, 0, :]
    right_contact_forces = force_matrix_w[:, 0, 1, :]
    
    left_contact_forces_plate = quat_apply_inverse(
        plate_quat_w,  # [N,4]
        left_contact_forces  # [N,3]
    ).to(device) # [N, 3]
    right_contact_forces_plate = quat_apply_inverse(
        plate_quat_w,  # [N, 4]
        right_contact_forces  # [N, 3]
    ).to(device) # [N, 3]

    # friction cone violation
    left_normal_force = torch.abs(left_contact_forces_plate[:, 2]) + 1e-6
    right_normal_force = torch.abs(right_contact_forces_plate[:, 2]) + 1e-6
    no_left_contact = left_normal_force < contact_threshold
    no_right_contact = right_normal_force < contact_threshold
    no_contact = torch.logical_or(no_left_contact, no_right_contact).to(device)
    
    left_tangential_force = torch.norm(left_contact_forces_plate[:, :2], dim=1).to(device)
    left_friction_ratio = left_tangential_force / (mu_static_left * left_normal_force)
    left_friction_violation = left_friction_ratio ** 2

    right_tangential_force = torch.norm(right_contact_forces_plate[:, :2], dim=1).to(device)
    right_friction_ratio = right_tangential_force / (mu_static_right * right_normal_force)
    right_friction_violation = right_friction_ratio ** 2

    penalty = (left_friction_violation + right_friction_violation) * weight
    penalty = torch.where(no_contact, torch.zeros_like(penalty, device=device), penalty)
    return penalty


def plate_friction_penalty(
    plate_contact_sensor: ContactSensor, 
    plate_quat_w: torch.Tensor,
    mu_static_plate: torch.Tensor,
    mu_static_left_ee: torch.Tensor,
    mu_static_right_ee: torch.Tensor,
    contact_threshold: float = 0.1,
    weight: float = -0.01,
) -> torch.Tensor:
    """Penalize friction with plate by friction cone."""
    force_matrix_w = plate_contact_sensor.data.force_matrix_w
    device = plate_quat_w.device
    if force_matrix_w is None:
        return torch.zeros(plate_quat_w.shape[0], device=device)
    
    # average static friction coefficients
    mu_static = ((mu_static_plate + mu_static_left_ee) / 2.0 + 
                     (mu_static_plate + mu_static_right_ee) / 2.0) / 2.0
    mu_static = mu_static.to(device)
    

    
    # transform contact forces to plate frame
    total_contact_forces = force_matrix_w[:, 0, :, :].sum(dim=1)

    total_contact_forces_plate = quat_apply_inverse(
        plate_quat_w,  # [N,4]
        total_contact_forces  # [N,3]
    ).to(device) # [N, 3]

    # friction cone violation
    normal_force = torch.abs(total_contact_forces_plate[:, 2]) + 1e-6
    no_contact = normal_force < contact_threshold
    tangential_force = torch.norm(total_contact_forces_plate[:, :2], dim=1).to(device)
    friction_ratio = tangential_force / (mu_static * normal_force)
    return torch.where(no_contact, torch.zeros_like(friction_ratio, device=device), friction_ratio ** 2) * weight


def object_friction_penalty(
    object_contact_sensor: ContactSensor, 
    plate_quat_w: torch.Tensor,
    mu_static_object: torch.Tensor,
    mu_static_plate: torch.Tensor,
    contact_threshold: float = 0.1,
    weight: float = -0.01,
) -> torch.Tensor:
    """Penalize object friction with plate by friction cone."""
    force_matrix_w = object_contact_sensor.data.force_matrix_w
    device = plate_quat_w.device
    if force_matrix_w is None:
        return torch.zeros(plate_quat_w.shape[0], device=plate_quat_w.device)
    
    mu_static = (mu_static_object + mu_static_plate)/2.0
    mu_static = mu_static.to(device)
    
    # transform contact forces to plate frame
    contact_forces = force_matrix_w[:, 0, 0, :]
    forces_plate = quat_apply_inverse(plate_quat_w, contact_forces).to(device)
    
    # friction cone violation
    normal_force = torch.abs(forces_plate[:, 2]) + 1e-6
    no_contact = normal_force < contact_threshold
    tangential_force = torch.norm(forces_plate[:, :2], dim=1).to(device)
    friction_ratio = tangential_force / (mu_static * normal_force)
    return torch.where(no_contact, torch.zeros_like(friction_ratio, device=device), friction_ratio ** 2) * weight
    
@torch.jit.script
def energy(joint_vel: torch.Tensor, joint_torque: torch.Tensor, joint_idx: List[int], weight: float) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    return torch.sum(torch.abs(joint_vel[:, joint_idx]) * torch.abs(joint_torque[:, joint_idx]), dim=-1) * weight
