from __future__ import annotations

from typing import List

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster, FrameTransformer
from .joint_plate_balance_cfg import G1JointPlateBalanceEnvCfg
from steady_tray import mdp
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.managers import CommandManager, ObservationManager, ActionManager, CurriculumManager
from isaaclab.utils.string import resolve_matching_names
from .joint_locomotion_env import G1JointLocomotionEnv


class G1JointPlateBalanceEnv(G1JointLocomotionEnv):
    cfg: G1JointPlateBalanceEnvCfg

    def __init__(self, cfg: G1JointPlateBalanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)


        # defult relative pose of plate/right_tray_holder
        self.plate_transform_idx = self._left_tray_holder_transform.data.target_frame_names.index("plate")
        self.default_plate_pos2left_tray_holder = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_plate_quat2left_tray_holder = torch.zeros(self.num_envs, 4, device=self.device)
        
        self.right_tray_holder_transform_idx = self._left_tray_holder_transform.data.target_frame_names.index("right_tray_holder")
        self.default_right_tray_holder2left_tray_holder = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_right_tray_holder_quat2left_tray_holder = torch.zeros(self.num_envs, 4, device=self.device)
        
        # logging
        self._episode_sums['penalty_plate_holding_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_right_tray_holder_holding_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_holding_quat'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_right_tray_holder_holding_quat'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['tracking_plate_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_lin_vel_w'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_ang_vel_w'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_projected_gravity'] = torch.zeros(self.num_envs, device=self.device)


    def _setup_scene(self):

        # add plate contact sensor
        self._plate_contact_sensor = ContactSensor(self.cfg.plate_contact_sensor)
        self.scene.sensors["plate_contact_sensor"] = self._plate_contact_sensor

        # add tray holder transform
        self._left_tray_holder_transform = FrameTransformer(self.cfg.left_tray_holder_transform)
        self.scene.sensors["left_tray_holder_transform"] = self._left_tray_holder_transform

        super()._setup_scene()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()

        plate_dropped = self._plate.data.root_pos_w[:, 2] < self.cfg.termination_height
        died = torch.logical_or(died, plate_dropped)

        return died, time_out


    def _get_rewards(self) -> torch.Tensor:

        rewards = super()._get_rewards()

        # plate/right_tray_holder position deviation
        plate_pos2left_tray_holder = self._left_tray_holder_transform.data.target_pos_source[:, self.plate_transform_idx, :]
        right_tray_holder_pos2left_tray_holder = self._left_tray_holder_transform.data.target_pos_source[:, self.right_tray_holder_transform_idx, :]

        penalty_plate_holding_pos = mdp.penalty_pos_deviation_l2(
            current_pos=plate_pos2left_tray_holder,
            target_pos=self.default_plate_pos2left_tray_holder,
            weight=-2.0,
        )

        penalty_right_tray_holder_holding_pos = mdp.penalty_pos_deviation_l2(
            current_pos=right_tray_holder_pos2left_tray_holder,
            target_pos=self.default_right_tray_holder2left_tray_holder,
            weight=-2.0,
        )

        # plate/right_tray_holder quaternion deviation
        plate_quat2left_tray_holder = self._left_tray_holder_transform.data.target_quat_source[:, self.plate_transform_idx, :]
        right_tray_holder_quat2left_tray_holder = self._left_tray_holder_transform.data.target_quat_source[:, self.right_tray_holder_transform_idx, :]

        penalty_plate_holding_quat = mdp.penalty_quat_deviation(
            current_quat=plate_quat2left_tray_holder,
            target_quat=self.default_plate_quat2left_tray_holder,
            weight=-0.2,
        )

        penalty_right_tray_holder_holding_quat = mdp.penalty_quat_deviation(
            current_quat=right_tray_holder_quat2left_tray_holder,
            target_quat=self.default_right_tray_holder_quat2left_tray_holder,
            weight=-0.2,
        )


        # plate pose tracking
        tracking_plate_pos = mdp.track_plate_pose_exp(
            plate_pos_w=self._plate.data.root_pos_w,
            pelvis_pos_w=self.robot.data.body_pos_w[:, self.pelvis_indexes, :],
            pelvis_quat_w=self.robot.data.body_quat_w[:, self.pelvis_indexes, :],
            target_plate_pos_pelvis=self.command_manager.get_command("plate_pose"),
            weight=2.0,
            sigma=0.05,
        )


        penalty_plate_lin_vel_w = torch.sum(torch.square(self._plate.data.root_lin_vel_w), dim=1) * -0.01
        penalty_plate_ang_vel_w = torch.sum(torch.square(self._plate.data.root_ang_vel_w), dim=1) * -0.01

        # plate projected gravity
        penalty_plate_projected_gravity = mdp.flat_orientation_l2(
            projected_gravity_b=self._plate.data.projected_gravity_b,
            weight=-5.0,
        )

        plate_balance_reward = (
            tracking_plate_pos + 
            penalty_plate_lin_vel_w + 
            penalty_plate_ang_vel_w + 
            penalty_plate_projected_gravity +
            penalty_plate_holding_pos +
            penalty_right_tray_holder_holding_pos +
            penalty_plate_holding_quat +
            penalty_right_tray_holder_holding_quat)
        plate_balance_reward *= self.step_dt
        rewards['upper_body'] += plate_balance_reward
        
        self._episode_sums['penalty_plate_holding_pos'] += penalty_plate_holding_pos
        self._episode_sums['penalty_right_tray_holder_holding_pos'] += penalty_right_tray_holder_holding_pos
        self._episode_sums['penalty_plate_holding_quat'] += penalty_plate_holding_quat
        self._episode_sums['penalty_right_tray_holder_holding_quat'] += penalty_right_tray_holder_holding_quat
        self._episode_sums['tracking_plate_pos'] += tracking_plate_pos
        self._episode_sums['penalty_plate_lin_vel_w'] += penalty_plate_lin_vel_w
        self._episode_sums['penalty_plate_ang_vel_w'] += penalty_plate_ang_vel_w
        self._episode_sums['penalty_plate_projected_gravity'] += penalty_plate_projected_gravity
        

        return rewards
      


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        self.default_plate_pos2left_tray_holder[env_ids] = self._left_tray_holder_transform.data.target_pos_source[env_ids, self.plate_transform_idx, :]
        self.default_plate_quat2left_tray_holder[env_ids] = self._left_tray_holder_transform.data.target_quat_source[env_ids, self.plate_transform_idx, :]
       

        self.default_right_tray_holder2left_tray_holder[env_ids] = self._left_tray_holder_transform.data.target_pos_source[env_ids, self.right_tray_holder_transform_idx, :]
        self.default_right_tray_holder_quat2left_tray_holder[env_ids] = self._left_tray_holder_transform.data.target_quat_source[env_ids, self.right_tray_holder_transform_idx, :]
       


