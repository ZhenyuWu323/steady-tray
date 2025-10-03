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

        self.left_ee_indexes = self.robot.body_names.index(self.cfg.left_ee_name)
        self.right_ee_indexes = self.robot.body_names.index(self.cfg.right_ee_name)


        """
        Transform indices:
        """
        # plate to left/right ee
        self.plate2left_ee_idx = self._left_tray_holder_transform.data.target_frame_names.index("plate")
        self.plate2right_ee_idx = self._right_tray_holder_transform.data.target_frame_names.index("plate")

        # right/left ee to left/right ee
        self.right_ee2left_ee_idx = self._left_tray_holder_transform.data.target_frame_names.index("right_tray_holder")
        #self.left_ee2right_ee_idx = self._right_tray_holder_transform.data.target_frame_names.index("left_tray_holder")

        """
        Default relative pose:
        """
        # plate to left/right ee
        self.default_plate2left_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_plate2left_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.default_plate2right_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_plate2right_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)

        # right/left ee to left/right ee
        self.default_right_ee2left_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_right_ee2left_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.default_left_ee2right_ee_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_left_ee2right_ee_quat = torch.zeros(self.num_envs, 4, device=self.device)
        
        # logging
        self._episode_sums['penalty_plate_holding_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_right_tray_holder_holding_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_holding_quat'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_right_tray_holder_holding_quat'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['tracking_plate_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_lin_vel_w'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_ang_vel_w'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_projected_gravity'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_lin_acc_w'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_ang_acc_w'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_friction'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_force_l2'] = torch.zeros(self.num_envs, device=self.device)


    def _setup_scene(self):

        # add plate contact sensor
        self._plate_contact_sensor = ContactSensor(self.cfg.plate_contact_sensor)
        self.scene.sensors["plate_contact_sensor"] = self._plate_contact_sensor

        # add tray holder transform
        self._left_tray_holder_transform = FrameTransformer(self.cfg.left_tray_holder_transform)
        self.scene.sensors["left_tray_holder_transform"] = self._left_tray_holder_transform

        self._right_tray_holder_transform = FrameTransformer(self.cfg.right_tray_holder_transform)
        self.scene.sensors["right_tray_holder_transform"] = self._right_tray_holder_transform

        super()._setup_scene()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        died, time_out = super()._get_dones()

        plate_dropped = self._plate.data.root_pos_w[:, 2] < self.robot.data.body_pos_w[:, self.pelvis_indexes, 2]
        died = torch.logical_or(died, plate_dropped)

        return died, time_out


    def _get_rewards(self) -> torch.Tensor:

        rewards = super()._get_rewards()


        """
        Holding Pose Rewards
        """

        # plate position deviation
        plate2left_ee_pos = self._left_tray_holder_transform.data.target_pos_source[:, self.plate2left_ee_idx, :]
        plate2right_ee_pos = self._right_tray_holder_transform.data.target_pos_source[:, self.plate2right_ee_idx, :]

        # right/left ee position deviation
        right_ee2left_ee_pos = self._left_tray_holder_transform.data.target_pos_source[:, self.right_ee2left_ee_idx, :]
        

        penalty_plate2left_ee_pos = mdp.penalty_pos_deviation_l2(
            current_pos=plate2left_ee_pos,
            target_pos=self.default_plate2left_ee_pos,
            weight=-2.0,
        )
        penalty_plate2right_ee_pos = mdp.penalty_pos_deviation_l2(
            current_pos=plate2right_ee_pos,
            target_pos=self.default_plate2right_ee_pos,
            weight=-2.0,
        )
        penalty_plate_holding_pos = penalty_plate2left_ee_pos + penalty_plate2right_ee_pos


        penalty_right_tray_holder_holding_pos = mdp.penalty_pos_deviation_l2(
            current_pos=right_ee2left_ee_pos,
            target_pos=self.default_right_ee2left_ee_pos,
            weight=-2.0,
        )

        # plate/right_tray_holder quaternion deviation
        plate2left_ee_quat = self._left_tray_holder_transform.data.target_quat_source[:, self.plate2left_ee_idx, :]
        plate2right_ee_quat = self._right_tray_holder_transform.data.target_quat_source[:, self.plate2right_ee_idx, :]
        right_ee2left_ee_quat = self._left_tray_holder_transform.data.target_quat_source[:, self.right_ee2left_ee_idx, :]

        penalty_plate2left_ee_quat = mdp.penalty_quat_deviation(
            current_quat=plate2left_ee_quat,
            target_quat=self.default_plate2left_ee_quat,
            weight=-0.2,
        )
        penalty_plate2right_ee_quat = mdp.penalty_quat_deviation(
            current_quat=plate2right_ee_quat,
            target_quat=self.default_plate2right_ee_quat,
            weight=-0.2,
        )
        penalty_plate_holding_quat = penalty_plate2left_ee_quat + penalty_plate2right_ee_quat

        
        penalty_right_tray_holder_holding_quat = mdp.penalty_quat_deviation(
            current_quat=right_ee2left_ee_quat,
            target_quat=self.default_right_ee2left_ee_quat,
            weight=-0.2,
        )


        """
        Plate Pose Tracking Rewards
        """
        # plate pose tracking
        tracking_plate_pos = mdp.track_plate_pose_exp(
            plate_pos_w=self._plate.data.root_pos_w,
            pelvis_pos_w=self.robot.data.body_pos_w[:, self.pelvis_indexes, :],
            pelvis_quat_w=self.robot.data.body_quat_w[:, self.pelvis_indexes, :],
            target_plate_pos_pelvis=self.command_manager.get_command("plate_pose"),
            weight=1.0,
            sigma=0.05,
        )


        """
        Plate Motion Rewards
        """
        # penalty_plate_lin_vel_w = torch.sum(torch.square(self._plate.data.root_lin_vel_w), dim=1) * -0.01
        # penalty_plate_ang_vel_w = torch.sum(torch.square(self._plate.data.root_ang_vel_w), dim=1) * -0.01
        # penalty_plate_lin_acc_w = torch.sum(torch.square(self._plate.data.body_lin_acc_w[:, 0, :]), dim=1) * -0.001
        # penalty_plate_lin_acc_w = torch.clip(penalty_plate_lin_acc_w, min=-1.0)
        # penalty_plate_ang_acc_w = torch.sum(torch.square(self._plate.data.body_ang_acc_w[:, 0, :]), dim=1) * -1e-4
        # penalty_plate_ang_acc_w = torch.clip(penalty_plate_ang_acc_w, min=-1.0)

        # plate projected gravity
        penalty_plate_projected_gravity = mdp.flat_orientation_l2(
            projected_gravity_b=self._plate.data.projected_gravity_b,
            weight=-5.0,
        )


        """
        Plate Force Rewards
        """
        # penalty_plate_friction = mdp.plate_friction_penalty(
        #     plate_contact_sensor=self._plate_contact_sensor,
        #     plate_quat_w=self._plate.data.root_quat_w,
        #     mu_static_plate=self._plate.data._root_physx_view.get_material_properties()[:, 0, 0],
        #     mu_static_left_ee=self.robot.data._root_physx_view.get_material_properties()[:, self.left_ee_indexes, 0],
        #     mu_static_right_ee=self.robot.data._root_physx_view.get_material_properties()[:, self.right_ee_indexes, 0],
        #     weight=-0.2,
        # )
        # penalty_plate_friction = torch.clip(penalty_plate_friction, min=-1.0)

        # penalty_force_l2 = mdp.penalty_force_l2(
        #     plate_contact_sensor=self._plate_contact_sensor,
        #     plate_quat_w=self._plate.data.root_quat_w,
        #     weight=0,
        # )
        #penalty_force_l2 = torch.clip(penalty_force_l2, min=-1.0)


        plate_balance_reward = (
            tracking_plate_pos + 
            # penalty_plate_lin_vel_w + 
            # penalty_plate_ang_vel_w + 
            penalty_plate_projected_gravity +
            penalty_plate_holding_pos +
            penalty_right_tray_holder_holding_pos +
            penalty_plate_holding_quat +
            penalty_right_tray_holder_holding_quat
            # penalty_plate_friction +
            # penalty_force_l2 +
            # penalty_plate_lin_acc_w +
            # penalty_plate_ang_acc_w
        )
        plate_balance_reward *= self.step_dt
        rewards['upper_body'] += plate_balance_reward
        
        self._episode_sums['penalty_plate_holding_pos'] += penalty_plate_holding_pos
        self._episode_sums['penalty_right_tray_holder_holding_pos'] += penalty_right_tray_holder_holding_pos
        self._episode_sums['penalty_plate_holding_quat'] += penalty_plate_holding_quat
        self._episode_sums['penalty_right_tray_holder_holding_quat'] += penalty_right_tray_holder_holding_quat
        self._episode_sums['tracking_plate_pos'] += tracking_plate_pos
        # self._episode_sums['penalty_plate_lin_vel_w'] += penalty_plate_lin_vel_w
        # self._episode_sums['penalty_plate_ang_vel_w'] += penalty_plate_ang_vel_w
        # self._episode_sums['penalty_plate_lin_acc_w'] += penalty_plate_lin_acc_w
        # self._episode_sums['penalty_plate_ang_acc_w'] += penalty_plate_ang_acc_w
        self._episode_sums['penalty_plate_projected_gravity'] += penalty_plate_projected_gravity
        # self._episode_sums['penalty_plate_friction'] += penalty_plate_friction
        # self._episode_sums['penalty_force_l2'] += penalty_force_l2
        

        return rewards
      


    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # plate default relative pose
        self.default_plate2left_ee_pos[env_ids] = self._left_tray_holder_transform.data.target_pos_source[env_ids, self.plate2left_ee_idx, :]
        self.default_plate2left_ee_quat[env_ids] = self._left_tray_holder_transform.data.target_quat_source[env_ids, self.plate2left_ee_idx, :]
        self.default_plate2right_ee_pos[env_ids] = self._right_tray_holder_transform.data.target_pos_source[env_ids, self.plate2right_ee_idx, :]
        self.default_plate2right_ee_quat[env_ids] = self._right_tray_holder_transform.data.target_quat_source[env_ids, self.plate2right_ee_idx, :]

        # right/left ee default relative pose
        self.default_right_ee2left_ee_pos[env_ids] = self._left_tray_holder_transform.data.target_pos_source[env_ids, self.right_ee2left_ee_idx, :]
        self.default_right_ee2left_ee_quat[env_ids] = self._left_tray_holder_transform.data.target_quat_source[env_ids, self.right_ee2left_ee_idx, :]
        #self.default_left_ee2right_ee_pos[env_ids] = self._right_tray_holder_transform.data.target_pos_source[env_ids, self.left_ee2right_ee_idx, :]
        #self.default_left_ee2right_ee_quat[env_ids] = self._right_tray_holder_transform.data.target_quat_source[env_ids, self.left_ee2right_ee_idx, :]




