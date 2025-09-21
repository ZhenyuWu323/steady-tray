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
       
        # logging
        self._episode_sums['plate_tray_holder_in_contact'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['tracking_plate_pos'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_lin_vel_robot_frame'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_ang_vel_robot_frame'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_roll_pitch'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_force_l2'] = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums['penalty_plate_projected_gravity'] = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):

        # add plate contact sensor
        self._plate_contact_sensor = ContactSensor(self.cfg.plate_contact_sensor)
        self.scene.sensors["plate_contact_sensor"] = self._plate_contact_sensor
        

        super()._setup_scene()


    def _get_rewards(self) -> torch.Tensor:

        rewards = super()._get_rewards()


        # plate tray holder in contact
        plate_tray_holder_in_contact = mdp.plate_tray_holder_in_contact(
            plate_contact_sensor=self._plate_contact_sensor,
            force_threshold=0.05,
        )

        reward_plate_tray_holder_in_contact = 0.15 * plate_tray_holder_in_contact


        # plate pose tracking
        tracking_plate_pos = mdp.track_plate_pose_exp(
            plate_pos_w=self._plate.data.root_pos_w,
            pelvis_pos_w=self.robot.data.body_pos_w[:, self.pelvis_indexes, :],
            pelvis_quat_w=self.robot.data.body_quat_w[:, self.pelvis_indexes, :],
            target_plate_pos_pelvis=self.command_manager.get_command("plate_pose"),
            weight=5.0 * plate_tray_holder_in_contact,
            sigma=0.05,
        )

        # plate lin vel robot frame
        penalty_plate_lin_vel_robot_frame = mdp.penalty_plate_lin_vel_robot_frame(
            robot_quat_w=self.robot.data.root_quat_w,
            plate_lin_vel_w=self._plate.data.root_lin_vel_w,
            robot_lin_vel_w=self.robot.data.root_lin_vel_w,
            weight=-0.01 * plate_tray_holder_in_contact,
        )

        # plate ang vel robot frame
        penalty_plate_ang_vel_robot_frame = mdp.penalty_plate_ang_vel_robot_frame(
            robot_quat_w=self.robot.data.root_quat_w,
            plate_ang_vel_w=self._plate.data.root_ang_vel_w,
            robot_ang_vel_w=self.robot.data.root_ang_vel_w,
            weight=-0.01 * plate_tray_holder_in_contact,
        )

        # plate projected gravity
        penalty_plate_projected_gravity = mdp.flat_orientation_l2(
            projected_gravity_b=self._plate.data.projected_gravity_b,
            weight=-5.0 * plate_tray_holder_in_contact,
        )

        penalty_body_roll_pitch_l2 = mdp.penalty_body_roll_pitch_l2(
            body_root_quat_w=self._plate.data.root_quat_w,
            weight=-0.0 * plate_tray_holder_in_contact,
        )

        
        # plate force l2
        penalty_force_l2 = mdp.penalty_force_l2(
            plate_contact_sensor=self._plate_contact_sensor,
            weight=-0.001 * plate_tray_holder_in_contact,
        )

        

        plate_balance_reward = (
            reward_plate_tray_holder_in_contact +
            tracking_plate_pos + 
            penalty_plate_lin_vel_robot_frame + 
            penalty_plate_ang_vel_robot_frame + 
            penalty_body_roll_pitch_l2 +
            penalty_plate_projected_gravity +
            penalty_force_l2)
        plate_balance_reward *= self.step_dt
        rewards['upper_body'] += plate_balance_reward
        
        self._episode_sums['plate_tray_holder_in_contact'] += reward_plate_tray_holder_in_contact
        self._episode_sums['tracking_plate_pos'] += tracking_plate_pos
        self._episode_sums['penalty_plate_lin_vel_robot_frame'] += penalty_plate_lin_vel_robot_frame
        self._episode_sums['penalty_plate_ang_vel_robot_frame'] += penalty_plate_ang_vel_robot_frame
        self._episode_sums['penalty_plate_roll_pitch'] += penalty_body_roll_pitch_l2
        self._episode_sums['penalty_plate_projected_gravity'] += penalty_plate_projected_gravity
        self._episode_sums['penalty_force_l2'] += penalty_force_l2

        return rewards
      


    
        



