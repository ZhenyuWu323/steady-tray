from __future__ import annotations
import math
from typing import List

import torch

import gymnasium as gym
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_rotate,quat_apply
from isaaclab.sensors import ContactSensor, RayCaster, FrameTransformer, OffsetCfg
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.utils.noise.noise_model import uniform_noise
from .residual_cfg import G1ResidualEnvCfg
from isaaclab.managers import SceneEntityCfg
from steady_tray import mdp
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.utils.buffers import CircularBuffer
from isaaclab.utils.math import quat_apply_inverse
from isaaclab.managers import CommandManager, ObservationManager, ActionManager
from isaaclab.utils.string import resolve_matching_names


class G1ResidualEnv(DirectRLEnv):
    cfg: G1ResidualEnvCfg

    def __init__(self, cfg: G1ResidualEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        ##########################################################################################
        # DOF and key body indexes
        ##########################################################################################
        
        # body keys
        self.body_keys = self.cfg.body_keys

        # joint indexes
        self.upper_body_indexes = self.robot.find_joints(self.cfg.upper_body_names)[0] # arm and fingers
        self.feet_indexes = self.robot.find_joints(self.cfg.feet_names)[0]
        self.waist_indexes = self.robot.find_joints(self.cfg.waist_names)[0]
        self.hips_yaw_roll_indexes = self.robot.find_joints(self.cfg.hips_names[:2])[0]
        self.knee_indexes = self.robot.find_joints(self.cfg.hips_names[-1])[0]
        self.hips_indexes = self.robot.find_joints(self.cfg.hips_names)[0]
        self.lower_body_indexes = self.waist_indexes + self.hips_indexes + self.feet_indexes # lower body
        self.pelvis_indexes = self.robot.find_bodies(self.cfg.pelvis_names)[0][0]

        # ee indexes
        self.left_ee_indexes = self.robot.body_names.index(self.cfg.left_ee_name)
        self.right_ee_indexes = self.robot.body_names.index(self.cfg.right_ee_name)

        # camera body index
        self.camera_body_index = self.robot.data.body_names.index(self.cfg.camera_names)


        # body/link indexes
        self.feet_body_indexes = self.robot.find_bodies(self.cfg.feet_body_name)[0]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body) # torso link

        # action scale
        self.action_scale = self.cfg.action_scale

        # default joint positions
        self.default_joint_pos = self.robot.data.default_joint_pos
        self.default_lower_joint_pos = self.default_joint_pos[:,self.lower_body_indexes]
        self.default_upper_joint_pos = self.default_joint_pos[:,self.upper_body_indexes]

        # sdk joint sequence
        joint_ids_map, _ = resolve_matching_names(self.robot.joint_names, self.cfg.sdk_joint_sequence, preserve_order=True)
        

        # actions and previous actions
        self.base_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_base_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.residual_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)
        self.prev_residual_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.sim.device)

        # command 
        self.command_manager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)


        # observations
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager: ", self.observation_manager)


        # history
        self.obs_history_length = getattr(self.cfg, 'obs_history_length', 5)  # t-4:t (5 steps)

        
        """
        Transform indices
        """
        # plate to left/right ee
        self.plate2left_ee_idx = self._left_tray_holder_transform.data.target_frame_names.index("plate")
        self.plate2right_ee_idx = self._right_tray_holder_transform.data.target_frame_names.index("plate")
        # right ee to left ee
        self.right_ee2left_ee_idx = self._left_tray_holder_transform.data.target_frame_names.index("right_tray_holder")
        # object to plate
        self.object_tray_idx = self._object_tray_transform.data.target_frame_names.index("object")

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
        
        # object to plate
        self.default_object2plate_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.default_object2plate_quat = torch.zeros(self.num_envs, 4, device=self.device)

        # logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "tracking_lin_vel_xy",
                "tracking_ang_vel_z",
                "gait_phase_reward",
                "feet_clearance_reward",
                "penalty_plate_holding_pos",
                "penalty_right_tray_holder_holding_pos",
                "penalty_plate_holding_quat",
                "penalty_right_tray_holder_holding_quat",
                "tracking_plate_pos",
                "penalty_plate_lin_acc_w",
                "penalty_plate_ang_acc_w",
                "penalty_plate_projected_gravity",
                "penalty_plate_friction",
                "penalty_force_l2",
                "penalty_object_pose_deviation",
                "penalty_object_quat_deviation",
                "penalty_object_projected_gravity",
                "object_upright_bonus",
                "penalty_object_lin_vel_plate",
                "penalty_object_ang_vel_plate",
                "penalty_object_friction",
            ]
        }



           

    def _setup_scene(self):
        # robot
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        # contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # height scanner
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        #number of envs
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.scene._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # plate
        self._plate = RigidObject(self.cfg.plate_cfg)
        self.scene.rigid_objects["plate"] = self._plate

        # add plate contact sensor
        self._plate_contact_sensor = ContactSensor(self.cfg.plate_contact_sensor)
        self.scene.sensors["plate_contact_sensor"] = self._plate_contact_sensor

        # add tray holder transform
        self._left_tray_holder_transform = FrameTransformer(self.cfg.left_tray_holder_transform)
        self.scene.sensors["left_tray_holder_transform"] = self._left_tray_holder_transform

        self._right_tray_holder_transform = FrameTransformer(self.cfg.right_tray_holder_transform)
        self.scene.sensors["right_tray_holder_transform"] = self._right_tray_holder_transform

        # add object
        self._object = RigidObject(self.cfg.obj_cfg)
        self.scene.rigid_objects["object"] = self._object

        # add object contact sensor
        self._object_contact_sensor = ContactSensor(self.cfg.object_contact_sensor)
        self.scene.sensors["object_contact_sensor"] = self._object_contact_sensor

        # add object tray transform
        self._object_tray_transform = FrameTransformer(self.cfg.object_tray_transform)
        self.scene.sensors["object_tray_transform"] = self._object_tray_transform

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.cfg.sky_light_cfg.func("/World/Light", self.cfg.sky_light_cfg)


    def _pre_physics_step(self, base_action: torch.Tensor, residual_action: torch.Tensor):
        # update previous actions
        self.prev_base_actions = self.base_actions.clone()
        self.prev_residual_actions = self.residual_actions.clone()
        self.base_actions = base_action.clone()
        self.residual_actions = residual_action.clone()

    def _apply_action(self):
        actions = self.base_actions + self.residual_actions
        upper_actions = actions[:, :self.cfg.action_dim["upper_body"]]
        lower_actions = actions[:, self.cfg.action_dim["upper_body"]:]

        upper_body_target = self.default_upper_joint_pos + self.action_scale * upper_actions
        lower_body_target = self.default_lower_joint_pos + self.action_scale * lower_actions

        # set upper body
        self.robot.set_joint_position_target(upper_body_target, self.upper_body_indexes)
        # set lower body
        self.robot.set_joint_position_target(lower_body_target, self.lower_body_indexes)


    def _get_observations(self) -> dict:

        obs_dict = self.observation_manager.compute()
        action_dim = self.cfg.action_dim["upper_body"] + self.cfg.action_dim["lower_body"]
        shared_actor_obs = obs_dict['actor_obs'][:, :-action_dim * self.obs_history_length]
        residual_actor_obs = torch.cat([shared_actor_obs, obs_dict['residual_actor_obs']], dim=1)
        obs_dict['residual_actor_obs'] = residual_actor_obs
        return obs_dict
    

    def _get_locomotion_rewards(self) -> torch.Tensor:
        """
        Lower Body Tracking Rewards
        """
        tracking_lin_vel_xy = mdp.track_lin_vel_xy_yaw_frame_exp(
            root_quat_w=self.robot.data.root_quat_w,
            root_lin_vel_w=self.robot.data.root_lin_vel_w,
            vel_command=self.command_manager.get_command("base_velocity"),
            sigma=0.25,
            weight=1.0,
        )
        tracking_ang_vel_z = mdp.track_ang_vel_z_base_exp(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            vel_command=self.command_manager.get_command("base_velocity"),
            sigma=0.25,
            weight=0.5,
        )


        """
        Lower Body Penalty Terms
        """
        # terminate when the robot falls
        died, _ = self._get_dones()
      

        # linear velocity z
        penalty_lin_vel_z = mdp.lin_vel_z_l2(
            root_lin_vel_b=self.robot.data.root_lin_vel_b,
            weight=-2.0,
        )

        # angular velocity xy
        penalty_ang_vel_xy = mdp.ang_vel_xy_l2(
            root_ang_vel_b=self.robot.data.root_ang_vel_b,
            weight=-0.05,
        )

        # flat orientation
        penalty_flat_orientation = mdp.flat_orientation_l2(
            projected_gravity_b=self.robot.data.projected_gravity_b,
            weight=-5.0,
        )

        # joint deviation waist
        penalty_dof_pos_waist = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.waist_indexes,
            weight=-1.0,
        )

        # joint deviation hips
        penalty_dof_pos_hips = mdp.joint_deviation_l1(
            joint_pos=self.robot.data.joint_pos,
            default_joint_pos=self.robot.data.default_joint_pos,
            joint_idx=self.hips_yaw_roll_indexes,
            weight=-1.0,
        )

        # joint position limits
        penalty_lower_body_dof_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.lower_body_indexes,
            weight=-5.0,
        )


        # joint accelerations
        penalty_lower_body_dof_acc = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.lower_body_indexes,
            weight=-2.5e-7,
        )

        # joint velocities
        penalty_lower_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.lower_body_indexes,
            weight=-0.001,
        )

        # action rate
        penalty_lower_body_action_rate = mdp.action_rate_l2(
            action=self.residual_actions[:, self.cfg.action_dim["upper_body"]:],
            prev_action=self.prev_residual_actions[:, self.cfg.action_dim["upper_body"]:],
            weight=-0.05,
        )

        # base height
        penalty_base_height = mdp.base_height(
            body_pos_w=self.robot.data.body_pos_w,
            body_idx=self.ref_body_index,
            height_scanner=self._height_scanner,
            target_height=self.cfg.target_base_height,
            weight=-10,
        )

        """
        Lower Body Feet Contact Rewards
        """
        # feet slides penalty
        penalty_feet_slide = mdp.feet_slide(
            body_lin_vel_w=self.robot.data.body_lin_vel_w,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            weight=-0.2,
        )


        # feet gait
        feet_gait_reward = mdp.feet_gait(
            env=self,
            contact_sensor=self._contact_sensor,
            feet_body_indexes=self.feet_body_indexes,
            period=0.8,
            offset=[0.0, 0.5],
            threshold=0.55,
            command=self.command_manager.get_command("base_velocity"),
            weight=0.5,
        )

        # feet clearance
        feet_clearance_reward = mdp.feet_clearance(
            body_pos_w=self.robot.data.body_pos_w,
            body_lin_vel_w=self.robot.data.body_lin_vel_w,
            feet_body_indexes=self.feet_body_indexes,
            target_feet_height=self.cfg.target_feet_height,
            sigma=0.05,
            tanh_mult=2.0,
            weight=1.0,
        )

        """
        Upper Body Penalty Terms
        """
        # upper body accelerations
        penalty_upper_body_dof_acc = mdp.joint_accel_l2(
            joint_accel=self.robot.data.joint_acc,
            joint_idx=self.upper_body_indexes,
            weight=-2.5e-7,
        )

        # upper body position limits
        penalty_upper_body_dof_pos_limits = mdp.joint_pos_limits(
            joint_pos=self.robot.data.joint_pos,
            soft_joint_pos_limits=self.robot.data.soft_joint_pos_limits,
            joint_idx=self.upper_body_indexes,
            weight=-5.0,
        )

        # upper body action rate
        penalty_upper_body_action_rate = mdp.action_rate_l2(
            action=self.residual_actions[:, :self.cfg.action_dim["upper_body"]],
            prev_action=self.prev_residual_actions[:, :self.cfg.action_dim["upper_body"]],
            weight=-0.05,
        )

        # upper body velocities
        penalty_upper_body_dof_vel = mdp.joint_vel_l2(
            joint_vel=self.robot.data.joint_vel,
            joint_idx=self.upper_body_indexes,
            weight=-0.001,
        )

            
        # alive reward
        alive_reward = mdp.alive_reward(terminated=died, weight=0.15)

        locomotion_reward = (
            tracking_lin_vel_xy +          
            tracking_ang_vel_z + 
            penalty_lin_vel_z + 
            penalty_ang_vel_xy + 
            penalty_flat_orientation + 
            penalty_dof_pos_waist + 
            penalty_dof_pos_hips + 
            penalty_lower_body_dof_pos_limits + 
            penalty_lower_body_dof_acc + 
            penalty_lower_body_dof_vel + 
            penalty_lower_body_action_rate + 
            penalty_feet_slide + 
            penalty_base_height +
            feet_gait_reward +
            feet_clearance_reward +
            alive_reward + 
            penalty_upper_body_dof_acc + 
            penalty_upper_body_dof_pos_limits + 
            penalty_upper_body_action_rate + 
            penalty_upper_body_dof_vel
        )
        self._episode_sums["tracking_lin_vel_xy"] += tracking_lin_vel_xy
        self._episode_sums["tracking_ang_vel_z"] += tracking_ang_vel_z
        self._episode_sums["gait_phase_reward"] += feet_gait_reward
        self._episode_sums["feet_clearance_reward"] += feet_clearance_reward
        return locomotion_reward
    

    def _get_plate_rewards(self) -> torch.Tensor:
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
            weight=2.0,
            sigma=0.05,
        )

        """
        Plate Motion Rewards
        """
        penalty_plate_lin_acc_w = torch.sum(torch.square(self._plate.data.body_lin_acc_w[:, 0, :]), dim=1) * -0.001
        penalty_plate_lin_acc_w = torch.clip(penalty_plate_lin_acc_w, min=-1.0)
        penalty_plate_ang_acc_w = torch.sum(torch.square(self._plate.data.body_ang_acc_w[:, 0, :]), dim=1) * -1e-4
        penalty_plate_ang_acc_w = torch.clip(penalty_plate_ang_acc_w, min=-1.0)

        # plate projected gravity
        penalty_plate_projected_gravity = mdp.flat_orientation_l2(
            projected_gravity_b=self._plate.data.projected_gravity_b,
            weight=-5.0,
        )


        """
        Plate Force Rewards
        """
        penalty_plate_friction = mdp.plate_friction_penalty(
            plate_contact_sensor=self._plate_contact_sensor,
            plate_quat_w=self._plate.data.root_quat_w,
            mu_static_plate=self._plate.data._root_physx_view.get_material_properties()[:, 0, 0],
            mu_static_left_ee=self.robot.data._root_physx_view.get_material_properties()[:, self.left_ee_indexes, 0],
            mu_static_right_ee=self.robot.data._root_physx_view.get_material_properties()[:, self.right_ee_indexes, 0],
            weight=-0.2,
        )
        penalty_plate_friction = torch.clip(penalty_plate_friction, min=-1.0)

        penalty_force_l2 = mdp.penalty_force_l2(
            plate_contact_sensor=self._plate_contact_sensor,
            plate_quat_w=self._plate.data.root_quat_w,
            weight=-1e-3,
        )
        penalty_force_l2 = torch.clip(penalty_force_l2, min=-1.0)

        plate_balance_reward = (
            tracking_plate_pos + 
            penalty_plate_projected_gravity +
            penalty_plate_holding_pos +
            penalty_right_tray_holder_holding_pos +
            penalty_plate_holding_quat +
            penalty_right_tray_holder_holding_quat +
            penalty_plate_friction +
            penalty_force_l2 +
            penalty_plate_lin_acc_w +
            penalty_plate_ang_acc_w
        )
        self._episode_sums['penalty_plate_holding_pos'] += penalty_plate_holding_pos
        self._episode_sums['penalty_right_tray_holder_holding_pos'] += penalty_right_tray_holder_holding_pos
        self._episode_sums['penalty_plate_holding_quat'] += penalty_plate_holding_quat
        self._episode_sums['penalty_right_tray_holder_holding_quat'] += penalty_right_tray_holder_holding_quat
        self._episode_sums['tracking_plate_pos'] += tracking_plate_pos
        self._episode_sums['penalty_plate_lin_acc_w'] += penalty_plate_lin_acc_w
        self._episode_sums['penalty_plate_ang_acc_w'] += penalty_plate_ang_acc_w
        self._episode_sums['penalty_plate_projected_gravity'] += penalty_plate_projected_gravity
        self._episode_sums['penalty_plate_friction'] += penalty_plate_friction
        self._episode_sums['penalty_force_l2'] += penalty_force_l2
        return plate_balance_reward


    def _get_object_rewards(self) -> torch.Tensor:
        """
        Object Pose Rewards
        """
        object_pos_in_plate = self._object_tray_transform.data.target_pos_source[:, self.object_tray_idx, :]
        object_quat_in_plate = self._object_tray_transform.data.target_quat_source[:, self.object_tray_idx, :]

        penalty_object_pose_deviation = mdp.penalty_pos_deviation_l2(
            current_pos=object_pos_in_plate,
            target_pos=self.default_object2plate_pos,
            weight=-2.0,
        )

        penalty_object_quat_deviation = mdp.penalty_quat_deviation(
            current_quat=object_quat_in_plate,
            target_quat=self.default_object2plate_quat,
            weight=-0.2,
        )

        penalty_object_projected_gravity = mdp.flat_orientation_l2(
            projected_gravity_b=self._object.data.projected_gravity_b,
            weight=-5.0,
        )
        
        object_upright_bonus = mdp.cup_upright_bonus_exp_new(
            projected_gravity_b=self._object.data.projected_gravity_b,
            weight=1.0,
            sigma=0.1,
        )

        """
        Object Motion Rewards
        """
        object_twist_in_plate_frame = mdp.object_twist_in_plate_frame(self)
        penalty_object_lin_vel_plate = torch.sum(torch.square(object_twist_in_plate_frame[:, :3]), dim=1) * -0.1
        penalty_object_ang_vel_plate = torch.sum(torch.square(object_twist_in_plate_frame[:, 3:6]), dim=1) * -0.001

        """
        Object Force Rewards
        """
        penalty_object_friction = mdp.object_friction_penalty(
            object_contact_sensor=self._object_contact_sensor,
            plate_quat_w=self._plate.data.root_quat_w,
            mu_static_object=self._object.data._root_physx_view.get_material_properties()[:, 0, 0],
            mu_static_plate=self._plate.data._root_physx_view.get_material_properties()[:, 0, 0],
            weight=-0.2,
        )
        penalty_object_friction = torch.clip(penalty_object_friction, min=-1.0)


        object_stabilization_reward = (
            penalty_object_pose_deviation +
            penalty_object_quat_deviation +
            penalty_object_projected_gravity +
            object_upright_bonus +
            penalty_object_lin_vel_plate +
            penalty_object_ang_vel_plate +
            penalty_object_friction
        )

        self._episode_sums['penalty_object_pose_deviation'] += penalty_object_pose_deviation
        self._episode_sums['penalty_object_quat_deviation'] += penalty_object_quat_deviation
        self._episode_sums['penalty_object_projected_gravity'] += penalty_object_projected_gravity
        self._episode_sums['object_upright_bonus'] += object_upright_bonus
        self._episode_sums['penalty_object_lin_vel_plate'] += penalty_object_lin_vel_plate
        self._episode_sums['penalty_object_ang_vel_plate'] += penalty_object_ang_vel_plate
        self._episode_sums['penalty_object_friction'] += penalty_object_friction
        return object_stabilization_reward
        
    

    def _get_rewards(self) -> torch.Tensor:
        
        """
        Locomotion Rewards
        """
        locomotion_reward = self._get_locomotion_rewards()

        """
        Plate Balance Rewards
        """
        plate_balance_reward = self._get_plate_rewards()

            
        """
        Object Stablization Rewards
        """
        object_stabilization_reward = self._get_object_rewards()

        
        """
        Residual Rewards
        """
        residual_whole_body_reward = (locomotion_reward + plate_balance_reward + object_stabilization_reward) * self.step_dt
        
        
        return {'upper_body': torch.zeros(self.num_envs, device=self.device), 
                'lower_body': torch.zeros(self.num_envs, device=self.device), 
                'residual_whole_body': residual_whole_body_reward}


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time out
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # fall
        died = self.robot.data.body_pos_w[:, self.ref_body_index, 2] < self.cfg.termination_height

        plate_dropped = self._plate.data.root_pos_w[:, 2] < self.cfg.termination_height
        object_dropped = self._object.data.root_pos_w[:, 2] < self.cfg.termination_height
        died = torch.logical_or(died, plate_dropped)
        died = torch.logical_or(died, object_dropped)

        return died, time_out
    


    def _reset_idx(self, env_ids: torch.Tensor | None):
        extras = dict()
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # apply terrain curriculum
        if self.cfg.terrain_generator_cfg.curriculum:
            avg_terrain_level = mdp.terrain_levels(env=self, env_ids=env_ids, vel_command=self.command_manager.get_command("base_velocity"))
            extras["Curriculum/terrain_level"] = avg_terrain_level.item()

        
        # reset robot
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # plate default relative pose
        self.default_plate2left_ee_pos[env_ids] = self._left_tray_holder_transform.data.target_pos_source[env_ids, self.plate2left_ee_idx, :]
        self.default_plate2left_ee_quat[env_ids] = self._left_tray_holder_transform.data.target_quat_source[env_ids, self.plate2left_ee_idx, :]
        self.default_plate2right_ee_pos[env_ids] = self._right_tray_holder_transform.data.target_pos_source[env_ids, self.plate2right_ee_idx, :]
        self.default_plate2right_ee_quat[env_ids] = self._right_tray_holder_transform.data.target_quat_source[env_ids, self.plate2right_ee_idx, :]

        # right/left ee default relative pose
        self.default_right_ee2left_ee_pos[env_ids] = self._left_tray_holder_transform.data.target_pos_source[env_ids, self.right_ee2left_ee_idx, :]
        self.default_right_ee2left_ee_quat[env_ids] = self._left_tray_holder_transform.data.target_quat_source[env_ids, self.right_ee2left_ee_idx, :]
        
        # object default relative pose
        self.default_object2plate_pos[env_ids] = self._object_tray_transform.data.target_pos_source[env_ids, self.object_tray_idx, :]
        self.default_object2plate_quat[env_ids] = self._object_tray_transform.data.target_quat_source[env_ids, self.object_tray_idx, :]

        # reset command
        self.command_manager.reset(env_ids)
        self.event_manager.reset(env_ids)
        self.observation_manager.reset(env_ids)
        # reset actions
        self.base_actions[env_ids] = 0.0
        self.prev_base_actions[env_ids] = 0.0
        self.residual_actions[env_ids] = 0.0
        self.prev_residual_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        

        # reset logging
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        base_action = action['base_action'].to(self.device)
        residual_action = action['residual_action'].to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            base_action = self._action_noise_model.apply(base_action)
            residual_action = self._action_noise_model.apply(residual_action)

        # clip actions
        clip_actions = self.cfg.clip_action
        base_action = torch.clip(base_action, -clip_actions, clip_actions)
        residual_action = torch.clip(residual_action, -clip_actions, clip_actions)

        # process actions
        self._pre_physics_step(base_action, residual_action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update gait phase
        #self._post_physics_step()

        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # post-step: step interval event
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        #if self.cfg.observation_noise_model:
            #self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        # clip observations
        clip_observations = self.cfg.clip_observation
        for key, value in self.obs_buf.items():
            self.obs_buf[key] = torch.clip(value, -clip_observations, clip_observations)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def close(self):
        if not self._is_closed:
            del self.command_manager
            del self.observation_manager
        super().close()
