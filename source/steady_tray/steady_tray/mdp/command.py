from __future__ import annotations

from dataclasses import MISSING
from typing import Sequence

import torch

from isaaclab.envs.mdp import UniformVelocityCommandCfg,UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CommandTerm
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.utils.math import quat_apply
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.markers import VisualizationMarkers
from steady_tray.assets import PLATE_OFFSET

"""
Velocity Command
"""
class UniformLevelVelocityCommand(UniformVelocityCommand):
    cfg: UniformLevelVelocityCommandCfg

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.delay_steps = getattr(cfg, 'delay_steps', 0)
        self.steps_since_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        

    @property
    def command(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self.vel_command_b)
        result[mask] = self.vel_command_b[mask]
        return result
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.steps_since_reset[env_ids] = 0
        return super().reset(env_ids)
    
    def compute(self, dt: float):
        self.steps_since_reset += 1
        super().compute(dt)
    


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    delay_steps: int = 0
    """The number of steps to delay the command."""


"""
Joint Position Command
"""


class UpperBodyJointPositionCommand(CommandTerm):
    cfg: UpperBodyJointPositionCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UpperBodyJointPositionCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.joint_pos_command = torch.tensor(cfg.joint_pos_command, device=env.device).repeat(env.num_envs, 1)


    def __str__(self) -> str:
        return f"Joint Position Command: {self.joint_pos_command}"
    
    @property
    def command(self) -> torch.Tensor:
        return self.joint_pos_command
    
    def _update_command(self):
        pass
    
    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_metrics(self):
        pass
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass
    
    def _debug_vis_callback(self, event):
        pass


@configclass
class UpperBodyJointPositionCommandCfg(CommandTermCfg):
    class_type: type = UpperBodyJointPositionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    joint_names: list[str] = MISSING
    """Names of the joints in the asset for which the commands are generated."""

    joint_pos_command: list[float] = MISSING
    """Joint position command for the joints in the asset."""



"""
Plate Pose Command
"""

class PlatePoseCommand(CommandTerm):
    cfg: PlatePoseCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: PlatePoseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.robot_asset_name]
        self.pelvis_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.plate: RigidObject = env.scene[cfg.plate_asset_name]
        self.default_pose = torch.tensor(cfg.default_pose, device=env.device).repeat(env.num_envs, 1)
        
        self._offset = torch.zeros(env.num_envs, 3, device=env.device)
         
        self.pose_command_b = torch.zeros(env.num_envs, 3, device=env.device)
        self.pose_command_w = torch.zeros(env.num_envs, 3, device=env.device)

        self.delay_steps = getattr(cfg, 'delay_steps', 0)
        self.steps_since_reset = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        
        
        
        self.metrics["position_error"] = torch.zeros(env.num_envs, device=env.device)


    def __str__(self) -> str:
        msg = "Plate Pose Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self.pose_command_b)
        result[mask] = self.pose_command_b[mask]
        return result
    
    @property
    def offset(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self._offset)
        result[mask] = self._offset[mask]
        return result

    @property
    def command_world(self) -> torch.Tensor:
        mask = self.steps_since_reset >= self.delay_steps
        result = torch.zeros_like(self.pose_command_w)
        result[mask] = self.pose_command_w[mask]
        return result
    
    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = slice(None)
        self.steps_since_reset[env_ids] = 0
        return super().reset(env_ids)
    
    def compute(self, dt: float):
        self.steps_since_reset += 1
        super().compute(dt)

    def _update_command(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self._offset[env_ids, 0] = r.uniform_(*self.cfg.ranges.offset_x)
        self._offset[env_ids, 1] = r.uniform_(*self.cfg.ranges.offset_y)
        self._offset[env_ids, 2] = r.uniform_(*self.cfg.ranges.offset_z)
        
        
        self.pose_command_b[env_ids] = self.default_pose[env_ids] + self._offset[env_ids]
    
    def _update_metrics(self):
        """Update metrics for the plate pose tracking."""
        # Get pelvis pose in world frame
        pelvis_pos_w = self.robot.data.body_pos_w[:, self.pelvis_idx]
        pelvis_quat_w = self.robot.data.body_quat_w[:, self.pelvis_idx]
        
        # Transform command from pelvis frame to world frame
        # Apply rotation to the command offset
        offset_world = quat_apply(pelvis_quat_w, self.pose_command_b)
        self.pose_command_w = pelvis_pos_w + offset_world
        
        # Get current plate position in world frame
        plate_pos_w = self.plate.data.root_pos_w
        
        # Compute position error
        position_error = self.pose_command_w - plate_pos_w
        self.metrics["position_error"] = torch.norm(position_error, dim=-1)
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], orientations=None)
        # -- current body pose
        plate_pos_w = self.plate.data.root_pos_w
        self.current_pose_visualizer.visualize(plate_pos_w[:, :3], orientations=None)

    


@configclass
class PlatePoseCommandCfg(CommandTermCfg):
    class_type: type = PlatePoseCommand

    robot_asset_name: str = MISSING
    """Name of the robot asset in the environment for which the commands are generated."""

    body_name: str = "pelvis"
    """Name of the body in the robot asset for which the commands are generated."""

    plate_asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    default_pose: list[float] = PLATE_OFFSET
    """Default pose for the plate pose command."""

    delay_steps: int = 0
    """The number of steps to delay the command."""

    @configclass
    class Ranges:
        offset_x: tuple[float, float] = MISSING
        offset_y: tuple[float, float] = MISSING
        offset_z: tuple[float, float] = MISSING
       
    ranges: Ranges = Ranges()


    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    