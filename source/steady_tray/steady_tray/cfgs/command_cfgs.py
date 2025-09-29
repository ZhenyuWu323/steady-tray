from isaaclab.utils import configclass
from steady_tray import mdp
from .env_cfgs import G1RobotObjectSceneCfg
from steady_tray.assets import PLATE_OFFSET




@configclass
class JointCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        delay_steps=0,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )

    plate_pose = mdp.PlatePoseCommandCfg(
        robot_asset_name="robot",
        body_name="pelvis",
        plate_asset_name="plate",
        default_pose=PLATE_OFFSET,
        ranges=mdp.PlatePoseCommandCfg.Ranges(
            offset_x=(0, 0),
            offset_y=(-0.1, 0.1),
            offset_z=(0,0),
        ),
        delay_steps=0,
        resampling_time_range=(5.0, 15.0),
        debug_vis=True,
    )


@configclass
class ResidualCommandsCfg:
    """Command specifications for the MDP."""
    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        delay_steps=10, # 10 steps to delay the command
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.2, 0.2)
        ),
    )

    plate_pose = mdp.PlatePoseCommandCfg(
        robot_asset_name="robot",
        body_name="pelvis",
        plate_asset_name="plate",
        default_pose=PLATE_OFFSET,
        ranges=mdp.PlatePoseCommandCfg.Ranges(
            offset_x=(0, 0),
            offset_y=(-0.1, 0.1),
            offset_z=(0,0),
        ),
        delay_steps=10, # 10 steps to delay the command
        resampling_time_range=(5.0, 15.0),
        debug_vis=True,
    )