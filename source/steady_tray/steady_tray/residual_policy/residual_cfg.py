from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from steady_tray.assets import G1_CIRCLE_TRAY_CFG, G1_HOOK_TRAY_CFG
from steady_tray.cfgs import G1RobotObjectEventCfg, G1RobotObjectSceneCfg, ResidualObservationsCfg, ResidualCommandsCfg



@configclass
class G1ResidualEnvCfg(G1RobotObjectSceneCfg):
    """ G1 Residual Environment Configuration """

    # MDP configuration
    # NOTE: Remember to update these if any updates are made to env
    observation_space = {
        # base action
        "actor_obs": 471,
        "critic_obs": 471 + 15,
        # residual action
        "residual_actor_obs": 471 + 130,
        "residual_critic_obs": 471 + 15 + 130,
        "encoder_obs": 130,
    }
    encoder_output_dim = 32
    action_dim= {
        "upper_body": 14,
        "lower_body": 15,
    }

    # body keys
    body_keys = ['upper_body', 'lower_body', 'residual_whole_body']
    


    # robot configuration
    robot: ArticulationCfg = G1_HOOK_TRAY_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    tray_holder_name = ['left_tray_holder_link', 'right_tray_holder_link']
    

    # events
    events: G1RobotObjectEventCfg = G1RobotObjectEventCfg()

    # command
    commands: ResidualCommandsCfg = ResidualCommandsCfg()

    # observations
    observations: ResidualObservationsCfg = ResidualObservationsCfg()


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)






