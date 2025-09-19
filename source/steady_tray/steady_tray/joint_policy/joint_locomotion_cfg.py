from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from steady_tray.assets import G1_CIRCLE_TRAY_CFG
from steady_tray.cfgs import G1RobotPlateEventCfg, CommandsCfg, JointLocomotionObservationsCfg, JointActionsCfg, G1RobotPlateSceneCfg





@configclass
class G1JointLocomotionEnvCfg(G1RobotPlateSceneCfg):
    """ G1 Joint Locomotion Environment Configuration """



    # MDP configuration
    # NOTE: Remember to update these if any updates are made to env
    observation_space = {
        # upper body
        "upper_body_actor_obs": 480,
        "upper_body_critic_obs": 480 + 15,
        # lower body
        "lower_body_actor_obs": 480,
        "lower_body_critic_obs": 480 + 15,
    }
    action_dim= {
        "upper_body": 14,
        "lower_body": 15,
    }

    # body keys
    body_keys = ['upper_body', 'lower_body']
    


    # robot configuration
    robot: ArticulationCfg = G1_CIRCLE_TRAY_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    tray_holder_name = ['left_tray_holder_link', 'right_tray_holder_link']
    

    # events
    events: G1RobotPlateEventCfg = G1RobotPlateEventCfg()

    # command
    commands: CommandsCfg = CommandsCfg()

    # actions
    actions: JointActionsCfg = JointActionsCfg()

    # observations
    observations: JointLocomotionObservationsCfg = JointLocomotionObservationsCfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)


