from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from steady_tray.assets import G1_CIRCLE_TRAY_CFG
from steady_tray.cfgs import G1RobotPlateEventCfg, CommandsCfg, JointActionsCfg, G1RobotPlateSceneCfg
from steady_tray.cfgs import CurriculumCfg
from steady_tray import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from steady_tray.utils import *
from isaaclab.managers import SceneEntityCfg
from .joint_locomotion_cfg import G1JointLocomotionEnvCfg
from isaaclab.sensors import FrameTransformerCfg
from steady_tray.cfgs import JointPolicyObservationsCfg





@configclass
class G1JointPlateBalanceEnvCfg(G1JointLocomotionEnvCfg):
    """ G1 Joint Plate Balance Environment Configuration """

    # MDP configuration
    # NOTE: Remember to update these if any updates are made to env
    observation_space = {
        # upper body
        "upper_body_actor_obs": 471,
        "upper_body_critic_obs": 471 + 15,
        # lower body
        "lower_body_actor_obs": 471,
        "lower_body_critic_obs": 471 + 15,
    }
    

    # observations
    observations: JointPolicyObservationsCfg = JointPolicyObservationsCfg()

    

   
    
