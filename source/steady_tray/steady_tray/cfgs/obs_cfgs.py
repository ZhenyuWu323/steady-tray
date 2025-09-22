from isaaclab.utils import configclass
from steady_tray import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from steady_tray.utils import *



@configclass
class JointPolicyObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=5)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), history_length=5)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        plate_position_offset = ObsTerm(func=mdp.plate_position_offset, params={"command_name": "plate_pose"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=5)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=5)
        last_action = ObsTerm(func=mdp.last_action, history_length=5)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    upper_body_actor_obs: PolicyCfg = PolicyCfg()
    lower_body_actor_obs: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        plate_position_offset = ObsTerm(func=mdp.plate_position_offset, params={"command_name": "plate_pose"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5)
        last_action = ObsTerm(func=mdp.last_action, history_length=5)
        def __post_init__(self):
           self.concatenate_terms = True

    # privileged observations
    upper_body_critic_obs: CriticCfg = CriticCfg()
    lower_body_critic_obs: CriticCfg = CriticCfg()