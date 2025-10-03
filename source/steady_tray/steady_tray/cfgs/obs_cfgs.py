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



@configclass
class ResidualObservationsCfg:

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
        last_action = ObsTerm(func=mdp.base_action, history_length=5)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    actor_obs: PolicyCfg = PolicyCfg()

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
        last_action = ObsTerm(func=mdp.base_action, history_length=5)
        def __post_init__(self):
           self.concatenate_terms = True

    # privileged observations
    critic_obs: CriticCfg = CriticCfg()



    @configclass
    class ResidualActorCfg(ObsGroup):
        """Observations for residual group."""

        # observation terms (order preserved)
        last_action = ObsTerm(func=mdp.residual_action)

        # plate observations
        # plate_position_in_robot = ObsTerm(func=mdp.plate_position_robot_frame,noise=Unoise(n_min=-0.01, n_max=0.01))
        # plate_orientation_in_robot = ObsTerm(func=mdp.plate_orientation_robot_frame, noise=UniformNoiseQuatCfg(min=-0.005, max=0.005))
        # plate_twist_in_robot = ObsTerm(func=mdp.plate_twist_robot_frame, noise=Unoise(n_min=-0.2, n_max=0.2))
        
        #object observations
        object_position_in_plate = ObsTerm(func=mdp.object_position_in_plate_frame, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_orientation_in_plate = ObsTerm(func=mdp.object_orientation_in_plate_frame, noise=UniformNoiseQuatCfg(min=-0.005, max=0.005))
        object_twist_in_plate = ObsTerm(func=mdp.object_twist_in_plate_frame, noise=Unoise(n_min=-0.2, n_max=0.2))
        object_physics = ObsTerm(func=mdp.object_physics)
        object_mass = ObsTerm(func=mdp.object_mass)
        object_projected_gravity = ObsTerm(func=mdp.object_projected_gravity)
        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True


    @configclass
    class ResidualCriticCfg(ObsGroup):
        """Observations for residual group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=5)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, history_length=5)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, history_length=5)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        plate_position_offset = ObsTerm(func=mdp.plate_position_offset, params={"command_name": "plate_pose"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, history_length=5)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, history_length=5)
        last_action = ObsTerm(func=mdp.residual_action, history_length=5)

        # plate observations
        # plate_position_in_robot = ObsTerm(func=mdp.plate_position_robot_frame, history_length=5)
        # plate_orientation_in_robot = ObsTerm(func=mdp.plate_orientation_robot_frame, history_length=5)
        # plate_twist_in_robot = ObsTerm(func=mdp.plate_twist_robot_frame, history_length=5)
        
        #object observations
        object_position_in_plate = ObsTerm(func=mdp.object_position_in_plate_frame, history_length=5)
        object_orientation_in_plate = ObsTerm(func=mdp.object_orientation_in_plate_frame, history_length=5)
        object_twist_in_plate = ObsTerm(func=mdp.object_twist_in_plate_frame, history_length=5)
        object_physics = ObsTerm(func=mdp.object_physics, history_length=5)
        object_mass = ObsTerm(func=mdp.object_mass, history_length=5)
        object_projected_gravity = ObsTerm(func=mdp.object_projected_gravity, history_length=5)

        def __post_init__(self):
            self.concatenate_terms = True

    residual_actor_obs: ResidualActorCfg = ResidualActorCfg()
    residual_critic_obs: ResidualCriticCfg = ResidualCriticCfg()