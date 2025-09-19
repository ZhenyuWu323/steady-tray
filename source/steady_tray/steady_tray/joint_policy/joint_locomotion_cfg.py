import math
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from . import mdp
from steady_tray.assets import G1_CIRCLE_TRAY_CFG
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen
from steady_tray.cfgs import CircleTrayEventCfg, CommandsCfg, JointLocomotionObservationsCfg, JointActionsCfg





@configclass
class G1JointLocomotionEnvCfg(DirectRLEnvCfg):
    """ G1 Joint Locomotion Environment Configuration """


    # simulation configuration
    episode_length_s = 20.0
    decimation = 4
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count = 10 * 2**15
        ),
    )
    body_keys = ['upper_body', 'lower_body']

    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        eye=[2.5, -2.0, 1.8],
        lookat=[0.0, 0.0, 0.0],
    )


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
    action_space = 29
    action_scale = 0.25
    state_space = 0
    obs_history_length = 5



    # terrain configuration
    terrain_generator_cfg = terrain_gen.TerrainGeneratorCfg(
        size=(8.0, 8.0),
        border_width=20.0,
        num_rows=9,
        num_cols=21,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        difficulty_range=(0.0, 1.0),
        use_cache=False,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        },
        curriculum=True,
    )


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_generator_cfg,
        max_init_terrain_level=terrain_generator_cfg.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # lights
    sky_light_cfg = sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",)

    # robot configuration
    robot: ArticulationCfg = G1_CIRCLE_TRAY_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, update_period=sim.dt
    )
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        update_period=sim.dt * decimation,
    )
    reference_body = "torso_link"

    arm_names = [".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",]
    
    
    waist_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

    hips_names = [".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"]

    feet_names = [".*_ankle_pitch_joint", ".*_ankle_roll_joint"]

    lower_body_names = waist_names + hips_names + feet_names
    upper_body_names = arm_names 
    feet_body_name = ".*_ankle_roll_link"
    pelvis_names = "pelvis"
    hand_names = "^(left|right)_(hand|palm|base|thumb|index|middle|ring|little).*$"
    not_hand_names = "^(?!.*_(hand|palm|base|thumb|index|middle|ring|little)).*$"
    #plate_name = "plate"

    # gait phase
    gait_period = 0.8
    phase_offset = 0.5
    stance_phase_threshold = 0.55

    # events
    events: EventCfg = EventCfg()

    # command
    commands: CommandsCfg = CommandsCfg()

    # actions
    actions: ActionsCfg = ActionsCfg()

    # observations
    observations: ObservationsCfg = ObservationsCfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=2.5, replicate_physics=True)

    # reward scales
    reward_scales = {
        "track_line_vel_xy":1.0,
        "track_ang_vel_z":0.5,
        "alive": 0.15,
        "penalty_lin_vel_z":-2.0,
        "penalty_ang_vel_xy":-0.05,
        "penalty_lower_body_dof_vel":-0.001,
        "penalty_lower_body_dof_acc": -2.5e-7,
        "penalty_lower_body_action_rate": -0.05,
        "penalty_lower_body_dof_pos_limits": -5.0,
        "penalty_dof_pos_waist": -1.0,
        "penalty_dof_pos_hips": -1.0,
        "penalty_flat_orientation": -5.0,
        "penalty_base_height": -10.0,
        "gait_phase_reward": 0.5,
        "feet_slide": -0.2,
        "feet_clearance": 1.0,


        # upper body
        "tracking_upper_body_dof_pos": 0.5,
        "penalty_upper_body_dof_torques": -1e-5,
        "penalty_upper_body_dof_acc": -2.5e-7,
        "penalty_upper_body_dof_pos_limits": -5.0,
        "penalty_upper_body_dof_action_rate": -0.05,
        "penalty_upper_body_dof_vel": -1e-3,
        #"penalty_upper_body_termination": -100.0,
    }

    # observation scales
    obs_scales = {
        "root_lin_vel_b": 2.0,
        "root_ang_vel_b": 0.25,
        "projected_gravity_b": 1.0,
        "dof_pos": 1.0,
        "dof_vel": 0.05,
    }

    # clips
    clip_action = 100
    clip_observation = 100

    # terminations
    termination_height = 0.5
    

    # target base height
    target_base_height = 0.78

    # target feet height
    target_feet_height = 0.12

    # knee joint threshold
    knee_joint_threshold = 0.2

    # sdk joint sequence
    sdk_joint_sequence = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    # object configuration
    plate_offset = [0.42, 0.0, 0.11256] # x, y, z offset from pelvis
    plate_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plate",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,
            height=0.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            #visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

