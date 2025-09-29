from dataclasses import MISSING
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, FrameTransformerCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from steady_tray.assets import G1_CFG
from steady_tray.assets import ASSETS_DATA_DIR, PLATE_OFFSET
from isaaclab.envs.common import ViewerCfg
import isaaclab.terrains as terrain_gen

@configclass
class G1RobotSceneCfg(DirectRLEnvCfg):

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
            gpu_max_rigid_patch_count = 30 * 2**15
        ),
    )

    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        eye=[2.5, -2.0, 1.8],
        lookat=[0.0, 0.0, 0.0],
    )


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
    robot: ArticulationCfg = G1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, track_air_time=True, update_period=sim.dt
    )
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        #attach_yaw_only=True,
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
    camera_names = "d435_link"
    left_ee_name = "left_tray_holder_link"
    right_ee_name = "right_tray_holder_link"


    # gait phase
    gait_period = 0.8
    phase_offset = 0.5
    stance_phase_threshold = 0.55

    # action space
    action_space = 29
    action_scale = 0.25
    state_space = 0
    obs_history_length = 5


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=2.5, replicate_physics=True)
    

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


    # observation space
    observation_space: dict[str, int] = MISSING
    # action dim
    action_dim: dict[str, int] = MISSING

    # body keys
    body_keys: list[str] = MISSING


    # events
    events: object = MISSING

    # command
    commands: object = MISSING

    # actions
    actions: object | None = None

    # observations
    observations: object = MISSING



@configclass
class G1RobotPlateSceneCfg(G1RobotSceneCfg):

    # plate configuration
    plate_offset = PLATE_OFFSET # x, y, z offset from pelvis
    plate_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Plate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSETS_DATA_DIR}/tray_flat.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    plate_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Plate", 
        update_period=0.005, 
        track_air_time=True,
        history_length=3,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/left_tray_holder_link", "/World/envs/env_.*/Robot/right_tray_holder_link"],
    )

    # tray holder transform
    left_tray_holder_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/left_tray_holder_link",
        #debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_tray_holder_link",
                name="right_tray_holder",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Plate",
                name="plate",
            ),
        ],
    )

    right_tray_holder_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/right_tray_holder_link",
        #debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_tray_holder_link",
                name="left_tray_holder",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Plate",
                name="plate",
            ),
        ],
    )


@configclass
class G1RobotObjectSceneCfg(G1RobotPlateSceneCfg):

    # object configuration
    obj_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.10,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    object_contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Object", 
        update_period=0.005, 
        track_pose=True,
        track_air_time=False,
        filter_prim_paths_expr=["/World/envs/env_.*/Plate"],
        history_length=3,
    )

    # object tray transform
    object_tray_transform: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Plate",
        #debug_vis=True,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Object",
                name="object",
            ),
        ],
    )

