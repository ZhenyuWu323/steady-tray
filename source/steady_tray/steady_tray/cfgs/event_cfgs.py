from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from steady_tray import mdp
from steady_tray.assets import PLATE_OFFSET

@configclass
class G1RobotEventCfg:
    """Configuration for events."""

    # startup
    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )


    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )



@configclass
class G1RobotPlateEventCfg(G1RobotEventCfg):
    """Configuration for events."""

    # startup
    plate_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("plate", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_plate_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("plate", body_names=".*"),
            "mass_distribution_params": (0.5, 1.0),
            "operation": "abs",
        },
    )

    # reset
    reset_plate = EventTerm(
        func=mdp.reset_plate_object_state,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "plate_asset_cfg": SceneEntityCfg("plate"),
            "plate_offset": PLATE_OFFSET,
            "plate_xy_rand_radius": 0.00,
        },
    )





@configclass
class G1RobotObjectEventCfg(G1RobotPlateEventCfg):
    """Configuration for events."""

    # startup
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "mass_distribution_params": (0.05, 0.15),
            "operation": "abs",
        },
    )

    set_object_com = EventTerm(
        func=mdp.randomize_object_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.02, 0.02)},
        },
    )


    # reset
    reset_object = EventTerm(
        func=mdp.reset_plate_object_state,
        mode="reset",
        params={
            "robot_asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "plate_asset_cfg": SceneEntityCfg("plate"),
            "object_asset_cfg": SceneEntityCfg("object"),
            "plate_offset": PLATE_OFFSET,
            "plate_xy_rand_radius": 0.00,
            "object_xy_rand_radius": 0.09,
            "object_z_up": 0.1,
        },
    )
