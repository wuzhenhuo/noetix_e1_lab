# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Ablation: + Adaptive Tracking
# Change vs walkrun_cfg.py: replaced track_lin_vel_xy_exp with
# track_lin_vel_xy_adaptive_exp so the tracking std scales with the number of
# feet in contact (double-stance strict, single-stance lenient, flight most lenient).

import glob
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from NoetixE1.assets import ASSET_DIR
from NoetixE1.terrains import GRAVEL_TERRAINS_CFG

from . import mdp

##
# Pre-defined configs
##

from NoetixE1.assets.robots.e1.e1 import E1_24DOF_CFG, E1_24DOF_ACTION_SCALE  # isort:skip


##
# Scene definition
##


@configclass
class E1SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
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

    # robot
    robot: ArticulationCfg = E1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.VelocityWithMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.2,
        debug_vis=True,
        ranges=mdp.VelocityWithMotionCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), zero_prob=[0.2, 0.8, 0.2]
        ),
    )

    # motion_commands = mdp.MotionCommandCfg(
    #     asset_name="robot",
    #     rel_standing_envs=0.2,
    #     resampling_time_range=(10.0, 15.0),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
    joint_pos.scale = E1_24DOF_ACTION_SCALE


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity", "obs_command_type": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = True
            self.history_length = 5
            self.flatten_history_dim = False
        motion_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity", "obs_command_type": "motion_command"})

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity", "obs_command_type": "base_velocity"})
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        motion_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity", "obs_command_type": "motion_command"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class RndObsCfg(ObsGroup):
        """Observations for rnd_state group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        key_pos = ObsTerm(func=mdp.amp_key_pos)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class AmpObsCfg(ObsGroup):
        """Observations for discriminator group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.amp_joint_pos)
        key_pos = ObsTerm(func=mdp.amp_key_pos)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_vel = ObsTerm(func=mdp.amp_joint_vel)
        feet_contact = ObsTerm(
            func=mdp.current_feet_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            },
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False
            self.flatten_history_dim = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    # rnd_state: RndObsCfg = RndObsCfg()
    discriminator: AmpObsCfg = AmpObsCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.05, 0.05)},
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["waist_yaw_link"]),
            "mass_distribution_params": (-5, 5),
            "operation": "add",
        },
    )

    reset_robot_states = EventTerm(
        func=mdp.reset_root_state_amp,
        mode="reset",
        params={
            "reference_state_initialization": True,
            "prob_rsi": 0.8,
            "root_pos_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "root_vel_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "joint_pos_range": (-0.2, 0.2),
            "joint_vel_range": (-1.0, 1.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    randomize_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.01, 0.1),
            "operation": "abs",
        }
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 15.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.2, 0.2),
                "roll": (-0.78, 0.78),
                "pitch": (-0.78, 0.78),
                "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ------------------------------------------- Tracking Velocity -------------------------------------------
    # [Adaptive Tracking] std now scales with number of feet in contact
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_adaptive_exp,
        weight=1.2,
        params={
            "base_std": 0.5,
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
        },
    )
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    # ------------------------------------------------- Base --------------------------------------------------
    flat_orientation_exp = RewTerm(func=mdp.flat_orientation_exp, weight=0.5, params={"std": 0.5})
    # ------------------------------------------------- Feet --------------------------------------------------
    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 0.3,
        },
    )
    feet_force = RewTerm(
        func=mdp.feet_force,
        weight=-1e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 400,
            "max_reward": 600,
        },
    )
    # ------------------------------------------------- Joint -------------------------------------------------
    joint_deviation_leg = RewTerm(
        func=mdp.joint_deviation_l1_with_zero_flag,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_ankle_roll_joint"],
            )
        },
    )
    joint_deviation_upbody = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_yaw_joint",
                    "waist_yaw_joint",
                ],
            )
        },
    )
    joint_deviation_standing = RewTerm(
        func=mdp.joint_deviation_l1_when_standing,
        weight=-0.25,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_.*_joint",
                    ".*_elbow_.*_joint",
                    "waist_.*_joint",
                ],
            )
        },
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    # ----------------------------------------------- Regulization -----------------------------------------------
    energy = RewTerm(func=mdp.energy, weight=-1e-3)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Fall Down （Do not use in plane）
    # fall_down = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    # (3) Bad Orientation
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5})


##
# Environment configuration
##


@configclass
class WalkRunFlatEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: E1SceneCfg = E1SceneCfg(num_envs=4096, env_spacing=3.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 10
        self.episode_length_s = 20.0
        # viewer settings
        # self.viewer.eye = (1.5, 1.5, 1.5)
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
        # simulation settings
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_sensor.update_period = self.sim.dt
        # self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # ----------------------------------------------AMP----------------------------------------------
        self.motion_loader_name = "MotionLoaderE1"
        # self.motion_files = sorted(glob.glob(f"{ASSET_DIR}/datasets/walkrun/e1/*"))
        self.motion_files = sorted(glob.glob(f"{ASSET_DIR}/datasets/boxing/*"))
        self.num_skill_labels = len(self.motion_files)
        self.reference_observation_horizon = 4
        self.num_preload_transitions = 500000
        self.joint_pose_size = 23
        self.observations.discriminator.history_length = self.reference_observation_horizon
