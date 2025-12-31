# # BSD 3-Clause License
# # Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# # All rights reserved.

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets.articulation import ArticulationCfg
# from NoetixE1.assets import ASSET_DIR

# ARMATURE_4315 = 0.033048
# ARMATURE_4340 = 0.032
# ARMATURE_8112 = 0.04752756
# ARMATURE_10020 = 0.068575968

# NATURAL_FREQ = 4 * 2.0 * 3.1415926535  # 4Hz

# STIFFNESS_4315 = ARMATURE_4315 * NATURAL_FREQ**2
# STIFFNESS_4340 = ARMATURE_4340 * NATURAL_FREQ**2
# STIFFNESS_8112 = ARMATURE_8112 * NATURAL_FREQ**2
# STIFFNESS_10020 = ARMATURE_10020 * NATURAL_FREQ**2

# DAMPING_4315 = 2.0 * ARMATURE_4315 * NATURAL_FREQ
# DAMPING_4340 = 2.0 * ARMATURE_4340 * NATURAL_FREQ
# DAMPING_8112 = 2.0 * ARMATURE_8112 * NATURAL_FREQ
# DAMPING_10020 = 2.0 * ARMATURE_10020 * NATURAL_FREQ

# E1_23DOF_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         fix_base=False,
#         asset_path=f"{ASSET_DIR}/robots/e1/urdf/e1_23dof.urdf",
#         activate_contact_sensors=True,
#         replace_cylinders_with_capsules=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.85),
#         joint_pos={
#             "l_leg_hip_yaw_joint": 0.0,
#             "l_leg_hip_roll_joint": 0.0,
#             "l_leg_hip_pitch_joint": -0.1495,
#             "l_leg_knee_joint": 0.3215,
#             "l_leg_ankle_pitch_joint": -0.1720,
#             "l_leg_ankle_roll_joint": 0.0,
#             "r_leg_hip_yaw_joint": 0.0,
#             "r_leg_hip_roll_joint": 0.0,
#             "r_leg_hip_pitch_joint": -0.1495,
#             "r_leg_knee_joint": 0.3215,
#             "r_leg_ankle_pitch_joint": -0.1720,
#             "r_leg_ankle_roll_joint": 0.0,
#             "waist_yaw_joint": 0.0,
#             "l_arm_shoulder_pitch_joint": 0.0,
#             "l_arm_shoulder_roll_joint": 0.2618,
#             "l_arm_shoulder_yaw_joint": 0.0,
#             "l_arm_elbow_pitch_joint": 0.0,
#             "l_arm_elbow_yaw_joint": 0.0,
#             "r_arm_shoulder_pitch_joint": 0.0,
#             "r_arm_shoulder_roll_joint": -0.2618,
#             "r_arm_shoulder_yaw_joint": 0.0,
#             "r_arm_elbow_pitch_joint": 0.0,
#             "r_arm_elbow_yaw_joint": 0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_yaw_joint",
#                 ".*_hip_roll_joint",
#                 ".*_hip_pitch_joint",
#                 ".*_knee_joint",
#             ],
#             effort_limit_sim={
#                 ".*_hip_yaw_joint": 80.0,
#                 ".*_hip_roll_joint": 80.0,
#                 ".*_hip_pitch_joint": 100.0,
#                 ".*_knee_joint": 100.0,
#             },
#             velocity_limit_sim={
#                 ".*_hip_yaw_joint": 10.0,
#                 ".*_hip_roll_joint": 10.0,
#                 ".*_hip_pitch_joint": 12.0,
#                 ".*_knee_joint": 12.0,
#             },
#             stiffness={
#                 ".*_hip_yaw_joint": STIFFNESS_8112 * 2,
#                 ".*_hip_roll_joint": STIFFNESS_8112 * 2,
#                 ".*_hip_pitch_joint": STIFFNESS_10020 * 2,
#                 ".*_knee_joint": STIFFNESS_10020 * 2,
#             },
#             damping={
#                 ".*_hip_yaw_joint": DAMPING_8112 * 2,
#                 ".*_hip_roll_joint": DAMPING_8112 * 2,
#                 ".*_hip_pitch_joint": DAMPING_10020 * 1.45,
#                 ".*_knee_joint": DAMPING_10020 * 1.45,
#             },
#             armature={
#                 ".*_hip_yaw_joint": ARMATURE_8112,
#                 ".*_hip_roll_joint": ARMATURE_8112,
#                 ".*_hip_pitch_joint": ARMATURE_10020,
#                 ".*_knee_joint": ARMATURE_10020,
#             },
#         ),
#         "waist": ImplicitActuatorCfg(
#             effort_limit_sim=60.0,
#             velocity_limit_sim=12.0,
#             joint_names_expr=["waist_yaw_joint"],
#             stiffness=STIFFNESS_4315 * 2,
#             damping=DAMPING_4315 * 2,
#             armature=ARMATURE_4315,
#         ),
#         "feet": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_ankle_pitch_joint",
#                 ".*_ankle_roll_joint",
#             ],
#             effort_limit_sim=70.0,
#             velocity_limit_sim=10.0,
#             stiffness=STIFFNESS_4315 * 1.05,
#             damping=DAMPING_4315 * 1.05,
#             armature=ARMATURE_4315,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_shoulder_pitch_joint",
#                 ".*_shoulder_roll_joint",
#                 ".*_shoulder_yaw_joint",
#                 ".*_elbow_pitch_joint",
#                 ".*_elbow_yaw_joint",
#             ],
#             effort_limit_sim=20.0,
#             velocity_limit_sim=10.0,
#             stiffness={
#                 ".*_shoulder_pitch_joint": STIFFNESS_4340 * 2,
#                 ".*_shoulder_roll_joint": STIFFNESS_4340 * 2,
#                 ".*_shoulder_yaw_joint": STIFFNESS_4340 * 2,
#                 ".*_elbow_pitch_joint": STIFFNESS_4340 * 1.2,
#                 ".*_elbow_yaw_joint": STIFFNESS_4340 * 1.2,
#             },
#             damping={
#                 ".*_shoulder_pitch_joint": DAMPING_4340 * 2,
#                 ".*_shoulder_roll_joint": DAMPING_4340 * 2,
#                 ".*_shoulder_yaw_joint": DAMPING_4340 * 2,
#                 ".*_elbow_pitch_joint": DAMPING_4340 * 1.2,
#                 ".*_elbow_yaw_joint": DAMPING_4340 * 1.2,
#             },
#             armature=ARMATURE_4340,
#         ),
#     },
# )

# E1_23DOF_JOINT_NAMES = [
#     "l_leg_hip_yaw_joint",
#     "l_leg_hip_roll_joint",
#     "l_leg_hip_pitch_joint",
#     "l_leg_knee_joint",
#     "l_leg_ankle_pitch_joint",
#     "l_leg_ankle_roll_joint",
#     "r_leg_hip_yaw_joint",
#     "r_leg_hip_roll_joint",
#     "r_leg_hip_pitch_joint",
#     "r_leg_knee_joint",
#     "r_leg_ankle_pitch_joint",
#     "r_leg_ankle_roll_joint",
#     "waist_yaw_joint",
#     "l_arm_shoulder_pitch_joint",
#     "l_arm_shoulder_roll_joint",
#     "l_arm_shoulder_yaw_joint",
#     "l_arm_elbow_pitch_joint",
#     "l_arm_elbow_yaw_joint",
#     "r_arm_shoulder_pitch_joint",
#     "r_arm_shoulder_roll_joint",
#     "r_arm_shoulder_yaw_joint",
#     "r_arm_elbow_pitch_joint",
#     "r_arm_elbow_yaw_joint",
# ]

# E1_23DOF_ACTION_SCALE = {}
# for a in E1_23DOF_CFG.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     names = a.joint_names_expr
#     if not isinstance(e, dict):
#         e = {n: e for n in names}
#     if not isinstance(s, dict):
#         s = {n: s for n in names}
#     for n in names:
#         if n in e and n in s and s[n]:
#             E1_23DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

# E1_24DOF_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         fix_base=False,
#         asset_path=f"{ASSET_DIR}/robots/e1/urdf/e1_24dof.urdf",
#         activate_contact_sensors=True,
#         replace_cylinders_with_capsules=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.85),
#         joint_pos={
#             "l_leg_hip_yaw_joint": 0.20482691,
#             "l_leg_hip_roll_joint": -0.12772794,
#             "l_leg_hip_pitch_joint": -0.44179791,
#             "l_leg_knee_joint": 0.53102249,
#             "l_leg_ankle_pitch_joint": -0.0850352,
#             "l_leg_ankle_roll_joint": 0.07077004,
#             "r_leg_hip_yaw_joint": -0.12583318,
#             "r_leg_hip_roll_joint": -0.08734387,
#             "r_leg_hip_pitch_joint": -0.59284598,
#             "r_leg_knee_joint": 1.0108006,
#             "r_leg_ankle_pitch_joint": -0.14891951,
#             "r_leg_ankle_roll_joint": 0.13455576,
#             "waist_yaw_joint": -0.03297309,
#             "waist_roll_joint": 0.3116861,
#             "l_arm_shoulder_pitch_joint": 0.3229278,
#             "l_arm_shoulder_roll_joint": 0.55496484,
#             "l_arm_shoulder_yaw_joint": -0.26042867,
#             "l_arm_elbow_pitch_joint": -2.2652061,
#             "l_arm_elbow_yaw_joint": -0.20749661,
#             "r_arm_shoulder_pitch_joint": -0.27206859,
#             "r_arm_shoulder_roll_joint": -0.33009294,
#             "r_arm_shoulder_yaw_joint": 0.80902374,
#             "r_arm_elbow_pitch_joint": -1.69986355,
#             "r_arm_elbow_yaw_joint": 0.00400056,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_yaw_joint",
#                 ".*_hip_roll_joint",
#                 ".*_hip_pitch_joint",
#                 ".*_knee_joint",
#             ],
#             effort_limit_sim={
#                 ".*_hip_yaw_joint": 80.0,
#                 ".*_hip_roll_joint": 80.0,
#                 ".*_hip_pitch_joint": 100.0,
#                 ".*_knee_joint": 100.0,
#             },
#             velocity_limit_sim={
#                 ".*_hip_yaw_joint": 10.0,
#                 ".*_hip_roll_joint": 10.0,
#                 ".*_hip_pitch_joint": 12.0,
#                 ".*_knee_joint": 12.0,
#             },
#             stiffness={
#                 ".*_hip_yaw_joint": STIFFNESS_8112 * 2,
#                 ".*_hip_roll_joint": STIFFNESS_8112 * 2,
#                 ".*_hip_pitch_joint": STIFFNESS_10020 * 2,
#                 ".*_knee_joint": STIFFNESS_10020 * 2,
#             },
#             damping={
#                 ".*_hip_yaw_joint": DAMPING_8112 * 2,
#                 ".*_hip_roll_joint": DAMPING_8112 * 2,
#                 ".*_hip_pitch_joint": DAMPING_10020 * 1.45,
#                 ".*_knee_joint": DAMPING_10020 * 1.45,
#             },
#             armature={
#                 ".*_hip_yaw_joint": ARMATURE_8112,
#                 ".*_hip_roll_joint": ARMATURE_8112,
#                 ".*_hip_pitch_joint": ARMATURE_10020,
#                 ".*_knee_joint": ARMATURE_10020,
#             },
#         ),
#         "waist": ImplicitActuatorCfg(
#             effort_limit_sim=60.0,
#             velocity_limit_sim=12.0,
#             joint_names_expr=["waist_.*_joint"],
#             stiffness=STIFFNESS_4315 * 2,
#             damping=DAMPING_4315 * 2,
#             armature=ARMATURE_4315,
#         ),
#         "feet": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_ankle_pitch_joint",
#                 ".*_ankle_roll_joint",
#             ],
#             effort_limit_sim=70.0,
#             velocity_limit_sim=10.0,
#             stiffness=STIFFNESS_4315 * 1.05,
#             damping=DAMPING_4315 * 1.05,
#             armature=ARMATURE_4315,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_shoulder_pitch_joint",
#                 ".*_shoulder_roll_joint",
#                 ".*_shoulder_yaw_joint",
#                 ".*_elbow_pitch_joint",
#                 ".*_elbow_yaw_joint",
#             ],
#             effort_limit_sim=20.0,
#             velocity_limit_sim=10.0,
#             stiffness={
#                 ".*_shoulder_pitch_joint": STIFFNESS_4340 * 2,
#                 ".*_shoulder_roll_joint": STIFFNESS_4340 * 2,
#                 ".*_shoulder_yaw_joint": STIFFNESS_4340 * 2,
#                 ".*_elbow_pitch_joint": STIFFNESS_4340 * 1.2,
#                 ".*_elbow_yaw_joint": STIFFNESS_4340 * 1.2,
#             },
#             damping={
#                 ".*_shoulder_pitch_joint": DAMPING_4340 * 2,
#                 ".*_shoulder_roll_joint": DAMPING_4340 * 2,
#                 ".*_shoulder_yaw_joint": DAMPING_4340 * 2,
#                 ".*_elbow_pitch_joint": DAMPING_4340 * 1.2,
#                 ".*_elbow_yaw_joint": DAMPING_4340 * 1.2,
#             },
#             armature=ARMATURE_4340,
#         ),
#     },
# )

# E1_24DOF_JOINT_NAMES = [
#     "l_leg_hip_yaw_joint",
#     "r_leg_hip_yaw_joint",
#     "waist_yaw_joint",
#     "l_leg_hip_roll_joint",
#     "r_leg_hip_roll_joint",
#     "waist_roll_joint",
#     "l_leg_hip_pitch_joint",
#     "r_leg_hip_pitch_joint",
#     "l_arm_shoulder_pitch_joint",
#     "r_arm_shoulder_pitch_joint",
#     "l_leg_knee_joint",
#     "r_leg_knee_joint",
#     "l_arm_shoulder_roll_joint",
#     "r_arm_shoulder_roll_joint",
#     "l_leg_ankle_pitch_joint",
#     "r_leg_ankle_pitch_joint",
#     "l_arm_shoulder_yaw_joint",
#     "r_arm_shoulder_yaw_joint",
#     "l_leg_ankle_roll_joint",
#     "r_leg_ankle_roll_joint",
#     "l_arm_elbow_pitch_joint",
#     "r_arm_elbow_pitch_joint",
#     "l_arm_elbow_yaw_joint",
#     "r_arm_elbow_yaw_joint",
# ]

# E1_24DOF_ACTION_SCALE = {}
# for a in E1_24DOF_CFG.actuators.values():
#     e = a.effort_limit_sim
#     s = a.stiffness
#     names = a.joint_names_expr
#     if not isinstance(e, dict):
#         e = {n: e for n in names}
#     if not isinstance(s, dict):
#         s = {n: s for n in names}
#     for n in names:
#         if n in e and n in s and s[n]:
#             E1_24DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from NoetixE1.actuators import DelayedImplicitActuatorCfg
from NoetixE1.assets import ASSET_DIR

ARMATURE_4315 = 0.033048
ARMATURE_4340 = 0.032
ARMATURE_8112 = 0.04752756
ARMATURE_10020 = 0.068575968

NATURAL_FREQ = 4 * 2.0 * 3.1415926535  # 4Hz

STIFFNESS_4315 = ARMATURE_4315 * NATURAL_FREQ**2
STIFFNESS_4340 = ARMATURE_4340 * NATURAL_FREQ**2
STIFFNESS_8112 = ARMATURE_8112 * NATURAL_FREQ**2
STIFFNESS_10020 = ARMATURE_10020 * NATURAL_FREQ**2

DAMPING_4315 = 2.0 * ARMATURE_4315 * NATURAL_FREQ
DAMPING_4340 = 2.0 * ARMATURE_4340 * NATURAL_FREQ
DAMPING_8112 = 2.0 * ARMATURE_8112 * NATURAL_FREQ
DAMPING_10020 = 2.0 * ARMATURE_10020 * NATURAL_FREQ

E1_23DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/robots/e1/urdf/e1_23dof.urdf",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "l_leg_hip_yaw_joint": 0.0,
            "l_leg_hip_roll_joint": 0.0,
            "l_leg_hip_pitch_joint": -0.1495,
            "l_leg_knee_joint": 0.3215,
            "l_leg_ankle_pitch_joint": -0.1720,
            "l_leg_ankle_roll_joint": 0.0,
            "r_leg_hip_yaw_joint": 0.0,
            "r_leg_hip_roll_joint": 0.0,
            "r_leg_hip_pitch_joint": -0.1495,
            "r_leg_knee_joint": 0.3215,
            "r_leg_ankle_pitch_joint": -0.1720,
            "r_leg_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "l_arm_shoulder_pitch_joint": 0.0,
            "l_arm_shoulder_roll_joint": 0.2618,
            "l_arm_shoulder_yaw_joint": 0.0,
            "l_arm_elbow_pitch_joint": 0.0,
            "l_arm_elbow_yaw_joint": 0.0,
            "r_arm_shoulder_pitch_joint": 0.0,
            "r_arm_shoulder_roll_joint": -0.2618,
            "r_arm_shoulder_yaw_joint": 0.0,
            "r_arm_elbow_pitch_joint": 0.0,
            "r_arm_elbow_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 80.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 100.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 10.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_pitch_joint": 14.0,
                ".*_knee_joint": 14.0,
            },
            stiffness={
                ".*_hip_yaw_joint": STIFFNESS_8112 * 2,
                ".*_hip_roll_joint": STIFFNESS_8112 * 2,
                ".*_hip_pitch_joint": STIFFNESS_10020 * 2,
                ".*_knee_joint": STIFFNESS_10020 * 2,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_8112 * 1.5,
                ".*_hip_roll_joint": DAMPING_8112 * 1.5,
                ".*_hip_pitch_joint": DAMPING_10020 * 1.1,
                ".*_knee_joint": DAMPING_10020 * 1.1,
            },
            armature={
                ".*_hip_yaw_joint": ARMATURE_8112,
                ".*_hip_roll_joint": ARMATURE_8112,
                ".*_hip_pitch_joint": ARMATURE_10020,
                ".*_knee_joint": ARMATURE_10020,
            },
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=60.0,
            velocity_limit_sim=12.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_4315 * 2,
            damping=DAMPING_4315 * 2,
            armature=ARMATURE_4315,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=70.0,
            velocity_limit_sim=10.0,
            stiffness=STIFFNESS_4315 * 1.05,
            damping=DAMPING_4315 * 1.05,
            armature=ARMATURE_4315,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_4340 * 2,
                ".*_shoulder_roll_joint": STIFFNESS_4340 * 2,
                ".*_shoulder_yaw_joint": STIFFNESS_4340 * 2,
                ".*_elbow_pitch_joint": STIFFNESS_4340 * 1.2,
                ".*_elbow_yaw_joint": STIFFNESS_4340 * 1.2,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_4340 * 2,
                ".*_shoulder_roll_joint": DAMPING_4340 * 2,
                ".*_shoulder_yaw_joint": DAMPING_4340 * 2,
                ".*_elbow_pitch_joint": DAMPING_4340 * 1.2,
                ".*_elbow_yaw_joint": DAMPING_4340 * 1.2,
            },
            armature=ARMATURE_4340,
        ),
    },
)

E1_23DOF_ACTION_SCALE = {}
for a in E1_23DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            E1_23DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

E1_24DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/robots/e1/urdf/e1_24dof.urdf",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            # "l_leg_hip_yaw_joint": 0.7151, "r_leg_hip_yaw_joint": -0.1499,
            # "waist_yaw_joint": 0.2685, 
            # "l_leg_hip_roll_joint": 0.0447, "r_leg_hip_roll_joint": -0.2432,
            # "waist_roll_joint": 0.3168,
            # "l_leg_hip_pitch_joint": -1.0529, "r_leg_hip_pitch_joint": -0.7069,
            # "l_arm_shoulder_pitch_joint": -0.4301, "r_arm_shoulder_pitch_joint": -0.5439,
            # "l_leg_knee_joint": 1.0411, "r_leg_knee_joint": 0.9334,
            # "l_arm_shoulder_roll_joint": 0.8508, "r_arm_shoulder_roll_joint": -0.1046,
            # "l_leg_ankle_pitch_joint": -0.1036, "r_leg_ankle_pitch_joint": -0.4245,
            # "l_arm_shoulder_yaw_joint": 0.3434, "r_arm_shoulder_yaw_joint": 1.0730,
            # "l_leg_ankle_roll_joint": -0.0569, "r_leg_ankle_roll_joint": 0.1104,
            # "l_arm_elbow_pitch_joint": -1.7162, "r_arm_elbow_pitch_joint": -2.0828,
            # "l_arm_elbow_yaw_joint": -0.2014, "r_arm_elbow_yaw_joint": 0.0340,
            "l_leg_hip_yaw_joint": 0.0,
            "l_leg_hip_roll_joint": 0.0,
            "l_leg_hip_pitch_joint": -0.1495,
            "l_leg_knee_joint": 0.3215,
            "l_leg_ankle_pitch_joint": -0.1720,
            "l_leg_ankle_roll_joint": 0.0,
            "r_leg_hip_yaw_joint": 0.0,
            "r_leg_hip_roll_joint": 0.0,
            "r_leg_hip_pitch_joint": -0.1495,
            "r_leg_knee_joint": 0.3215,
            "r_leg_ankle_pitch_joint": -0.1720,
            "r_leg_ankle_roll_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "l_arm_shoulder_pitch_joint": 0.0,
            "l_arm_shoulder_roll_joint": 0.2618,
            "l_arm_shoulder_yaw_joint": 0.0,
            "l_arm_elbow_pitch_joint": 0.0,
            "l_arm_elbow_yaw_joint": 0.0,
            "r_arm_shoulder_pitch_joint": 0.0,
            "r_arm_shoulder_roll_joint": -0.2618,
            "r_arm_shoulder_yaw_joint": 0.0,
            "r_arm_elbow_pitch_joint": 0.0,
            "r_arm_elbow_yaw_joint": 0.0,
            # "l_leg_hip_yaw_joint": 0.3382284343242645,
            # "l_leg_hip_roll_joint": -0.053336139768362045,
            # "l_leg_hip_pitch_joint": -0.755505383014679,
            # "l_leg_knee_joint": 0.9783093929290771,
            # "l_leg_ankle_pitch_joint": -0.12228915095329285,
            # "l_leg_ankle_roll_joint": 0.009284205734729767,
            # "r_leg_hip_yaw_joint": -0.04868125170469284,
            # "r_leg_hip_roll_joint": 0.07678689062595367,
            # "r_leg_hip_pitch_joint": -0.3240882456302643,
            # "r_leg_knee_joint": 0.7333481311798096,
            # "r_leg_ankle_pitch_joint": -0.43591752648353577,
            # "r_leg_ankle_roll_joint": -0.0054446132853627205,
            # "waist_roll_joint": -0.028743883594870567,
            # "waist_yaw_joint": 0.13841310143470764,
            # "l_arm_shoulder_pitch_joint": 0.17767517268657684,
            # "l_arm_shoulder_roll_joint": 0.19579221308231354,
            # "l_arm_shoulder_yaw_joint": -0.3707764148712158,
            # "l_arm_elbow_pitch_joint": -2.232343912124634,
            # "l_arm_elbow_yaw_joint": -0.07095427811145782,
            # "r_arm_shoulder_pitch_joint": -0.11448931694030762,
            # "r_arm_shoulder_roll_joint": -0.1885678768157959,
            # "r_arm_shoulder_yaw_joint": 0.676032543182373,
            # "r_arm_elbow_pitch_joint": -2.2305612564086914,
            # "r_arm_elbow_yaw_joint": 0.027833547443151474,
            
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 80.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 100.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 10.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_pitch_joint": 14.0,
                ".*_knee_joint": 14.0,
            },
            stiffness={
                ".*_hip_yaw_joint": STIFFNESS_8112 * 2,
                ".*_hip_roll_joint": STIFFNESS_8112 * 2,
                ".*_hip_pitch_joint": STIFFNESS_10020 * 2,
                ".*_knee_joint": STIFFNESS_10020 * 2,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_8112 * 2,
                ".*_hip_roll_joint": DAMPING_8112 * 2,
                ".*_hip_pitch_joint": DAMPING_10020 * 1.45,
                ".*_knee_joint": DAMPING_10020 * 1.45,
            },
            armature={
                ".*_hip_yaw_joint": ARMATURE_8112,
                ".*_hip_roll_joint": ARMATURE_8112,
                ".*_hip_pitch_joint": ARMATURE_10020,
                ".*_knee_joint": ARMATURE_10020,
            },
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=60.0,
            velocity_limit_sim=12.0,
            joint_names_expr=["waist_.*_joint"],
            stiffness=STIFFNESS_4315 * 2,
            damping=DAMPING_4315 * 2,
            armature=ARMATURE_4315,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=70.0,
            velocity_limit_sim=10.0,
            stiffness=STIFFNESS_4315 * 1.05,
            damping=DAMPING_4315 * 1.05,
            armature=ARMATURE_4315,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_yaw_joint",
            ],
            effort_limit_sim=20.0,
            velocity_limit_sim=10.0,
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_4340 * 2,
                ".*_shoulder_roll_joint": STIFFNESS_4340 * 2,
                ".*_shoulder_yaw_joint": STIFFNESS_4340 * 2,
                ".*_elbow_pitch_joint": STIFFNESS_4340 * 1.2,
                ".*_elbow_yaw_joint": STIFFNESS_4340 * 1.2,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_4340 * 2,
                ".*_shoulder_roll_joint": DAMPING_4340 * 2,
                ".*_shoulder_yaw_joint": DAMPING_4340 * 2,
                ".*_elbow_pitch_joint": DAMPING_4340 * 1.2,
                ".*_elbow_yaw_joint": DAMPING_4340 * 1.2,
            },
            armature=ARMATURE_4340,
        ),
    },
)

E1_24DOF_ACTION_SCALE = {}
for a in E1_24DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            E1_24DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


# ----------------------------- 12 DOF CFG -----------------------------
E1_12DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        asset_path=f"{ASSET_DIR}/robots/e1/urdf/e1_12dof.urdf",
        activate_contact_sensors=True,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            "l_leg_hip_yaw_joint": 0.0,
            "l_leg_hip_roll_joint": 0.0,
            "l_leg_hip_pitch_joint": -0.1495,
            "l_leg_knee_joint": 0.3215,
            "l_leg_ankle_pitch_joint": -0.1720,
            "l_leg_ankle_roll_joint": 0.0,
            "r_leg_hip_yaw_joint": 0.0,
            "r_leg_hip_roll_joint": 0.0,
            "r_leg_hip_pitch_joint": -0.1495,
            "r_leg_knee_joint": 0.3215,
            "r_leg_ankle_pitch_joint": -0.1720,
            "r_leg_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 80.0,
                ".*_hip_roll_joint": 80.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 100.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 10.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_pitch_joint": 14.0,
                ".*_knee_joint": 14.0,
            },
            stiffness={
                ".*_hip_yaw_joint": STIFFNESS_8112 * 2,
                ".*_hip_roll_joint": STIFFNESS_8112 * 2,
                ".*_hip_pitch_joint": STIFFNESS_10020 * 2,
                ".*_knee_joint": STIFFNESS_10020 * 2,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_8112 * 1.5,
                ".*_hip_roll_joint": DAMPING_8112 * 1.5,
                ".*_hip_pitch_joint": DAMPING_10020 * 1.1,
                ".*_knee_joint": DAMPING_10020 * 1.1,
            },
            armature={
                ".*_hip_yaw_joint": ARMATURE_8112,
                ".*_hip_roll_joint": ARMATURE_8112,
                ".*_hip_pitch_joint": ARMATURE_10020,
                ".*_knee_joint": ARMATURE_10020,
            },
            min_delay=0,
            max_delay=10,
        ),
        "feet": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
            ],
            effort_limit_sim=70.0,
            velocity_limit_sim=10.0,
            stiffness=STIFFNESS_4315 * 1.05,
            damping=DAMPING_4315 * 1.05,
            armature=ARMATURE_4315,
            min_delay=0,
            max_delay=10,
        ),
    },
)

E1_12DOF_ACTION_SCALE = {}
for a in E1_12DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            E1_12DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
