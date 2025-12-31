# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from isaaclab.utils import configclass
from NoetixE1.assets import ASSET_DIR
from NoetixE1.assets.robots.e1.e1 import E1_24DOF_ACTION_SCALE, E1_24DOF_CFG
from NoetixE1.tasks.mimic.tracking_env_cfg import TrackingEnvCfg


@configclass
class E1FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = E1_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = E1_24DOF_ACTION_SCALE
        self.commands.motion.anchor_body_name = "waist_roll_link"
        self.commands.motion.motion_file = f"{ASSET_DIR}/datasets/mimic/dance1.npz"
        self.commands.motion.ankle_dof_names = [
            "l_leg_ankle_roll_joint",
            "r_leg_ankle_roll_joint",
            "l_leg_ankle_pitch_joint",
            "r_leg_ankle_pitch_joint",
        ]
        self.commands.motion.body_names = [
            "base_link",
            "waist_roll_link",
            "l_leg_hip_pitch_link",
            "r_leg_hip_pitch_link",
            "l_leg_knee_link",
            "r_leg_knee_link",
            "l_arm_shoulder_roll_link",
            "r_arm_shoulder_roll_link",
            "l_leg_ankle_roll_link",
            "r_leg_ankle_roll_link",
            "l_arm_elbow_pitch_link",
            "r_arm_elbow_pitch_link",
            "l_arm_elbow_yaw_link",
            "r_arm_elbow_yaw_link",
        ]
