# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_root_state_amp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    root_pos_range: dict[str, tuple[float, float]],
    root_vel_range: dict[str, tuple[float, float]],
    joint_pos_range: tuple[float, float],
    joint_vel_range: tuple[float, float],
    reference_state_initialization: bool = False,
    prob_rsi: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    if (
        env.unwrapped.motion_loader is not None
        and reference_state_initialization
        and prob_rsi > torch.rand(1, dtype=torch.float32, device=env.device)
    ):
        # get frames of amp motions
        frames = env.motion_loader.get_full_frame_batch(len(env_ids))

        # get root state of amp motions
        positions = env.unwrapped.motion_loader_class.get_root_pos_batch(frames) + env.scene.env_origins[env_ids]
        positions[:, 2] += 0.05
        quat_xyzw = env.unwrapped.motion_loader_class.get_root_rot_batch(frames)  # xyzw
        orientations = torch.cat((quat_xyzw[:, -1].unsqueeze(1), quat_xyzw[:, :-1]), dim=1)  # xyzw -> wxyz

        # base root velocities
        lin_vel = math_utils.quat_apply(orientations, env.unwrapped.motion_loader_class.get_linear_vel_batch(frames))
        ang_vel = math_utils.quat_apply(orientations, env.unwrapped.motion_loader_class.get_angular_vel_batch(frames))
        velocities = torch.cat([lin_vel, ang_vel], dim=-1)

        # get joint state of amp motions
        joint_pos = env.unwrapped.motion_loader_class.get_joint_pose_batch(frames)[
            ..., env.unwrapped.motion_loader.joint_mapping["motion2sim"]
        ]
        joint_vel = env.unwrapped.motion_loader_class.get_joint_vel_batch(frames)[
            ..., env.unwrapped.motion_loader.joint_mapping["motion2sim"]
        ]
    else:
        # get default root state
        root_states = asset.data.default_root_state[env_ids].clone()

        # poses
        range_list = [root_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # base root velocities
        range_list = [root_vel_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        # get default joint state
        joint_pos = asset.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

        # bias these values randomly
        joint_pos += math_utils.sample_uniform(*joint_pos_range, joint_pos.shape, joint_pos.device)
        joint_vel += math_utils.sample_uniform(*joint_vel_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos
