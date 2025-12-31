# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_joint_names,
    record_shape,
)


def current_feet_contact(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
    return feet_contact


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def amp_joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if env.unwrapped.motion_loader is None:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    else:
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids][
            ..., env.unwrapped.motion_loader.joint_mapping["sim2motion"]
        ]
    return joint_pos


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def amp_joint_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if env.unwrapped.motion_loader is None:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    else:
        joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids][
            ..., env.unwrapped.motion_loader.joint_mapping["sim2motion"]
        ]
    return joint_vel


def amp_key_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Key pos on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w[:, :3].unsqueeze(1)
    if env.unwrapped.motion_loader is None:
        key_pos = asset.data.body_pos_w.clone()
    else:
        key_pos = asset.data.body_pos_w.clone()[:, env.unwrapped.motion_loader.key_pos_mapping["sim2motion"], :]
    local_key_pos = (key_pos - root_pos).flatten(1, 2)
    return local_key_pos
