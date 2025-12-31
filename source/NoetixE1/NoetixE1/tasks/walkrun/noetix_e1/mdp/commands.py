# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class VelocityWithMotionCommand(CommandTerm):
    """Command generator that outputs velocity-based motion command from AMP."""

    cfg: VelocityWithMotionCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: VelocityWithMotionCommandCfg, env: ManagerBasedRLEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)
        # obtain the robot asset
        self.robot: Articulation = env.scene[cfg.asset_name]

        # -------------------step 1: base velocity-------------------
        # create buffers to store the command
        # -- command: x vel, y vel, yaw vel
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

        # create buffers for zero commands envs
        self.is_zero_vel_x_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_zero_vel_y_env = torch.zeros_like(self.is_zero_vel_x_env)
        self.is_zero_vel_yaw_env = torch.zeros_like(self.is_zero_vel_x_env)

        # -------------------step 2: motion -------------------
        self.num_skill_labels = env.cfg.num_skill_labels
        self.motion_commands = torch.zeros((self.num_envs, self.num_skill_labels), dtype=torch.float, device=self.device)
    
    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "NormalVelocityCommand:\n"
        msg += f"\tVelocity Command dimension: {tuple(self.command[0].shape[1:])}\n"
        msg += f"\tMotion Command dimension: {tuple(self.command[1].shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg
    
    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired motion command. Shape is (num_envs, num_skill_labels)."""
        return self.vel_command_b, self.motion_commands
    
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids):
        '''
        1022 更新： 两种命令解耦 -->> 根据速度决定motion
        '''
        # -------------------step 1: base velocity-------------------
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- angular velocity - yaw direction
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # update element wise zero velocity command
        self.is_zero_vel_x_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[0]
        self.is_zero_vel_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[1]
        self.is_zero_vel_yaw_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.ranges.zero_prob[2]

        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0

        # -------------------step 2: motion -------------------
        vx = self.vel_command_b[env_ids, 0]
        vy = self.vel_command_b[env_ids, 1]
        yaw = self.vel_command_b[env_ids, 2]

        motion_mode = torch.zeros_like(vx, dtype=torch.long, device=self.device)
        cond_side = vy.abs() > 0
        self.vel_command_b[env_ids[cond_side], 0] = 0.0
        self.vel_command_b[env_ids[cond_side], 2] = 0.0
        motion_mode[cond_side] = 5

        # ---- 6: 跑步 ----
        run_threshold = 1.5
        cond_run= vx > run_threshold
        motion_mode[cond_run] = 6

        # ---- 1: 前向行走 ----
        cond_walk = (vx > 0) & (vx <= 1.5) & (yaw.abs() == 0)
        motion_mode[cond_walk] = 1

        # ---- 4: 行走转向 ----
        cond_turn_walk = (vx > 0.0) & (vx <= 1.5) & (yaw.abs() > 0)
        motion_mode[cond_turn_walk] = 4

        # ---- 2: 后退行走 ----
        cond_turn_walk = (vx < 0)
        motion_mode[cond_turn_walk] = 2

        # ---- 3: 原地转弯 ----
        cond_turn_in_place = (vx == 0.0) & (yaw.abs() > 0.0)
        motion_mode[cond_turn_in_place] = 3

        # ---- 0: 站立 ----
        cond_stand = (vx == 0.0) & (vy == 0.0) & (yaw == 0.0)
        motion_mode[cond_stand] = 0
        self.motion_commands = torch.nn.functional.one_hot(
            motion_mode, num_classes=self.num_skill_labels
        ).float()

    def _update_command(self):
        """
        Sets velocity command to zero
        Sets motion command to stand motion for standing envs.
        """
        # Enforce standing (i.e., zero velocity command) for standing envs
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        self.vel_command_b[standing_env_ids, :] = 0.0

        # Enforce zero velocity for individual elements
        # velocity command
        zero_vel_x_env_ids = self.is_zero_vel_x_env.nonzero(as_tuple=False).flatten()
        zero_vel_y_env_ids = self.is_zero_vel_y_env.nonzero(as_tuple=False).flatten()
        zero_vel_yaw_env_ids = self.is_zero_vel_yaw_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[zero_vel_x_env_ids, 0] = 0.0
        self.vel_command_b[zero_vel_y_env_ids, 1] = 0.0
        self.vel_command_b[zero_vel_yaw_env_ids, 2] = 0.0
        # motion command
        vx = self.vel_command_b[:, 0]
        vy = self.vel_command_b[:, 1]
        yaw = self.vel_command_b[:, 2]
        motion_mode = torch.zeros_like(vx, dtype=torch.long, device=self.device)

        # 优先判断 y 方向速度，如果不为零，则把 x 方向速度和 yaw 方向速度都置零
        # 在 command cfg 里把 y 分量随机为0的概率设置到了比较大的值（0.9）
        cond_side = vy.abs() > 0
        self.vel_command_b[cond_side, 0] = 0.0
        self.vel_command_b[cond_side, 2] = 0.0
        motion_mode[cond_side] = 5

        # ---- 6: 跑步 ----
        run_threshold = 1.5
        cond_run= vx > run_threshold
        motion_mode[cond_run] = 6

        # ---- 1: 前向行走 ----
        cond_walk = (vx > 0) & (vx <= 1.5) & (yaw.abs() == 0)
        motion_mode[cond_walk] = 1

        # ---- 4: 行走转向 ----
        cond_turn_walk = (vx > 0.0) & (vx <= 1.5) & (yaw.abs() > 0)
        motion_mode[cond_turn_walk] = 4

        # ---- 2: 后退行走 ----
        cond_turn_walk = (vx < 0)
        motion_mode[cond_turn_walk] = 2

        # ---- 3: 原地转弯 ----
        cond_turn_in_place = (vx == 0.0) & (yaw.abs() > 0.0)
        motion_mode[cond_turn_in_place] = 3

        # ---- 0: 站立 ----
        cond_stand = (vx == 0.0) & (vy == 0.0) & (yaw == 0.0)
        motion_mode[cond_stand] = 0

        # ---- 强制站立环境设为站立模式 ----
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()  # TODO check if conversion is needed
        motion_mode[standing_env_ids] = 0

        # ========== Step 4. 转换为 one-hot ==========
        self.motion_commands = torch.nn.functional.one_hot(
            motion_mode, num_classes=self.num_skill_labels
        ).float()
    
    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[0][:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class VelocityWithMotionCommandCfg(CommandTermCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = VelocityWithMotionCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
