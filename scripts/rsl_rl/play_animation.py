# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import NoetixE1.tasks  # noqa: F401
import torch
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners import *


@hydra_task_config(args_cli.task, args_cli.agent)
def play_animation(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None
    # env_cfg.commands.base_velocity.debug_vis = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ------ visualize -----
    mimic_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TrackingMarker",
        markers={
            "motion": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "robot": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )
    rigid_body_pos = env.unwrapped.robot.data.body_pos_w
    tracking_markers = VisualizationMarkers(mimic_marker_cfg)
    marker_indices = torch.zeros(rigid_body_pos.shape[1] * 2, dtype=torch.long, device=env.unwrapped.device)
    marker_indices[rigid_body_pos.shape[1] // 2 :] = 1

    frame_cnt = 0
    motion_index = 0  # 当前播放的motion索引
    while simulation_app.is_running():
        current_motion_frames = env.unwrapped.motion_loader.trajectory_num_frames[motion_index]
        frame_duration = env.unwrapped.motion_loader.trajectory_frame_durations[motion_index]
        time = (frame_cnt % (current_motion_frames - 1)) * frame_duration
        key_pos = env.unwrapped.visualize_motion(time, motion_index)
        frame_cnt += 1

        if frame_cnt >= current_motion_frames:
            frame_cnt = 0  # 重置帧计数器
            motion_index = (motion_index + 1) % len(env.unwrapped.motion_loader.trajectory_num_frames)
            # break

        rigid_body_pos = env.unwrapped.robot.data.body_pos_w.view(-1, 3)
        translations = torch.cat((rigid_body_pos, key_pos.view(-1, 3)), dim=0)
        tracking_markers.visualize(translations=translations, marker_indices=marker_indices)


if __name__ == "__main__":
    # run the main function
    play_animation()
    # close sim app
    simulation_app.close()
