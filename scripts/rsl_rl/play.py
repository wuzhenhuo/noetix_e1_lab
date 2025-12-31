# # BSD 3-Clause License
# # Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# # All rights reserved.

# """Script to play a checkpoint if an RL agent from RSL-RL."""

# """Launch Isaac Sim Simulator first."""

# import argparse
# import sys

# from isaaclab.app import AppLauncher

# # local imports
# import cli_args  # isort: skip

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
# parser.add_argument("--video_length", type=int, default=5000, help="Length of the recorded video (in steps).")
# parser.add_argument(
#     "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
# )
# parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# parser.add_argument(
#     "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
# )
# parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# # append RSL-RL cli arguments
# cli_args.add_rsl_rl_args(parser)
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli, hydra_args = parser.parse_known_args()
# # always enable cameras to record video
# if args_cli.video:
#     args_cli.enable_cameras = True

# # clear out sys.argv for Hydra
# sys.argv = [sys.argv[0]] + hydra_args

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# """Rest everything follows."""

# import os
# import time

# import gymnasium as gym
# import isaaclab_tasks  # noqa: F401
# import NoetixE1.tasks  # noqa: F401
# import torch
# from isaaclab.envs import (
#     DirectMARLEnv,
#     DirectMARLEnvCfg,
#     DirectRLEnvCfg,
#     ManagerBasedRLEnvCfg,
#     multi_agent_to_single_agent,
# )
# from isaaclab.utils.assets import retrieve_file_path
# from isaaclab.utils.dict import print_dict
# from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
# from isaaclab_tasks.utils import get_checkpoint_path
# from isaaclab_tasks.utils.hydra import hydra_task_config

# from rsl_rl.runners import *
# from rsl_rl.utils.exporter import attach_onnx_metadata, export_policy_as_onnx


# @hydra_task_config(args_cli.task, args_cli.agent)
# def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
#     """Play with RSL-RL agent."""
#     # grab task name for checkpoint path

#     # override configurations with non-hydra CLI arguments
#     agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

#     # set the environment seed
#     # note: certain randomizations occur in the environment initialization so we set the seed here
#     env_cfg.seed = agent_cfg.seed
#     env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

#     # set the play environment config
#     env_cfg.episode_length_s = 20.0
#     env_cfg.scene.num_envs = 4
#     env_cfg.scene.env_spacing = 2.5

#     # env_cfg.commands.base_velocity.rel_standing_envs = 0.0
#     # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
#     # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.3, 0.5)
#     # # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
#     # env_cfg.commands.base_velocity.ranges.zero_prob = [0.0, 0.0, 0.0]
#     env_cfg.observations.policy.enable_corruption = False
#     env_cfg.viewer.origin_type = None
#     env_cfg.viewer.asset_name = None

#     env_cfg.events.physics_material = None
#     env_cfg.events.add_joint_default_pos = None
#     env_cfg.events.base_com = None
#     env_cfg.events.randomize_joint_parameters = None
#     env_cfg.events.randomize_actuator_gains = None
#     env_cfg.events.push_robot = None

#     env_cfg.scene.terrain.terrain_type = "plane"
#     # env_cfg.scene.terrain.terrain_generator = None

#     if env_cfg.scene.terrain.terrain_generator is not None:
#         env_cfg.scene.terrain.terrain_generator.num_rows = 5
#         env_cfg.scene.terrain.terrain_generator.num_cols = 5
#         env_cfg.scene.terrain.terrain_generator.curriculum = False
#         env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.4, 0.4)

#     env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

#     # specify directory for logging experiments
#     log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
#     log_root_path = os.path.abspath(log_root_path)
#     print(f"[INFO] Loading experiment from directory: {log_root_path}")
#     if args_cli.checkpoint:
#         resume_path = retrieve_file_path(args_cli.checkpoint)
#     else:
#         resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

#     log_dir = os.path.dirname(resume_path)

#     # create isaac environment
#     env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

#     # convert to single-agent instance if required by the RL algorithm
#     if isinstance(env.unwrapped, DirectMARLEnv):
#         env = multi_agent_to_single_agent(env)

#     # wrap for video recording
#     if args_cli.video:
#         video_kwargs = {
#             "video_folder": os.path.join(log_dir, "videos", "play"),
#             "step_trigger": lambda step: step == 0,
#             "video_length": args_cli.video_length,
#             "disable_logger": True,
#         }
#         print("[INFO] Recording videos during training.")
#         print_dict(video_kwargs, nesting=4)
#         env = gym.wrappers.RecordVideo(env, **video_kwargs)

#     # wrap around environment for rsl-rl
#     env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

#     print(f"[INFO]: Loading model checkpoint from: {resume_path}")
#     # load previously trained model
#     runner_class: OnPolicyRunner | AmpHimOnPolicyRunner = eval(agent_cfg.class_name)
#     runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
#     runner.load(resume_path)

#     # obtain the trained policy for inference
#     policy = runner.get_inference_policy(device=env.unwrapped.device)

#     # extract the neural network module
#     # we do this in a try-except to maintain backwards compatibility.
#     try:
#         # version 2.3 onwards
#         policy_nn = runner.alg.policy
#     except AttributeError:
#         # version 2.2 and below
#         policy_nn = runner.alg.actor_critic

#     # extract the normalizer
#     if hasattr(policy_nn, "actor_obs_normalizer"):
#         normalizer = policy_nn.actor_obs_normalizer
#     elif hasattr(policy_nn, "student_obs_normalizer"):
#         normalizer = policy_nn.student_obs_normalizer
#     else:
#         normalizer = None

#     # export policy to onnx/jit
#     export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
#     export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
#     attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

#     dt = env.unwrapped.step_dt

#     # reset environment
#     # env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 1.0], device=env.unwrapped.device)  # set motion command to "walk forward"

#     obs = env.get_observations()
#     timestep = 0
#     # simulate environment
#     if args_cli.video:
#         print(f"[INFO] Recording video of length {args_cli.video_length} frames.")
#         from tqdm import tqdm
#         import time
#         pbar = tqdm(total=args_cli.video_length, desc="Playing Recording Video")
#     while simulation_app.is_running():
#         start_time = time.time()
#         # run everything in inference mode
#         with torch.inference_mode():
#             # agent stepping
#             # if timestep < 200:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # elif timestep >= 200 and timestep < 400:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # elif timestep >= 400 and timestep < 800:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # elif timestep >= 800 and timestep < 1100:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # elif timestep >= 1100 and timestep < 1500:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # elif timestep >= 1500 and timestep < 1800:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # elif timestep >= 1800:
#             #     env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device)  # set motion command to "walk forward"


#             # env.unwrapped.command_manager._terms["motion_velocity"].motion_commands[0] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # env.unwrapped.command_manager._terms["motion_velocity"].vel_command_b[0] = torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             # env.unwrapped.command_manager._terms["motion_commands"].motion_commands[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=env.unwrapped.device)  # set motion command to "walk forward"
#             print(f"current motion: {env.unwrapped.command_manager.get_command('motion_velocity')[1]}, current velocity: {env.unwrapped.command_manager.get_command('motion_velocity')[0]}")
#             actions = policy(obs)
#             # env stepping
#             obs, _, _, _ = env.step(actions)
#         if args_cli.video:
#             timestep += 1
#             pbar.update(1)
#             # Exit the play loop after recording one video
#             if timestep == args_cli.video_length:
#                 break

#         # time delay for real-time evaluation
#         sleep_time = dt - (time.time() - start_time)
#         if args_cli.real_time and sleep_time > 0:
#             time.sleep(sleep_time)

#     # close the simulator
#     if args_cli.video:
#         pbar.close()
#     env.close()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()

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

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import NoetixE1.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rsl_rl.runners import *
from rsl_rl.utils.exporter import attach_onnx_metadata, export_policy_as_onnx


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # set the play environment config
    env_cfg.episode_length_s = 20.0
    env_cfg.scene.num_envs = 10
    env_cfg.scene.env_spacing = 2.5

    # env_cfg.commands.base_velocity.rel_standing_envs = 0.0
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.zero_prob = [0.0, 0.0, 0.0]
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.push_robot = None

    # env_cfg.scene.terrain.terrain_type = "plane"
    # env_cfg.scene.terrain.terrain_generator = None

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.4, 0.4)

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner_class: OnPolicyRunner | AmpHimOnPolicyRunner = eval(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # print(f"Command: {env.unwrapped.command_manager.get_command('base_velocity').cpu().numpy()[0]}, Current Velocity: {env.unwrapped.robot.data.root_lin_vel_b.cpu().numpy()[0]}")
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            print(f"[INFO] Recording video: {timestep}/{args_cli.video_length}")
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
