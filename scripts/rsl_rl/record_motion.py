# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import json
import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

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
from tqdm import tqdm
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

mujoco2isaaclab_joint = [0, 6, 12, 1, 7, 13, 18, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
isaaclab2mujoco_joint = [0, 3, 7, 11, 15, 19, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 6, 10, 14, 18, 22]

mujoco2isaaclab_keypose = [0, 1, 7, 13, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22, 6, 12, 18, 23]
isaaclab2mujoco_keypose = [0, 1, 4, 8, 12, 16, 20, 2, 5, 9, 13, 17, 21, 3, 6, 10, 14, 18, 22, 7, 11, 15, 19, 23]

joint_mujoco = ["l_leg_hip_yaw_joint", "l_leg_hip_roll_joint", "l_leg_hip_pitch_joint", "l_leg_knee_joint", "l_leg_ankle_pitch_joint", "l_leg_ankle_roll_joint", 
 "r_leg_hip_yaw_joint", "r_leg_hip_roll_joint", "r_leg_hip_pitch_joint", "r_leg_knee_joint", "r_leg_ankle_pitch_joint", "r_leg_ankle_roll_joint", 
 "waist_yaw_joint", "waist_roll_joint", 
 "l_arm_shoulder_pitch_joint", "l_arm_shoulder_roll_joint", "l_arm_shoulder_yaw_joint", "l_arm_elbow_pitch_joint", "l_arm_elbow_yaw_joint", 
 "r_arm_shoulder_pitch_joint", "r_arm_shoulder_roll_joint", "r_arm_shoulder_yaw_joint", "r_arm_elbow_pitch_joint", "r_arm_elbow_yaw_joint"]

joint_isaaclab = ['l_leg_hip_yaw_joint', 'r_leg_hip_yaw_joint', 
                  'waist_yaw_joint', 
                  'l_leg_hip_roll_joint', 'r_leg_hip_roll_joint', 
                  'waist_roll_joint', 
                  'l_leg_hip_pitch_joint', 'r_leg_hip_pitch_joint', 
                  'l_arm_shoulder_pitch_joint', 'r_arm_shoulder_pitch_joint', 
                  'l_leg_knee_joint', 'r_leg_knee_joint', 
                  'l_arm_shoulder_roll_joint', 'r_arm_shoulder_roll_joint', 
                  'l_leg_ankle_pitch_joint', 'r_leg_ankle_pitch_joint', 
                  'l_arm_shoulder_yaw_joint', 'r_arm_shoulder_yaw_joint', 
                  'l_leg_ankle_roll_joint', 'r_leg_ankle_roll_joint', 
                  'l_arm_elbow_pitch_joint', 'r_arm_elbow_pitch_joint', 
                  'l_arm_elbow_yaw_joint', 'r_arm_elbow_yaw_joint']
mujoco2isaaclab_joint = [joint_mujoco.index(name) for name in joint_isaaclab]
isaaclab2mujoco_joint = [joint_isaaclab.index(name) for name in joint_mujoco]

key_pos_mujoco = ["base_link", "l_leg_hip_yaw_link", "l_leg_hip_roll_link", "l_leg_hip_pitch_link", "l_leg_knee_link", "l_leg_ankle_pitch_link", "l_leg_ankle_roll_link", 
 "r_leg_hip_yaw_link", "r_leg_hip_roll_link", "r_leg_hip_pitch_link", "r_leg_knee_link", "r_leg_ankle_pitch_link", "r_leg_ankle_roll_link", ""
 "waist_yaw_link", "waist_roll_link", 
 "l_arm_shoulder_pitch_link", "l_arm_shoulder_roll_link", "l_arm_shoulder_yaw_link", "l_arm_elbow_pitch_link", "l_arm_elbow_yaw_link", 
 "r_arm_shoulder_pitch_link", "r_arm_shoulder_roll_link", "r_arm_shoulder_yaw_link", "r_arm_elbow_pitch_link", "r_arm_elbow_yaw_link"]

key_pos_isaaclab = ['base_link', 'l_leg_hip_yaw_link', 'r_leg_hip_yaw_link', 'waist_yaw_link', 'l_leg_hip_roll_link', 'r_leg_hip_roll_link', 'waist_roll_link', 'l_leg_hip_pitch_link', 'r_leg_hip_pitch_link', 'l_arm_shoulder_pitch_link', 'r_arm_shoulder_pitch_link', 'l_leg_knee_link', 'r_leg_knee_link', 'l_arm_shoulder_roll_link', 'r_arm_shoulder_roll_link', 'l_leg_ankle_pitch_link', 'r_leg_ankle_pitch_link', 'l_arm_shoulder_yaw_link', 'r_arm_shoulder_yaw_link', 'l_leg_ankle_roll_link', 'r_leg_ankle_roll_link', 'l_arm_elbow_pitch_link', 'r_arm_elbow_pitch_link', 'l_arm_elbow_yaw_link', 'r_arm_elbow_yaw_link']
mujoco2isaaclab_keypose = [key_pos_mujoco.index(name) for name in key_pos_isaaclab]
isaaclab2mujoco_keypose = [key_pos_isaaclab.index(name) for name in key_pos_mujoco]



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
    env_cfg.episode_length_s = 80.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5

    env_cfg.observations.policy.enable_corruption = False
    env_cfg.viewer.origin_type = None
    env_cfg.viewer.asset_name = None

    env_cfg.events.physics_material = None
    env_cfg.events.add_joint_default_pos = None
    env_cfg.events.base_com = None
    env_cfg.events.randomize_joint_parameters = None
    env_cfg.events.randomize_actuator_gains = None
    env_cfg.events.push_robot = None

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None

    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False
        env_cfg.scene.terrain.terrain_generator.difficulty_range = (0.4, 0.4)
    
    env_cfg.commands.base_velocity.resampling_time_range = (5.0, 5.0)
    env_cfg.commands.base_velocity.rel_standing_envs = 0.1
    # env_cfg.commands.base_velocity.ranges.lin_vel_x = (2.0, 2.5)
    # env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    # env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.zero_prob = [0.0, 0.0, 0.2]

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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_onnx(policy_nn, path=export_model_dir, filename="policy.onnx")
    attach_onnx_metadata(env=env.unwrapped, path=export_model_dir)

    dt = env.unwrapped.step_dt
    record_duration = 50 # s
    stop_state_log = round(record_duration / dt)

    # reset environment
    obs = env.get_observations()
    motion_frames = []
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        for i in tqdm(range(stop_state_log)):
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)

                # if i < 400: 
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 1] = -0.5
                # elif i < 550:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 1] = -0.25
                # elif i < 700:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 1] = 0.25
                # elif i < 850:
                #     env.unwrapped.command_manager._terms["base_velocity"].command[:, 1] = 0.5

                # env.unwrapped.command_manager._terms["base_velocity"].command[:, 0] = 0.0
                # env.unwrapped.command_manager._terms["base_velocity"].command[:, 2] = 0.0

                # env stepping
                obs, _, _, _ = env.step(actions)

                root_pos = env.unwrapped.robot.data.root_pos_w.clone()
                root_quat = env.unwrapped.robot.data.root_quat_w.clone()
                root_quat = torch.cat((root_quat[:, 1:], root_quat[:, 0].unsqueeze(1)), dim=1)  #  wxyz -> xyzw
                dof_pos = env.unwrapped.robot.data.joint_pos[:, isaaclab2mujoco_joint]
                key_pos = (env.unwrapped.robot.data.body_pos_w[:, isaaclab2mujoco_keypose, :] - root_pos.unsqueeze(1)).flatten(1, 2)
                root_vel = env.unwrapped.robot.data.root_lin_vel_b.clone()
                root_ang_vel = env.unwrapped.robot.data.root_ang_vel_b.clone()
                dof_vel = env.unwrapped.robot.data.joint_vel[:, isaaclab2mujoco_joint]
                contact_mask = obs['critic'][:, -2:].clone()
                cur_frames_tensor = torch.cat((root_pos, root_quat, dof_pos, key_pos, root_vel, root_ang_vel, dof_vel, contact_mask), dim=-1)
                cur_frames = cur_frames_tensor[0].tolist()
                motion_frames.append(cur_frames)
        
        motion_data = {}
        motion_data["FrameDuration"] = dt
        motion_data["JointNames"] = [env.unwrapped.robot.joint_names[i] for i in isaaclab2mujoco_joint]
        motion_data["KeyPosNames"] = [env.unwrapped.robot.body_names[i] for i in isaaclab2mujoco_keypose]
        motion_data["MotionWeight"] = 0.5
        motion_data["Frames"] = motion_frames[250:]
        # export motion capture to json format
        motion_name = "moving.txt"
        with open("source/NoetixE1/NoetixE1/assets/datasets/boxing/" + motion_name, "w") as f:
            json.dump(motion_data, f, indent=1, separators=(",", ":"))
            print("Generating " + motion_name)
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
