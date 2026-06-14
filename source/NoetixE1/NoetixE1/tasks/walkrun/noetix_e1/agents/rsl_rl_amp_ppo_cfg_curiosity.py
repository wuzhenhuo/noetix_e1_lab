# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

# Ablation: + Curiosity Reward
# Change vs rsl_rl_amp_ppo_cfg.py: rnd_cfg enabled with linear weight decay.
# The RND module trains a predictor network to match a fixed random target;
# prediction error becomes an intrinsic reward that encourages exploration.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)


@configclass
class RslRlAmpPpoAlgorithmCuriosityCfg(RslRlPpoAlgorithmCfg):
    class_name = "ConditionAMPHIMPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.005
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    discriminator_learning_rate = 5e-6
    discriminator_gradient_penalty_coef = 10
    discriminator_num_mini_batches = 80
    amp_replay_buffer_size = 1000000
    discriminator_loss_function = "MSELoss"
    normalize_advantage_per_mini_batch = False
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func="NoetixE1.tasks.walkrun.noetix_e1.mdp.symmetry.e1:data_augmentation_func_e1",
        mirror_loss_coeff=1.0,
    )
    # [Curiosity Reward] RND enabled
    # num_states / obs_groups are auto-resolved by resolve_rnd_config at runtime
    rnd_cfg = RslRlRndCfg(
        num_outputs=64,
        predictor_hidden_dims=[256, 256],
        target_hidden_dims=[256, 256],
        activation="elu",
        weight=0.5,
        learning_rate=1e-4,
        state_normalization=True,
        reward_normalization=True,
        weight_schedule={
            "mode": "linear",
            "initial_step": 0,
            "final_step": 5000,   # linearly decay intrinsic reward over first 5000 iters
            "final_value": 0.05,  # small residual weight to sustain mild exploration
        },
    )


@configclass
class AmpPpoRunnerCuriosityCfg(RslRlOnPolicyRunnerCfg):
    class_name = "ConditionAmpHimOnPolicyRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    policy = RslRlPpoActorCriticCfg(
        class_name="HimActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlAmpPpoAlgorithmCuriosityCfg()
    clip_actions = 18.0

    experiment_name = "walkrunCAMP_curiosity"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walkrunCAMP_curiosity"
    wandb_project = "walkrunCAMP_curiosity"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp discriminator parameter
    normalize_style_reward = False
    amp_reward_coef = 4.0
    amp_reward_lerp = 0.5
    style_reward_function = "quad_mapping"
    discriminator_shape = [1024, 512]
