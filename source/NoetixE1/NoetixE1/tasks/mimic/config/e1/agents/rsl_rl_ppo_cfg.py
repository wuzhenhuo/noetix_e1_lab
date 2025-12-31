# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoAlgorithmCfg1(RslRlPpoAlgorithmCfg):
    class_name = "PPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.01
    num_learning_epochs = 5
    num_mini_batches = 4
    learning_rate = 1.0e-3
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0


@configclass
class RslRlActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name = "ActorCritic"
    init_noise_std = 1.0
    actor_obs_normalization = True
    critic_obs_normalization = True
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    command_dim = 0
    estimate_dim = 6
    activation = "elu"


@configclass
class PpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    policy = RslRlActorCriticCfg()
    algorithm = RslRlPpoAlgorithmCfg1()
    clip_actions = 18.0

    experiment_name = "mimic"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "mimic"
    wandb_project = "mimic"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
