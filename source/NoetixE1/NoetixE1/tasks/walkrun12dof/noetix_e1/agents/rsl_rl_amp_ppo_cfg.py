# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)


@configclass
class RslRlAmpPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name = "AMPPPO"
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
    discriminator_learning_rate = 5e-6
    discriminator_gradient_penalty_coef = 5
    discriminator_num_mini_batches = 80
    amp_replay_buffer_size = 1000000
    discriminator_loss_function = "WassersteinLoss"  # BCEWithLogitsLoss, MSELoss, WassersteinLoss
    normalize_advantage_per_mini_batch = False
    symmetry_cfg = RslRlSymmetryCfg(
        use_data_augmentation=True,
        use_mirror_loss=False,
        data_augmentation_func="NoetixE1.tasks.walkrun12dof.noetix_e1.mdp.symmetry.e1:data_augmentation_func_e1",
        mirror_loss_coeff=1.0,
    )
    rnd_cfg = None


@configclass
class AmpPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpOnPolicyRunner"
    seed = 42
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 250
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlAmpPpoAlgorithmCfg()
    clip_actions = 18.0

    experiment_name = "walkrun12dof"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walkrun12dof"
    wandb_project = "walkrun12dof"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    # amp discriminator parameter
    normalize_style_reward = False
    amp_reward_coef = 8.0
    amp_reward_lerp = 0.5  # reward = reward_coef * (1 - reward_lerp) * style_reward + reward_lerp * task_reward
    style_reward_function = "wasserstein_mapping"  # log_mapping, quad_mapping, wasserstein_mapping
    discriminator_shape = [1024, 512]
