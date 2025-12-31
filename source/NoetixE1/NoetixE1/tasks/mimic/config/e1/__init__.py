# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

# gym.register(
#     id="Mimic-E1-v0",
#     entry_point="NoetixE1.tasks.mimic.env:ManagerBasedRLMimicEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.E1FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_him_ppo_cfg:HimPpoRunnerCfg",
#     },
# )

gym.register(
    id="Mimic-E1-v0",
    entry_point="NoetixE1.tasks.mimic.env:ManagerBasedRLMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.E1FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PpoRunnerCfg",
    },
)
