# BSD 3-Clause License
# Copyright (c) 2025-2026, Beijing Noetix Robotics TECHNOLOGY CO.,LTD.
# All rights reserved.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="WalkRun12dof-E1-v0",
    entry_point="NoetixE1.tasks.walkrun12dof.noetix_e1.env:ManagerBasedRLAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walkrun_cfg:WalkRunFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_ppo_cfg:AmpPpoRunnerCfg",
    },
)
