# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cube environment.
"""

import gymnasium as gym

from . import agents
from .franka_cube_residual_env_fixed import FrankaCubeResidualFixedEnv, FrankaCubeResidualFixedEnvCfg
from .franka_cube_residual_env_rand import FrankaCubeResidualRandEnv, FrankaCubeResidualRandEnvCfg
from ..xarm_mug.xarm_mug_residual_env_fixed import XArmMugResidualFixedEnv, XArmMugResidualFixedEnvCfg
from .franka_mug_residual_env_rand import FrankaMugResidualRandEnv, FrankaMugResidualRandEnvCfg
from .franka_cube_residual_env_camera import FrankaCubeResidualCamEnv, FrankaCubeResidualCamEnvCfg
from .franka_cube_residual_env_camera_multi import FrankaCubeResidualCamMultiEnv, FrankaCubeResidualCamMultiEnvCfg


##
# Register Gym environments.
##


gym.register(
    id="Residual-Cube-Fixed",
    entry_point="omni.isaac.lab_tasks.direct.franka_cube:FrankaCubeResidualFixedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCubeResidualFixedEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Residual-Cube-Rand",
    entry_point="omni.isaac.lab_tasks.direct.franka_cube:FrankaCubeResidualRandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCubeResidualRandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Residual-Mug-Rand",
    entry_point="omni.isaac.lab_tasks.direct.franka_cube:FrankaMugResidualRandEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaMugResidualRandEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Residual-Mug-Fixed",
    entry_point="omni.isaac.lab_tasks.direct.franka_cube:XArmMugResidualFixedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": XArmMugResidualFixedEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Residual-Cube-Cam",
    entry_point="omni.isaac.lab_tasks.direct.franka_cube:FrankaCubeResidualCamEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCubeResidualCamEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePPORunnerCamCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Residual-Cube-Cam-Multi",
    entry_point="omni.isaac.lab_tasks.direct.franka_cube:FrankaCubeResidualCamMultiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaCubeResidualCamMultiEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaCubePPORunnerCamCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
