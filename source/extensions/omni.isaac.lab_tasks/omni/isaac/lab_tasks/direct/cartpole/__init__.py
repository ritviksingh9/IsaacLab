# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Cartpole balancing environment.
"""

import gymnasium as gym

from . import agents
from .cartpole_camera_env import CartpoleCameraEnv, CartpoleDepthCameraEnvCfg, CartpoleRGBCameraEnvCfg
from .cartpole_env import CartpoleEnv, CartpoleEnvCfg
from .cartpole_dino_env import CartpoleDinoEnv, CartpoleDinoEnvCfg
from .cartpole_embeddings_env import CartpoleEmbeddingEnv, CartpoleEmbeddingEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Cartpole-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cartpole:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CartpolePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-RGB-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cartpole:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleRGBCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Depth-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cartpole:CartpoleCameraEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleDepthCameraEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Cartpole-Embedding-Camera-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.cartpole:CartpoleEmbeddingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": CartpoleEmbeddingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
    },
)