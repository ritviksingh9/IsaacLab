"""Script to perform student-teacher distillation"""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import math
import os
from datetime import datetime
import pathlib

from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner
from rl_games.algos_torch import model_builder

from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper


from dataset_collector import DatasetCollector

def main():
    """ Performs distillation. """

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = env.env

    parent_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())
    agent_cfg_folder = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/agents"
    teacher_cfg = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_cfg.yaml"
    )

    num_teacher_obs = ov_env.num_teacher_observations
    num_actions = ov_env.num_actions
    student_ckpt = None
    teacher_ckpt = "pretrained_ckpts/teacher.ckpt"
    teacher_ckpt = os.path.join(
        parent_path,
        teacher_ckpt
    )

    dataset_collector_config = {
        "cfg": teacher_cfg,
        "ckpt": teacher_ckpt,
    }
    dataset_collector = DatasetCollector(env, dataset_collector_config)
    dataset_collector.collect_data(num_trajectories=5_000)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
