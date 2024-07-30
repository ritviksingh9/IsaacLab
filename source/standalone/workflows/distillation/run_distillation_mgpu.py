"""Script to perform student-teacher distillation"""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
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
from rl_games_actor import AgentRLG
from distillation_mgpu import Dagger
from complex_net import A2CBuilder as ComplexNetworkBuilder
from a2c_with_aux import A2CBuilder as A2CWithAuxBuilder


def main():
    """ Performs distillation. """
    world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
    rank = int(os.environ['RANK'])  # Global rank of this process
    local_rank = int(os.environ['LOCAL_RANK']) # local rank of the process 

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, use_gpu=not args_cli.cpu, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += app_launcher.global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{app_launcher.local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        # update env config device
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    ov_env = env.env

    parent_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())
    agent_cfg_folder = "source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/agents"
    # student_cfg = "/home/ritviks/workspace/git/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/shadow_hand/agents/rl_games_ppo_lstm_cfg.yaml"
    student_cfg = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_lstm_aux_cfg.yaml"
    )
    teacher_cfg = os.path.join(
        parent_path,
        agent_cfg_folder,
        "rl_games_ppo_cfg.yaml"
    )

    num_student_obs = ov_env.num_observations
    num_teacher_obs = ov_env.num_teacher_observations
    num_actions = ov_env.num_actions
    student_ckpt = None
    teacher_ckpt = "pretrained_ckpts/teacher.ckpt"
    teacher_ckpt = os.path.join(
        parent_path,
        teacher_ckpt
    )

    if rank == 0:
        train_dir = "runs"
        experiment_name = (
            "Shadow-Hand-Camera-Distillation-MGPU"
            + datetime.now().strftime("_%d-%H-%M-%S")
        )
        experiment_dir = os.path.join(train_dir, experiment_name)
        nn_dir = os.path.join(experiment_dir, "nn")
        summaries_dir = os.path.join(experiment_dir, "summaries")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(nn_dir, exist_ok=True)
        os.makedirs(summaries_dir, exist_ok=True)
    else:
        summaries_dir = None

    dagger_config = {
        "student": {
            "cfg": student_cfg,
            "ckpt": student_ckpt,
            "obs_type": "policy",
        },
        "teacher": {
            "cfg": teacher_cfg,
            "ckpt": teacher_ckpt,
            "obs_type": "expert_policy",
        },
    }
    model_builder.register_network("complex_net", ComplexNetworkBuilder)
    model_builder.register_network("a2c_aux_net", A2CWithAuxBuilder)
    dagger = Dagger(env, dagger_config, summaries_dir=summaries_dir)
    dagger.distill()
    if rank == 0:
        dagger.save("sh_dist_no_vel_ff")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
