
import torch
import yaml
import os

from rl_games.common import a2c_common 
from rl_games.algos_torch import torch_ext 
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs 
from rl_games.algos_torch import central_value 
from rl_games.common import common_losses 
from rl_games.common import datasets
from rl_games.common import tr_helpers
from rl_games.common import vecenv
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.algos_torch import torch_ext
from rl_games.common import schedulers
from rl_games.common.experience import ExperienceBuffer
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.model_builder import ModelBuilder
from datetime import datetime
from tensorboardX import SummaryWriter
import wandb


from typing import Dict


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


class DatasetCollector:
    def __init__(self, env, config, device="cuda:0"):
        self.env = env
        self.ov_env = env.env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.device = device
        self.config = config
        self.teacher_network_params = self.load_param_dict(self.config["cfg"])["params"]
        self.teacher_network = self.load_networks(self.teacher_network_params)

        self.value_size = 1
        self.normalize_value = self.teacher_network_params["config"]["normalize_value"]
        self.normalize_input = self.teacher_network_params["config"]["normalize_input"]
        self.student_proprio_obs_shape = self.ov_env.num_observations
        self.student_img_obs_shape = (3, 224, 224)
        self.teacher_model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_teacher_observations,),
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value, 
            'normalize_input': self.normalize_input,
        }
        self.teacher_model = self.teacher_network.build(self.teacher_model_config).to(self.device)
        # load weights for teacher
        self.set_weights(self.config["ckpt"], "teacher")
        # get the observation type of the student and teacher
        self.student_obs_type = "policy"
        self.teacher_obs_type = "expert_policy"

        self.max_timesteps = 1_000
        
        self.init_tensors()

    def init_tensors(self):
        # dummy variable so that calculating neglogp doesn't give error (we don't care about the value)
        self.prev_actions_student = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32).to(self.device)
        self.prev_actions_teacher = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float32).to(self.device)

        self.current_rewards = torch.zeros(
            (self.num_envs, self.value_size), dtype=torch.float32, device=self.device
        )
        self.current_lengths = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.dones = torch.ones(
            (self.num_envs,), dtype=torch.uint8, device=self.device
        )

    def collect_data(self, num_trajectories):
        self.teacher_model.eval()

        obs = self.env.reset()[0]

        proprio_obs_buffer_NTO = torch.zeros(
            (self.num_envs, self.max_timesteps, self.student_proprio_obs_shape)
        ).to(self.device)
        img_obs_buffer_NTCHW = torch.zeros(
            (self.num_envs, self.max_timesteps, *self.student_img_obs_shape)
        ).to(self.device)
        actions_buffer_NA = torch.zeros(
            (self.num_envs, self.max_timesteps, self.num_actions)
        ).to(self.device)
        timesteps_buffer_N = torch.zeros(self.num_envs, dtype=torch.int32).to(self.device)

        num_saved_trajectories = 0

        while num_saved_trajectories < num_trajectories:
            with torch.no_grad():
                actions_teacher = self.get_actions(obs, "teacher")["mus"]
            
            proprio_obs_buffer_NTO[:, timesteps_buffer_N] = obs["policy"]
            img_obs_buffer_NTCHW[:, timesteps_buffer_N] = obs["img"].permute((0, 3, 1, 2))
            actions_buffer_NA[:, timesteps_buffer_N] = actions_teacher

            obs, rew, out_of_reach, timed_out, info = self.env.step(actions_teacher)

            timesteps_buffer_N += 1
            self.current_lengths += 1
            self.dones = out_of_reach | timed_out
            all_done_indices = self.dones.nonzero()
            done_indices = all_done_indices[:]
            num_done = done_indices.size()[0]

            # some envs finished, save their trajectories
            if num_done > 0:
                num_saved_trajectories += num_done
                self.save_trajectory(
                    proprio_obs_buffer_NTO[done_indices],
                    img_obs_buffer_NTCHW[done_indices],
                    actions_buffer_NA[done_indices],
                    timesteps_buffer_N[done_indices],
                    num_saved_trajectories
                )
                proprio_obs_buffer_NTO[done_indices] = 0.
                img_obs_buffer_NTCHW[done_indices] = 0.
                actions_buffer_NA[done_indices] = 0.
                timesteps_buffer_N[done_indices] = 0
                print(f"finished saving {num_done} trajectories")


    def save_trajectory(self, proprio_buffer, img_buffer, actions_buffer, timesteps, step):
        torch.save(proprio_buffer[:, :timesteps], f"proprio_buffer_{step}.pth")
        torch.save(img_buffer[:, :timesteps], f"img_buffer_{step}.pth")
        torch.save(actions_buffer[:, :timesteps], f"actions_buffer_{step}.pth")

    def get_actions(self, obs, policy_type):
        aux = None
        if policy_type == "student":
            batch_dict = {
                "is_train": True,
                "obs": obs[self.student_obs_type],
                "prev_actions": self.prev_actions_student,
            }
            if self.is_rnn:
                batch_dict["rnn_states"] = self.student_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            res_dict = self.student_model(batch_dict)
            mus = res_dict["mus"]
            sigmas = res_dict["sigmas"]
            if self.is_rnn:
                self.student_hidden_states = [s.detach() for s in res_dict["rnn_states"][0]]
            if self.is_aux:
                # aux = self.student_model.a2c_network.last_aux_out
                aux = res_dict["rnn_states"][1]
        else:
            batch_dict = {
                "is_train": False,
                "obs": obs[self.teacher_obs_type],
                "prev_actions": self.prev_actions_teacher,
            }
            res_dict = self.teacher_model(batch_dict)
            mus = res_dict["mus"]
            sigmas = res_dict["sigmas"]
        distr = torch.distributions.Normal(mus, sigmas, validate_args=False)
        selected_action = distr.sample().squeeze()

        return {
            "mus": mus,
            "sigmas": sigmas,
            "actions": selected_action,
            "aux": aux
        }

    def set_weights(self, ckpt, policy_type):
        """Set the weights of the model."""
        weights = torch_ext.load_checkpoint(ckpt)
        if policy_type == "student":
            model = self.student_model
            # self.epoch_num = weights.get('epoch', 0)
            # self.optimizer.load_state_dict(weights['optimizer'])
            # self.frame = weights.get('frame', 0)
        else:
            model = self.teacher_model
        model.load_state_dict(weights["model"])
        if self.normalize_input and 'running_mean_std' in weights:
            model.running_mean_std.load_state_dict(weights["running_mean_std"])
    
    def save(self, filename):
        """Save the checkpoint to filename"""
        state = {
            "model": self.student_model.state_dict()
        }
        state['epoch'] = self.epoch_num
        state['optimizer'] = self.optimizer.state_dict()
        state['frame'] = self.frame
        torch_ext.save_checkpoint(filename, state)

    def load_networks(self, params):
        """Loads the network """
        builder = ModelBuilder()
        return builder.load(params)
    
    def load_param_dict(self, cfg_path) -> Dict:
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
