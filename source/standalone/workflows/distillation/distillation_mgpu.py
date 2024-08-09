
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
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



def l2(model, target):
    """Computes the L2 norm between model and target.
    """

    return torch.norm(model - target, p=2, dim=-1)

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action



class Dagger:
    def __init__(self, env, config, summaries_dir, nn_dir):
        self.world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
        self.rank = int(os.environ['RANK'])  # Global rank of this process
        self.local_rank = int(os.environ['LOCAL_RANK']) # local rank of the process 
        # dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)
        torch.cuda.set_device(self.local_rank)
        # torch.autograd.set_detect_anomaly(True)

        self.env = env
        self.ov_env = env.env
        self.num_envs = self.ov_env.num_envs
        self.num_actions = self.ov_env.num_actions
        self.device = self.local_rank
        self.config = config
        self.student_network_params = self.load_param_dict(self.config["student"]["cfg"])["params"]
        self.teacher_network_params = self.load_param_dict(self.config["teacher"]["cfg"])["params"]
        self.student_network = self.load_networks(self.student_network_params)
        self.teacher_network = self.load_networks(self.teacher_network_params)

        self.value_size = 1
        self.horizon_length = self.student_network_params["config"]["horizon_length"]
        self.normalize_value = self.student_network_params["config"]["normalize_value"]
        self.normalize_input = self.student_network_params["config"]["normalize_input"]

        # get student and teacher models
        self.num_actions_student = self.num_actions
        self.student_model_config = {
            "actions_num": self.num_actions_student,
            "input_shape": (self.ov_env.num_observations,),
            # "num_seqs": self.horizon_length // self.student_network_params["config"]["seq_length"],
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value, 
            'normalize_input': self.normalize_input,
        }
        self.teacher_model_config = {
            "actions_num": self.num_actions,
            "input_shape": (self.ov_env.num_teacher_observations,),
            "num_seqs": self.num_envs,
            "value_size": self.value_size,
            'normalize_value': self.normalize_value, 
            'normalize_input': self.normalize_input,
        }
        self.student_model = self.student_network.build(self.student_model_config).to(self.device)
        for param in self.student_model.parameters():
            dist.broadcast(param.data, src=0)
        self.student_model_ddp = DDP(self.student_model, device_ids=[self.local_rank], find_unused_parameters=True)
        self.teacher_model = self.teacher_network.build(self.teacher_model_config).to(self.device)
        self.finetune_backbone = self.ov_env.finetune_backbone
        self.warm_up_lr = 1e-5
        self.peak_lr = 1e-3
        params = [{"params": self.student_model_ddp.parameters(), "lr": self.warm_up_lr, "eps": 1e-8}]
        if self.finetune_backbone:
            params.append({"params": self.ov_env.backbone_ddp.parameters(), "lr": 1e-5, "eps": 1e-8})
        # self.optimizer = torch.optim.Adam(params)
        self.optimizer = torch.optim.Adam(self.student_model_ddp.parameters(), lr=1e-3, eps=1e-8)
        self.num_warmup_steps = 1000
        self.num_iters = 350000
        def lr_lambda(current_step):
            if current_step < self.num_warmup_steps:
                # Linear warmup
                print(                    self.warm_up_lr + 
                    float(current_step) / self.num_warmup_steps * (self.peak_lr - self.warm_up_lr))
                return (
                    self.warm_up_lr + 
                    float(current_step) / self.num_warmup_steps * (self.peak_lr - self.warm_up_lr)
                )
            else:
                # Decay
                return (
                    float(self.num_iters - current_step) / 
                    float(self.num_iters - self.warmup_steps) * self.peak_lr
                )
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # load weights for student and teacher
        if self.config["student"]["ckpt"] is not None:
            self.set_weights(self.config["student"]["ckpt"], "student")
        self.set_weights(self.config["teacher"]["ckpt"], "teacher")
        # get the observation type of the student and teacher
        self.student_obs_type = self.config["student"]["obs_type"]
        self.teacher_obs_type = self.config["teacher"]["obs_type"]
        self.is_rnn = self.student_model.is_rnn()
        if self.is_rnn:
            self.seq_length = self.student_network_params["config"]["seq_length"]
            print("USING RNN")
        if hasattr(self.student_model.a2c_network, "is_aux") and self.student_model.a2c_network.is_aux:
            self.is_aux = True
            print("USING AUX")
        else:
            self.is_aux = False

        # logging
        self.games_to_track = 100
        self.frame = 0
        self.epoch_num = 0
        self.game_rewards = torch_ext.AverageMeter(
            self.value_size, self.games_to_track
        ).to(self.device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.device)

        if self.rank == 0:
            self.writer = SummaryWriter(summaries_dir)
            self.use_wandb = True
            import pathlib
            parent_path = str(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())
            summaries_dir = os.path.join(parent_path, summaries_dir)
            self.nn_dir = os.path.join(parent_path, nn_dir)
            if self.use_wandb:
                wandb.login(key=os.environ["WANDB_API_KEY"])
                # wandb.tensorboard.patch(root_logdir=summaries_dir)
                wandb.init(
                    project=os.environ["WANDB_PROJECT"], 
                    entity=os.environ["WANDB_ENTITY"],
                    name=os.environ["WANDB_NAME"],
                    notes=os.environ["WANDB_NOTES"],
                    # sync_tensorboard=True,
                )

        self.init_tensors()

    def init_tensors(self):
        # dummy variable so that calculating neglogp doesn't give error (we don't care about the value)
        self.prev_actions_student = torch.zeros((self.num_envs, self.num_actions_student), dtype=torch.float32).to(self.device)
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

        if self.is_rnn:
            self.student_hidden_states = self.student_model.get_default_rnn_state()
            self.student_hidden_states = [s.to(self.device) for s in self.student_hidden_states]
            self.num_seqs = self.horizon_length // self.seq_length

    def distill(self):
        self.student_model.train()
        self.teacher_model.eval()

        # actions = self.prev_actions.clone()

        obs = self.env.reset()[0]

        log_counter = 0
        total_loss = 0.

        # for param in self.student_model_ddp.parameters():
        #     param.grad = None

        self.optimizer.zero_grad()

        num_iters = 350000

        while log_counter < num_iters:
            beta = max(1 - log_counter / (num_iters / 2), 0)

            with torch.no_grad():
                actions_teacher = self.get_actions(obs, "teacher")
            actions_student = self.get_actions(obs, "student")
            aux_loss = list() if self.is_aux else [0.]
            if actions_student["aux"] is not None:
                aux_out = actions_student["aux"]
                self.aux_loss_names = aux_out.keys()
                aux_gt = obs["aux_info"]
                for aux_name in self.aux_loss_names:
                    num_vals = aux_out[aux_name].shape[-1]
                    aux_loss.append(
                        self.loss(aux_out[aux_name], aux_gt[aux_name].reshape(self.num_envs, -1)) #/ num_vals
                    )

            student_loss = (
                self.loss(actions_student["mus"], actions_teacher["mus"]) +
                self.loss(actions_student["sigmas"], actions_teacher["sigmas"])
            )
            total_loss += student_loss + sum(aux_loss)

            if self.rank == 0:
                self.log_information(log_counter, total_loss, aux_loss)
                
            log_counter += 1

            if self.is_rnn:
                if log_counter % self.seq_length == 0:
                    total_loss.backward()
                    self.optimizer.step()
                    # for param in self.student_model_ddp.parameters():
                    #     param.grad = None
                    self.optimizer.zero_grad()
                    for s in self.student_hidden_states:
                        s = s.detach()
                    total_loss = 0.
                    # self.scheduler.step()
            else:
                # for param in self.student_model_ddp.parameters():
                #     param.grad = None
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                # self.scheduler.step()
                total_loss = 0.
            
            stepping_actions = actions_student["actions"]
            obs, rew, out_of_reach, timed_out, info = self.env.step(
                stepping_actions.detach()
            )

            self.frame += self.num_envs
            self.current_rewards += rew.unsqueeze(-1)
            self.current_lengths += 1
            self.dones = out_of_reach | timed_out
            all_done_indices = self.dones.nonzero(as_tuple=False)

            if self.is_rnn and len(all_done_indices) > 0:
                for s in self.student_hidden_states:
                    s[:, all_done_indices, ...] *= 0.

            done_indices = all_done_indices[:]
            self.game_rewards.update(self.current_rewards[done_indices])
            self.game_lengths.update(self.current_lengths[done_indices])
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

            # if log_counter < self.num_warmup_steps:
            #     self.optimizer.param_groups[0]["lr"] = (log_counter / self.num_warmup_steps) * 1e-3 
            # elif log_counter % 10000 == 0:
            if log_counter % 10000 == 0 and log_counter > 10 and self.optimizer.param_groups[0]["lr"] > 1.2*1e-5:
                self.optimizer.param_groups[0]["lr"] /= 1.2
            #     # breakpoint()
            if self.rank == 0 and log_counter % 5000 == 0:
                ckpt_path = os.path.join(self.nn_dir,f"sh_{log_counter}_iters")
                self.save(ckpt_path)

        if self.use_wandb:
            wandb.finish()

    def log_information(self, log_counter, total_loss, aux_loss=None):
        student_loss = total_loss if aux_loss is None else total_loss - sum(aux_loss)
        
        if self.game_rewards.current_size > 0:
            mean_rewards = self.game_rewards.get_mean()
            mean_lengths = self.game_lengths.get_mean()
            self.mean_rewards = mean_rewards[0]
            for i in range(self.value_size):
                rewards_name = "rewards" if i == 0 else "rewards{0}".format(i)
                self.writer.add_scalar(
                    rewards_name + "/step", mean_rewards[i], self.frame
                )
                self.writer.add_scalar(
                    "average consecutive successes", self.ov_env.consecutive_successes.cpu().numpy()[0], self.frame
                )
                self.writer.add_scalar(
                    "total_loss", total_loss.detach().cpu().numpy(), self.frame
                )
                self.writer.add_scalar(
                    "imitation_loss", student_loss.detach().cpu().numpy(), self.frame
                )
                if self.use_wandb:
                    wandb.log({
                        "cs": self.ov_env.consecutive_successes.cpu().numpy()[0],
                        "imitation_loss": student_loss.detach().cpu().numpy(),
                        "total_loss": total_loss.detach().cpu().numpy(),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "iteration": self.frame
                    })
                if self.is_aux:
                    for idx, name in enumerate(self.aux_loss_names):
                        self.writer.add_scalar(
                            f"aux_loss_{name}", aux_loss[i].detach().cpu().numpy(), self.frame
                        )
                        if self.use_wandb:
                            wandb.log({
                                f"aux_loss_{name}": aux_loss[i].detach().cpu().numpy(),
                                "iteration": self.frame
                            })

        if log_counter % 10 == 0:
            print("="*10)
            print("LR: ", self.optimizer.param_groups[0]["lr"])
            print("Imitation Loss: ", student_loss)
            if self.is_aux:
                print("Aux Loss: ", aux_loss)
            print("Total Loss: ", total_loss)
            if self.game_rewards.current_size > 0:
                print("\tMean Rewards: ", mean_rewards)
                print("\tMean Length: ", mean_lengths)
                print("\tConsecutive Successes: ", self.ov_env.consecutive_successes)

    def get_actions(self, obs, policy_type):
        aux = None
        if policy_type == "student":
            batch_dict = {
                "is_train": True,
                "obs": obs[self.student_obs_type],
                "observations": obs[self.student_obs_type],
                "prev_actions": self.prev_actions_student,
            }
            if self.is_rnn:
                batch_dict["rnn_states"] = self.student_hidden_states
                batch_dict["seq_length"] = 1
                batch_dict["rnn_masks"] = None
            res_dict = self.student_model_ddp(batch_dict)
            mus = res_dict["mus"]
            sigmas = res_dict["sigmas"]
            if self.is_rnn:
                self.student_hidden_states = [s.detach() for s in res_dict["rnn_states"][0]]
            if self.is_aux:
                aux = res_dict["rnn_states"][-1]
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
    
    def loss(self, student_result, target_result):
        loss = l2(student_result, target_result)
        rnn_masks = None
        losses, sum_mask = torch_ext.apply_masks(
            [loss.unsqueeze(1)], rnn_masks
        )
        return losses[0]
    

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
