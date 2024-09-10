# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as pth_transforms
import torchvision.models as models
import torch.distributed as dist
from torch.autograd import Variable
from PIL import Image
from collections.abc import Sequence
from contextlib import nullcontext

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, RigidObject
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import TiledCamera, save_images_to_file
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from omni.isaac.lab_tasks.direct.allegro_hand import AllegroHandEnvCfg
from omni.isaac.lab_tasks.direct.shadow_hand import ShadowHandEnvCfg, ShadowHandCameraEnvCfg


class InHandManipulationCameraEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.world_size = int(os.environ['WORLD_SIZE'])  # Total number of processes
        self.rank = int(os.environ['RANK'])  # Global rank of this process
        self.local_rank = int(os.environ['LOCAL_RANK']) # local rank of the process 
        dist.init_process_group("nccl", rank=self.rank, world_size=self.world_size)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        # default goal positions
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # initialize goal marker
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        # embedding model settings
        self.finetune_backbone = self.cfg.finetune_backbone
        self.backbone = self._get_embeddings_model(model_name=self.cfg.embedding_model)
        if self.finetune_backbone:
            self.backbone_ddp = DDP(self.backbone, device_ids=[self.local_rank], find_unused_parameters=True)
        self.img_width = self.cfg.tiled_camera.width
        self.img_height = self.cfg.tiled_camera.height
        self.batch_size = min(32, self.num_envs)
        self.embedding_size = self.cfg.embedding_size
        self.padded_width, self.padded_height = self._get_padded_dims(
            self.img_width,
            self.img_height,
            n=14
        )
        self.num_observations = self.cfg.num_observations
        self.num_teacher_observations = self.cfg.num_teacher_observations
        self.teacher_observations_type = self.cfg.teacher_obs_type
        self.img = torch.zeros(
            (self.num_envs, self.padded_height, self.padded_width, 3), 
            dtype=torch.float32
        ).to(self.device)
        self.transform = pth_transforms.Compose([
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.embeddings = torch.zeros(
            (self.num_envs, self.embedding_size), dtype=torch.float32, requires_grad=self.finetune_backbone
        ).to(self.device)

        self.visualize_marker = self.cfg.visualize_marker

    def _get_padded_dims(self, width, height, n):
        padded_dims = list()

        for x in [width, height]:
            remainder = x % n
            if remainder == 0:
                padded_dims.append(x)
            else:
                padded_dims.append(x + (n - remainder))

        return tuple(padded_dims)

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articultion to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _get_embeddings_model(self, model_name="dino"):
        if model_name == "dino":
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        elif model_name == "resnet":
            resnet18 = models.resnet18(pretrained=True)
            modules=list(resnet18.children())[:-1]
            resnet18=nn.Sequential(*modules)
            for p in resnet18.parameters():
                p.requires_grad = self.finetune_backbone
            model = resnet18
        elif model_name == "theia":
            import transformers
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                "theaiinstitute/theia-tiny-patch16-224-cdiv", trust_remote_code=True
            ).backbone
        elif model_name == "convnext":
            convnext = models.convnext_tiny(pretrained=True)
            modules = list(convnext.children())[:-1]
            convnext = nn.Sequential(*modules)
            for p in convnext.parameters():
                p.requires_grad = False
            model = convnext
        model.to(self.device)
        if not self.finetune_backbone: model.eval()
        return model

    def _get_embeddings(self, observations):
        self.img[:, :self.img_height, :self.img_width, :] = observations["policy"]
        transformed_img = self.transform(self.img.permute(0, 3, 1, 2))

        context_manager = torch.no_grad() if not self.finetune_backbone else nullcontext()

        # if we are manually finetuning we should zero out the gradients because we are doing in-place allocations
        if self.finetune_backbone: self.embeddings = self.embeddings.detach().clone().requires_grad_(True)
        embedding_list = []
        backbone = self.backbone if not self.finetune_backbone else self.backbone_ddp
        with context_manager:
            for i in range(0, self.num_envs, self.batch_size):
                bound = min(self.batch_size+i, self.num_envs)
                if self.cfg.embedding_model == "theia":
                    embeddings = backbone(self.img[i:bound])[:, 0, :].view(bound-i, -1)
                else:
                    embeddings = backbone(transformed_img[i:bound])
                embedding_list.append(embeddings[:, :, 0, 0])
        self.embeddings = torch.cat(embedding_list, dim=0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        # imgs = self._tiled_camera.data.output["rgb"].clone()
        # np_imgs = (imgs.cpu().numpy()*255).astype(np.uint8)
        # im = Image.fromarray(np_imgs[0])
        # breakpoint()
        if self.cfg.asymmetric_obs or self.cfg.obs_type == "full_rma":
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        elif self.cfg.obs_type == "embedding":
            self._get_embeddings({"policy": self._tiled_camera.data.output["rgb"].clone()})
            obs = self.compute_embeddings_observation_no_vel()
        elif self.cfg.obs_type == "full_rma":
            self._get_embeddings({"policy": self._tiled_camera.data.output["rgb"].clone()})
            obs = self.compute_full_rma_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}

        if "rma" in self.cfg.obs_type:
            observations["expert_policy"] = self.compute_full_rma_observations()
        elif self.teacher_observations_type == "dextreme":
            observations["expert_policy"] = self.compute_dextreme_observations()
        else:
            observations["expert_policy"] = self.compute_full_observations()
        self.keypoints = gen_keypoints(pose=torch.cat((self.object_pos, self.object_rot), dim=1))
        aux_info = {
            "keypoints": self.keypoints,
            "obj_vel": torch.cat([self.object_linvel, self.cfg.vel_obs_scale*self.object_angvel], dim=-1),
            "finger_tip_forces": self.cfg.force_torque_obs_scale * self.fingertip_force_sensors.view(
                self.num_envs, self.num_fingertips * 6)
        }
        observations["aux_info"] = aux_info
        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached
        return out_of_reach, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        self._reset_target_pose(env_ids)

        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # reset goal rotation
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        self.goal_rot[env_ids] = new_rot
        goal_pos = self.goal_pos + self.scene.env_origins
        if self.visualize_marker:
            self.goal_markers.visualize(goal_pos, self.goal_rot)
            for i in range(8):
                self.kpt_markers[i].visualize(self.keypoints[:, i, :] + self.scene.env_origins, self.goal_rot)

        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_embeddings_observations(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),  # 0:24
                self.cfg.vel_obs_scale * self.hand_dof_vel,  # 24:48
                # object
                self.embeddings,
                # goal
                self.goal_rot,  # 64:68
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),  # 72:87
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),  # 87:107
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),  # 107:137
                # actions
                self.actions,  # 137:157
            ),
            dim=-1,
        )
        return obs

    def compute_embeddings_observation_no_vel(self):
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),  # 0:24
                # object
                self.embeddings,
                # goal
                self.goal_rot,  # 64:68
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),  # 72:87
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),  # 87:107
                # actions
                self.actions,  # 137:157
            ),
            dim=-1,
        )
        return obs

    def compute_full_rma_observations(self):
        obs = torch.cat(
            (
                # env enc obs
                self.cfg.vel_obs_scale * self.hand_dof_vel, # 0:24
                self.object_pos, # 24:27
                self.object_rot, # 27:31
                self.object_linvel,  # 31:34
                self.cfg.vel_obs_scale * self.object_angvel, # 34:37
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3), # 37:52
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4), # 52:72
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6), # 72:102
                # self.cfg.force_torque_obs_scale
                # * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6), # 102:132
                # base policy obs
                self.goal_rot, # 132:136
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits), # 136:160
                self.actions, # 160:180
            ),
            dim=-1,
        )
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states


    def compute_rma_embeddings_observations(self):
        obs = torch.cat(
            (
                self.goal_rot, # 0:4
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits), # 4:28
                self.actions, # 28:48
                self.embeddings, # 48: 48+num_embeddings
            ),
            dim=-1
        )
        return obs

    def compute_dextreme_observations(self):
        object_pos = self.object_pos.clone()
        object_pos[:, 2] -= 0.04
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                # goal
                object_pos,
                self.object_rot[:, [1, 2, 3, 0]],
                torch.tensor([0.1064, 0.0088, 0.0175-0.04]).repeat(self.num_envs, 1).to(self.device),
                self.goal_rot[:, [1, 2, 3, 0]],
                quat_mul(self.object_rot[:, [1, 2, 3, 0]], quat_conjugate(self.goal_rot[:, [1, 2, 3, 0]])),
                # actions
                self.actions,
            ),
            dim=-1
        )
        return obs


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes


@torch.jit.script
def compute_rewards_dextreme(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    joint_vel: torch.Tensor,
    vel_penalty_scale: float,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty_rew = torch.sum(actions**2, dim=-1) * action_penalty_scale
    joint_vel_penalty_rew = torch.sum(joint_vel**2, dim=-1) * vel_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty_rew + joint_vel_penalty_rew

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes



@torch.jit.script
def local_to_world_space(pos_offset_local: torch.Tensor, pose_global: torch.Tensor):
    """ Convert a point from the local frame to the global frame
    Args:
        pos_offset_local: Point in local frame. Shape: [N, 3]
        pose_global: The spatial pose of this point. Shape: [N, 7]
    Returns:
        Position in the global frame. Shape: [N, 3]
    """
    quat_pos_local = torch.cat(
        [
            torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device), 
            pos_offset_local
        ], dim=-1
    )
    quat_trans = torch.cat(
        [
            torch.zeros(pos_offset_local.shape[0], 1, dtype=torch.float32, device=pos_offset_local.device), 
            pose_global[:, 0:3]
        ], dim=-1
    )
    quat_global = pose_global[:, 3:7]
    quat_global_conj = quat_conjugate(quat_global)
    pos_offset_global = quat_mul(quat_global, quat_mul(quat_pos_local, quat_global_conj))[:, 1:4]

    result_pos_gloal = pos_offset_global + pose_global[:, 0:3]

    return result_pos_gloal


    size = [2*0.03, 2*0.03, 2*0.03]

@torch.jit.script
def gen_keypoints(
    pose: torch.Tensor, 
    num_keypoints: int = 8, 
    size: Tuple[float, float, float] = (2*0.03, 2*0.03, 2*0.03)
):

    num_envs = pose.shape[0]
    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)
    for i in range(num_keypoints):
        # which dimensions to negate
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf
