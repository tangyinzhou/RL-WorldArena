# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Neural network world model environment.

This environment replaces a physical simulator with a deep neural network
that predicts next observations and rewards given current observations and
actions.  The interface (obs format, action format) is intentionally kept
identical to RoboTwinEnv so that any policy / config that works with
RoboTwin can be switched to this environment with minimal changes.

Observation dict returned by reset() and step():
    main_images  : torch.Tensor  [N, H, W, 3]  uint8
    states       : torch.Tensor  [N, state_dim]  float32
    task_descriptions : list[str]  len == N

When model_path is None the world model is a randomly-initialised
placeholder – useful for verifying that the whole RLinf pipeline runs
end-to-end before training a real world model.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.envs.utils import list_of_dict_to_dict_of_list

__all__ = ["NNWorldModelEnv"]


# ---------------------------------------------------------------------------
# Lightweight placeholder world model
# ---------------------------------------------------------------------------

class _SimpleWorldModel(nn.Module):
    """Placeholder neural network world model.

    Input  : [images (N, 3, H, W) float], [states (N, state_dim)],
             [actions (N, action_dim)]
    Output : next_images (N, 3, H, W) float,
             next_states (N, state_dim),
             rewards     (N,),
             dones       (N,)  sigmoid logit → probability

    Replace this with a real trained model before running experiments.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        H, W = image_size
        # Tiny image encoder: 3×H×W → hidden_dim
        self.img_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, hidden_dim),
            nn.ReLU(),
        )
        # Fuse encoded image + state + action
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Heads
        self.next_state_head = nn.Linear(hidden_dim, state_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.done_head = nn.Linear(hidden_dim, 1)
        # Tiny image decoder: hidden_dim → 3×H×W
        _dec_h = H // 4
        _dec_w = W // 4
        self._dec_h = _dec_h
        self._dec_w = _dec_w
        self.img_proj = nn.Linear(hidden_dim, 16 * _dec_h * _dec_w)
        self.img_decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images: torch.Tensor,   # [N, 3, H, W] float32 in [0, 1]
        states: torch.Tensor,   # [N, state_dim]
        actions: torch.Tensor,  # [N, action_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.img_encoder(images)                          # [N, hidden]
        fused = self.fusion(torch.cat([feat, states, actions], dim=-1))  # [N, hidden]
        next_states = self.next_state_head(fused)                # [N, state_dim]
        rewards = self.reward_head(fused).squeeze(-1)            # [N]
        dones = torch.sigmoid(self.done_head(fused)).squeeze(-1) # [N]
        # decode next image
        dec_feat = self.img_proj(fused).view(
            -1, 16, self._dec_h, self._dec_w
        )
        next_images = self.img_decoder(dec_feat)                 # [N, 3, H, W]
        return next_images, next_states, rewards, dones


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class NNWorldModelEnv(gym.Env):
    """Neural-network world model environment compatible with RoboTwinEnv.

    Args:
        cfg: Hydra DictConfig containing environment configuration.
        num_envs: Number of parallel environments on this worker.
        seed_offset: Per-worker seed offset.
        total_num_processes: Total number of env processes in the cluster.
        worker_info: RLinf WorkerInfo object.
        record_metrics: Whether to track episode-level metrics.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info,
        record_metrics: bool = True,
    ):
        super().__init__()

        self.cfg = cfg
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        self.seed = cfg.seed + seed_offset
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.image_size: tuple[int, int] = tuple(cfg.image_size)  # (H, W)
        self.state_dim: int = cfg.state_dim
        self.action_dim: int = cfg.action_dim
        self.task_description: str = cfg.get("task_description", "")

        # Wrist camera config (for OpenPI compatibility)
        self.num_wrist_cameras: int = cfg.get("num_wrist_cameras", 2)

        self.video_cfg = cfg.video_cfg

        self._is_start = True
        self._elapsed_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # Build / load world model
        self.world_model = self._build_world_model().to(self.device).eval()

        # Current observations (will be populated on first reset)
        H, W = self.image_size
        self._current_images = torch.zeros(
            (self.num_envs, H, W, 3), dtype=torch.uint8, device=self.device
        )
        # Wrist images: [num_envs, num_wrist_cameras, H, W, 3]
        self._current_wrist_images = torch.zeros(
            (self.num_envs, self.num_wrist_cameras, H, W, 3),
            dtype=torch.uint8,
            device=self.device,
        )
        self._current_states = torch.zeros(
            (self.num_envs, self.state_dim), dtype=torch.float32, device=self.device
        )
        self._task_descriptions: list[str] = [self.task_description] * self.num_envs

        if self.record_metrics:
            self._init_metrics()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self) -> torch.Tensor:
        return self._elapsed_steps

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool):
        self._is_start = value

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _build_world_model(self) -> nn.Module:
        """Load a trained world model, or fall back to a random placeholder.

        Override this method to integrate a real trained model.
        """
        H, W = self.image_size
        model = _SimpleWorldModel(
            image_size=(H, W),
            state_dim=self.state_dim,
            action_dim=self.action_dim,
        )
        model_path = self.cfg.get("model_path", None)
        if model_path is not None and os.path.isfile(model_path):
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
        # else: use random weights as placeholder
        return model

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0.0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0.0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(
        self, step_reward: torch.Tensor, infos: dict
    ) -> dict:
        self.returns += step_reward
        episode_info: dict = {}
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self._elapsed_steps.clone()
        episode_info["reward"] = (
            self.returns / self._elapsed_steps.clamp(min=1)
        ).clone()
        infos["episode"] = episode_info
        return infos

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> dict:
        """Pack current internal state into the standard obs dict."""
        return {
            "main_images": self._current_images.clone(),
            "wrist_images": self._current_wrist_images.clone(),
            "states": self._current_states.clone(),
            "task_descriptions": list(self._task_descriptions),
        }

    def _random_reset_obs(self, env_idx: Optional[list[int]] = None):
        """Randomise observations for the given env indices (placeholder).

        In a real setting you would load initial frames from a reference
        dataset.  Here we simply draw uniform random pixels and states.
        """
        if env_idx is None:
            env_idx = list(range(self.num_envs))

        H, W = self.image_size
        for idx in env_idx:
            self._current_images[idx] = torch.randint(
                0, 256, (H, W, 3), dtype=torch.uint8, device=self.device
            )
            # Generate wrist images for each camera
            for cam_idx in range(self.num_wrist_cameras):
                self._current_wrist_images[idx, cam_idx] = torch.randint(
                    0, 256, (H, W, 3), dtype=torch.uint8, device=self.device
                )
            self._current_states[idx] = torch.randn(
                self.state_dim, device=self.device
            )

    def _apply_world_model(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward pass of the world model.

        Args:
            actions: float32 tensor of shape [N, action_dim].

        Returns:
            next_images : [N, H, W, 3] uint8
            next_states : [N, state_dim] float32
            rewards     : [N] float32
            dones       : [N] bool
        """
        H, W = self.image_size
        # Convert images from [N, H, W, 3] uint8 → [N, 3, H, W] float32 [0,1]
        imgs_f = (
            self._current_images.permute(0, 3, 1, 2).float() / 255.0
        )  # [N, 3, H, W]

        with torch.no_grad():
            next_imgs_f, next_states, rewards, done_probs = self.world_model(
                imgs_f.to(self.device),
                self._current_states.to(self.device),
                actions.to(self.device),
            )

        # Convert next images back to uint8 [N, H, W, 3]
        next_images = (
            (next_imgs_f.clamp(0.0, 1.0) * 255.0)
            .permute(0, 2, 3, 1)
            .to(torch.uint8)
        )

        dones = done_probs > 0.5

        return next_images, next_states, rewards, dones

    # ------------------------------------------------------------------
    # Reward helper
    # ------------------------------------------------------------------

    def _calc_step_reward(
        self, raw_rewards: torch.Tensor, terminations: torch.Tensor
    ) -> torch.Tensor:
        reward = self.cfg.reward_coef * raw_rewards
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward.clone()
            return reward_diff
        return reward

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ) -> tuple[dict, dict]:
        """Reset environments and return initial observations.

        Args:
            env_idx: Indices of environments to reset.  None resets all.
            env_seeds: Ignored (kept for API compatibility with RoboTwinEnv).

        Returns:
            obs   : observation dict
            infos : empty dict
        """
        if self._is_start:
            self._is_start = False

        if env_idx is None:
            env_idx = list(range(self.num_envs))
        elif isinstance(env_idx, (int, torch.Tensor)):
            env_idx = [int(env_idx)] if isinstance(env_idx, int) else env_idx.tolist()

        self._random_reset_obs(env_idx)
        self._reset_metrics(env_idx)

        return self._make_obs(), {}

    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
        auto_reset: bool = True,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance the environment by one step.

        Args:
            actions: [N, action_dim] or [N, 1, action_dim].
            auto_reset: Whether to auto-reset done environments.

        Returns:
            obs, step_reward, terminations, truncations, infos
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()

        # Support chunk input with chunk_size == 1
        if actions.dim() == 3:
            actions = actions[:, 0, :]  # [N, action_dim]

        actions = actions.to(self.device)

        # World model forward pass
        next_images, next_states, raw_rewards, dones = self._apply_world_model(actions)

        # Update internal state
        self._current_images = next_images
        # Generate next wrist images (placeholder: same random generation)
        H, W = self.image_size
        for idx in range(self.num_envs):
            for cam_idx in range(self.num_wrist_cameras):
                self._current_wrist_images[idx, cam_idx] = torch.randint(
                    0, 256, (H, W, 3), dtype=torch.uint8, device=self.device
                )
        self._current_states = next_states

        self._elapsed_steps += 1
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        terminations = dones
        truncations = truncated

        step_reward = self._calc_step_reward(raw_rewards, terminations)
        infos = self._record_metrics(step_reward, {})

        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)
            if self.record_metrics and "success" in infos:
                infos["episode"]["success_at_end"] = dones.clone()

        done_combined = torch.logical_or(terminations, truncations)
        _auto_reset = auto_reset and self.auto_reset
        if done_combined.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(done_combined, self._make_obs(), infos)
        else:
            obs = self._make_obs()

        return obs, step_reward, terminations, truncations, infos

    def chunk_step(
        self, chunk_actions: Union[torch.Tensor, np.ndarray]
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Advance the environment by a chunk of actions.

        Args:
            chunk_actions: [N, chunk_size, action_dim]

        Returns:
            obs_list          : list of obs dicts (one per chunk step)
            chunk_rewards     : [N, chunk_size]
            chunk_terminations: [N, chunk_size]
            chunk_truncations : [N, chunk_size]
            infos_list        : list of info dicts (one per chunk step)
        """
        if isinstance(chunk_actions, np.ndarray):
            chunk_actions = torch.from_numpy(chunk_actions).float()

        chunk_actions = chunk_actions.to(self.device)
        num_envs, chunk_size, _ = chunk_actions.shape

        obs_list = []
        infos_list = []
        chunk_rewards_list = []
        chunk_terminations_list = []
        chunk_truncations_list = []

        for t in range(chunk_size):
            obs, step_reward, terminations, truncations, infos = self.step(
                chunk_actions[:, t, :], auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards_list.append(step_reward)
            chunk_terminations_list.append(terminations)
            chunk_truncations_list.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards_list, dim=1)         # [N, chunk_size]
        chunk_terminations = torch.stack(chunk_terminations_list, dim=1)
        chunk_truncations = torch.stack(chunk_truncations_list, dim=1)

        past_dones = torch.logical_or(
            chunk_terminations.any(dim=1), chunk_truncations.any(dim=1)
        )

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        return obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list

    def _handle_auto_reset(
        self,
        dones: torch.Tensor,
        extracted_obs: dict,
        infos: dict,
    ) -> tuple[dict, dict]:
        def clone_value(v):
            if isinstance(v, torch.Tensor):
                return v.clone()
            elif isinstance(v, list):
                return list(v)
            elif isinstance(v, dict):
                return {k: clone_value(val) for k, val in v.items()}
            return v

        final_obs = {k: clone_value(v) for k, v in extracted_obs.items()}
        final_info = {k: clone_value(v) for k, v in infos.items()}
        env_idx = torch.arange(self.num_envs, device=self.device)[dones].tolist()

        obs, infos = self.reset(env_idx=env_idx)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def close(self):
        """Clean up resources."""
        pass

    def sample_action_space(self) -> np.ndarray:
        """Return random actions for all environments."""
        return np.random.randn(self.num_envs, self.action_dim)

    def update_reset_state_ids(self, env_idx=None):
        """Update reset state IDs for environments.

        For NNWorldModelEnv, this is a no-op since we don't use fixed seeds.
        This method exists for API compatibility with RoboTwinEnv.

        Args:
            env_idx: Optional indices of environments to update. Ignored.
        """
        pass
