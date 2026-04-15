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

"""Genie neural network world model environment.

Integrates the Genie world model (VQModel tokenizer + STMaskGIT) into the
RLinf framework. Data loading uses GenieVal10Dataset which reads val10-format
directories (mp4 video + actions.npy + states.npy + instructions.json).
On each reset() the first frame of the selected episode is encoded into Genie
tokens; on each step() the STMaskGIT generates the next-frame token
auto-regressively, then VQModel decodes it back to pixels.

Observation dict (compatible with OpenPI):
    main_images       : torch.Tensor [N, H, W, 3] uint8
    states            : torch.Tensor [N, state_dim] float32
    task_descriptions : list[str] len == N
    wrist_images      : torch.Tensor [N, num_wrist_cameras, H, W, 3] uint8
                        (only present when num_wrist_cameras > 0)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange

from rlinf.data.datasets.genie_val10_dataset import GenieVal10Dataset
from rlinf.envs.world_model.base_world_env import BaseWorldEnv

__all__ = ["GenieWorldModelEnv"]


class GenieWorldModelEnv(BaseWorldEnv):
    """Genie world model environment compatible with RLinf's embodied runner.

    Args:
        cfg: Hydra DictConfig with environment configuration.
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
        worker_info=None,
        record_metrics: bool = True,
    ):
        # BaseWorldEnv.__init__ calls self._build_dataset(cfg) and self._init_metrics()
        super().__init__(
            cfg, num_envs, seed_offset, total_num_processes, worker_info, record_metrics
        )

        # ------------------------------------------------------------------
        # Genie model loading (lazy imports to keep RLinf importable without
        # the Genie dependencies installed unless this env is actually used)
        # ------------------------------------------------------------------
        from magvit2.config import VQConfig  # noqa: PLC0415
        from magvit2.models.lfqgan import VQModel  # noqa: PLC0415
        from genie.config import GenieConfig  # noqa: PLC0415
        from genie.st_mask_git import STMaskGIT  # noqa: PLC0415

        self.tokenizer = VQModel(VQConfig(), ckpt_path=cfg.tokenizer_ckpt).to(self.device)
        self.tokenizer.eval()

        genie_config = GenieConfig.from_pretrained(cfg.genie_config)
        genie_config.H = cfg.genie_H  # latent grid height (default 16)
        genie_config.W = cfg.genie_W  # latent grid width  (default 16)
        self.genie = STMaskGIT.from_pretrained(cfg.genie_checkpoint).to(self.device)
        self.genie.eval()
        self.genie_config = genie_config

        # ------------------------------------------------------------------
        # Inference hyper-parameters (mirror GenieWorldModelAgent in
        # rollout_policy_genie_closedloop.py)
        # ------------------------------------------------------------------
        self.num_prompt_frames: int = cfg.num_prompt_frames  # 1
        self.maskgit_steps: int = cfg.maskgit_steps          # 2
        self.temperature: float = cfg.temperature            # 0.0
        self.resize: int = cfg.resize                        # 256
        self.action_dim: int = cfg.action_dim                # 14
        self.image_size: tuple[int, int] = tuple(cfg.image_size)  # (256, 256)
        self.num_wrist_cameras: int = cfg.get("num_wrist_cameras", 0)

        # ------------------------------------------------------------------
        # Reset-state management (same pattern as WanEnv)
        # ------------------------------------------------------------------
        self.use_fixed_reset_state_ids: bool = cfg.use_fixed_reset_state_ids
        self.group_size: int = cfg.group_size
        self.num_group: int = num_envs // cfg.group_size
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ------------------------------------------------------------------
        # Per-environment state buffers
        # ------------------------------------------------------------------
        gH, gW = cfg.genie_H, cfg.genie_W
        # Initialise tokens to mask_token_id so we can call decode before first reset
        self._current_tokens = torch.full(
            (num_envs, gH, gW),
            self.genie.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        self._current_states = torch.zeros(
            (num_envs, cfg.state_dim), dtype=torch.float32, device=self.device
        )
        self._task_descriptions: list[str] = [""] * num_envs
        self._elapsed_steps: int = 0

    # ------------------------------------------------------------------
    # BaseWorldEnv abstract method
    # ------------------------------------------------------------------

    def _build_dataset(self, cfg) -> GenieVal10Dataset:
        """Return the dataset built from a val10-format directory.

        Expects ``cfg.val10_root`` to point at the val10 dataset root.
        Optionally ``cfg.task_filter`` (list[str]) restricts which tasks
        are included.
        """
        task_filter = list(cfg.task_filter) if cfg.get("task_filter") else None
        return GenieVal10Dataset(
            val10_root=cfg.val10_root,
            task_filter=task_filter,
            resize=cfg.resize,
        )

    # ------------------------------------------------------------------
    # Reset-state helpers
    # ------------------------------------------------------------------

    def update_reset_state_ids(self) -> None:
        """Randomly sample episode indices for each env group."""
        total = len(self.dataset)
        ids = torch.randint(
            low=0,
            high=total,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = ids.repeat_interleave(self.group_size).to(self.device)

    # ------------------------------------------------------------------
    # Genie encode / decode helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_frame_tensor(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Encode a single frame tensor to Genie token IDs.

        Args:
            img_tensor: (3, H, W) float32 in [0, 1], CHW format.

        Returns:
            token_ids: (genie_H, genie_W) int64 tensor.
        """
        h = w = self.resize
        frame = TF.resize(img_tensor, [h, w])  # (3, h, w) float [0,1]
        # Normalise to [-1, 1] and add batch dim
        # Use the encoder conv_in weight dtype to ensure compatibility
        encoder_dtype = self.tokenizer.encoder.conv_in.weight.dtype
        frame = (frame * 2.0 - 1.0).unsqueeze(0).to(device=self.device, dtype=encoder_dtype)
        quant, _, _, _ = self.tokenizer.encode(frame)
        # quant shape: (1, C, gH, gW); bits_to_indices expects (..., C)
        token_ids = self.tokenizer.quantize.bits_to_indices(
            quant.permute(0, 2, 3, 1) > 0
        )  # (1, gH, gW)
        return token_ids.squeeze(0).long()  # (gH, gW)

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode a batch of Genie token IDs to pixel frames.

        Mirrors GenieWorldModelAgent.decode_tokens in
        rollout_policy_genie_closedloop.py exactly.

        Args:
            tokens: (N, gH, gW) int64 on any device.

        Returns:
            frames: (N, H, W, 3) uint8 CPU tensor.
        """
        decoded_batches = []
        batch_size = 8
        # Use the decoder conv_in weight dtype to ensure compatibility
        decoder_dtype = self.tokenizer.decoder.conv_in.weight.dtype
        for i in range(0, tokens.shape[0], batch_size):
            batch = tokens[i : i + batch_size].to(device=self.device, dtype=torch.long)
            codebook_dim = self.tokenizer.quantize.codebook_dim
            bhwc = batch.shape + (codebook_dim,)

            if getattr(self.tokenizer, "use_ema", False):
                with self.tokenizer.ema_scope():
                    quant = self.tokenizer.quantize.get_codebook_entry(
                        rearrange(batch, "b h w -> b (h w)"), bhwc=bhwc
                    ).flip(1)
                    dec = self.tokenizer.decode(
                        quant.to(device=self.device, dtype=decoder_dtype)
                    )
            else:
                quant = self.tokenizer.quantize.get_codebook_entry(
                    rearrange(batch, "b h w -> b (h w)"), bhwc=bhwc
                ).flip(1)
                dec = self.tokenizer.decode(
                    quant.to(device=self.device, dtype=decoder_dtype)
                )

            # dec: (B, 3, H, W) float in [-1, 1]  →  uint8 [0, 255]
            rescaled = (dec.detach().cpu().float() + 1.0) * 127.5
            decoded_batches.append(torch.clamp(rescaled, 0, 255).to(torch.uint8))

        decoded = torch.cat(decoded_batches, dim=0)  # (N, 3, H, W)
        return decoded.permute(0, 2, 3, 1)  # (N, H, W, 3) uint8

    # ------------------------------------------------------------------
    # World model forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward_wm(
        self,
        prompt_tokens: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Auto-regressively generate the next frame token.

        The Genie model's pos_embed_TSC has a fixed shape of (1, T, S, D)
        where T == config.T (16 for the n32_h8_d512 model).  The full window
        of size T must therefore always be provided to maskgit_generate,
        matching the behaviour in rollout_policy_genie_closedloop.py:

            window_size = num_prompt_frames + chunk_size  # == config.T

        Here we generate only one new frame (chunk_size=1) but still pad the
        prompt_THW tensor to T frames (filling extra slots with mask_token_id)
        so that the positional embedding broadcast succeeds.

        Args:
            prompt_tokens: (gH, gW) int64 – current frame token IDs.
            action: (action_dim,) float32 – action to apply.

        Returns:
            pred_token: (gH, gW) int64 – predicted next-frame token IDs.
        """
        gH = self.genie_config.H
        gW = self.genie_config.W
        num_prompt = self.num_prompt_frames  # 1
        # The model requires window_size == config.T (e.g. 16).
        # We always allocate the full T-frame window and predict frame at
        # index num_prompt (i.e. out_t=1).
        model_T = self.genie_config.T

        prompt_THW = torch.full(
            (1, model_T, gH, gW),
            self.genie.mask_token_id,
            dtype=torch.long,
            device=self.device,
        )
        prompt_THW[:, :num_prompt] = prompt_tokens.unsqueeze(0)

        # Build action tensor: (1, model_T, action_dim)
        # Place the real action at the predict-frame slot (index num_prompt);
        # all other slots are zero-padded (they correspond to masked frames
        # whose actions are not needed for single-step generation).
        input_actions = torch.zeros(
            1, model_T, self.action_dim, dtype=torch.float32, device=self.device
        )
        input_actions[:, num_prompt] = action.unsqueeze(0)

        sample_HW, _ = self.genie.maskgit_generate(
            prompt_THW,
            input_actions,
            out_t=num_prompt,
            maskgit_steps=self.maskgit_steps,
            temperature=self.temperature,
        )
        return sample_HW.squeeze(0)  # (gH, gW)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _make_obs(self) -> dict:
        """Decode current tokens into the standard obs dict."""
        decoded = self.decode_tokens(self._current_tokens)  # (N, raw_H, raw_W, 3) uint8

        H, W = self.image_size
        if decoded.shape[1] != H or decoded.shape[2] != W:
            decoded_f = decoded.float().permute(0, 3, 1, 2) / 255.0  # (N,3,h,w)
            decoded_f = F.interpolate(
                decoded_f.to(self.device), size=(H, W), mode="bilinear", align_corners=False
            )
            decoded = (decoded_f * 255.0).byte().permute(0, 2, 3, 1)  # (N,H,W,3)

        obs: dict = {
            "main_images": decoded,
            "states": self._current_states.clone(),
            "task_descriptions": list(self._task_descriptions),
        }
        if self.num_wrist_cameras > 0:
            # Reuse main image as wrist placeholder (same as NNWorldModelEnv)
            obs["wrist_images"] = decoded.unsqueeze(1).expand(
                -1, self.num_wrist_cameras, -1, -1, -1
            ).clone()
        return obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed=None,
        options=None,
        episode_indices=None,
    ) -> tuple[dict, dict]:
        """Reset environments and return initial observations.

        Args:
            seed: Ignored (kept for API compatibility).
            options: Ignored (kept for API compatibility).
            episode_indices: Optional tensor/array of episode indices to load.
                             None means random sampling.

        Returns:
            obs   : observation dict
            infos : empty dict
        """
        self.elapsed_steps = 0

        if self.is_start:
            if self.use_fixed_reset_state_ids:
                episode_indices = self.reset_state_ids
            self._is_start = False

        if episode_indices is None:
            episode_indices = torch.randint(
                low=0,
                high=len(self.dataset),
                size=(self.num_envs,),
                generator=self._generator,
            )
        if isinstance(episode_indices, torch.Tensor):
            episode_indices = episode_indices.cpu().numpy()

        for env_idx, ep_idx in enumerate(episode_indices):
            episode_data = self.dataset[int(ep_idx)]

            # image: (H, W, 3) uint8 from GenieVal10Dataset
            # encode_frame_tensor expects (3, H, W) float [0, 1]
            img_np = episode_data["image"]  # (H, W, 3) uint8
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0  # (3,H,W) [0,1]
            self._current_tokens[env_idx] = self.encode_frame_tensor(img_tensor)

            state_np = episode_data.get("state")
            if state_np is not None:
                # Adapt state dimensionality to match target state_dim.
                #
                # The val10 dataset stores 16-D raw states (RoboTwin raw16
                # layout: left-arm 6 joints + left-gripper + right-arm 6
                # joints + right-gripper + 2 extra dims at index 6 & 14).
                # The Genie WM was trained with joint14 action conditions
                # (indices [0,1,2,3,4,5,7,8,9,10,11,12,13,15] from raw16),
                # matching the layout used in rollout_policy_genie_closedloop
                # with stat file robotwin2_joint_14d/stat.json.
                # The policy server (policy_http_common.map_robotwin_state_to_policy14)
                # uses raw14 (indices 0-13) for Pi0 input instead.
                #
                # Rule applied here:
                #   raw16 (16D) -> joint14 (14D): skip indices 6 and 14
                #   raw16 (16D) -> any other target_dim: generic truncation/pad
                #   already target_dim: use as-is
                target_dim = self._current_states.shape[1]
                src_dim = state_np.shape[0]
                if src_dim == 16 and target_dim == 14:
                    # raw16 -> joint14: skip the two extra dims at index 6 & 14
                    _JOINT14_IDX = np.array(
                        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 15],
                        dtype=np.int64,
                    )
                    state_np = state_np[_JOINT14_IDX]
                elif src_dim > target_dim:
                    state_np = state_np[:target_dim]
                elif src_dim < target_dim:
                    pad = np.zeros(target_dim - src_dim, dtype=np.float32)
                    state_np = np.concatenate([state_np, pad])
                self._current_states[env_idx] = torch.from_numpy(state_np).to(self.device)
            self._task_descriptions[env_idx] = episode_data.get("task", "")

        self._reset_metrics()
        return self._make_obs(), {}

    def step(
        self,
        actions,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance each environment by one action step.

        Args:
            actions: (N, action_dim) float32 tensor or numpy array.

        Returns:
            obs, step_reward, terminations, truncations, infos
        """
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        actions = actions.to(self.device)  # (N, action_dim)

        # Per-env forward pass (sequential; can be batched later for speed)
        next_tokens = []
        for i in range(self.num_envs):
            next_tokens.append(self.forward_wm(self._current_tokens[i], actions[i]))
        self._current_tokens = torch.stack(next_tokens, dim=0)  # (N, gH, gW)

        # Update states with the applied actions, mirroring the closed-loop
        # script where `current_policy_state = chunk_actions[-1]` after each
        # chunk.  In the single-step case the latest action *is* the current
        # robot state fed to the policy on the next observation.
        self._current_states = actions.clone()

        self._elapsed_steps += 1
        truncated = torch.tensor(
            [self._elapsed_steps >= self.cfg.max_episode_steps] * self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        terminations = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        raw_rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        step_reward = self.cfg.reward_coef * raw_rewards

        infos = self._record_metrics(step_reward, terminations, {})

        if self.ignore_terminations:
            terminations = torch.zeros_like(terminations)

        return self._make_obs(), step_reward, terminations, truncated, infos

    def chunk_step(
        self,
        chunk_actions,
    ) -> tuple[list, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        """Advance environments by a chunk of actions.

        Args:
            chunk_actions: (N, chunk_size, action_dim) float32.

        Returns:
            obs_list, chunk_rewards [N,T], chunk_terminations [N,T],
            chunk_truncations [N,T], infos_list
        """
        if isinstance(chunk_actions, np.ndarray):
            chunk_actions = torch.from_numpy(chunk_actions).float()
        chunk_actions = chunk_actions.to(self.device)
        _, chunk_size, _ = chunk_actions.shape

        obs_list, rewards_list, term_list, trunc_list, infos_list = [], [], [], [], []
        for t in range(chunk_size):
            obs, r, term, trunc, info = self.step(chunk_actions[:, t, :])
            obs_list.append(obs)
            rewards_list.append(r)
            term_list.append(term)
            trunc_list.append(trunc)
            infos_list.append(info)

        return (
            obs_list,
            torch.stack(rewards_list, dim=1),   # (N, chunk_size)
            torch.stack(term_list, dim=1),
            torch.stack(trunc_list, dim=1),
            infos_list,
        )

    def close(self) -> None:
        """Release resources."""
        pass
