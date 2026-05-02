# GENIE 世界模型集成为 RL 环境指南

本文档对应 `docs/world_model_rl_integration.md` 的通用指南，专门说明如何将基于 **STMaskGIT + MagViT2** 的 GENIE 世界模型（即 `roboscape/genie/` 目录下训练的模型）接入 RLinf，作为 RL 训练的仿真环境。

**Pipeline 三件套：**
- 数据编码：`roboscape/genie/encode_vla_robotwin.sh` → `encode_vla_robotwin.py`
- 模型训练：`roboscape/genie/train_action.sh` → `train.py`
- 推理可视化：`roboscape/genie/generate.sh` → `genie/st_mask_git.py::generate()`

现有参考实现（`roboscape/genie/roboscape_env.py`）已经把这套模型接成了一个独立的 gym Env，但接口与 RLinf `BaseWorldEnv` 不同。本文描述如何按 RLinf 规范重新实现。

---

## 目录

1. [架构差异与核心挑战](#1-架构差异与核心挑战)
2. [数据表示：Token 空间 vs Pixel 空间](#2-数据表示token-空间-vs-pixel-空间)
3. [Step 1：注册新环境类型](#step-1注册新环境类型)
4. [Step 2：实现 GenieEnv 子类](#step-2实现-genieenv-子类)
   - [4.1 `__init__` 构造函数](#41-__init__-构造函数)
   - [4.2 `_build_dataset`：数据集加载](#42-_build_dataset数据集加载)
   - [4.3 `reset`：环境重置](#43-reset环境重置)
   - [4.4 `_infer_next_chunk_frames`：世界模型推理（核心接口）](#44-_infer_next_chunk_frames世界模型推理核心接口)
   - [4.5 `_infer_next_chunk_rewards`：奖励计算](#45-_infer_next_chunk_rewards奖励计算)
   - [4.6 `_wrap_obs`：观测打包（策略输入接口）](#46-_wrap_obs观测打包策略输入接口)
   - [4.7 `chunk_step`：驱动单步推进](#47-chunk_step驱动单步推进)
   - [4.8 `offload` / `onload`：显存管理](#48-offload--onload显存管理)
5. [Step 3：动作适配（prepare_actions）](#step-3动作适配prepare_actions)
6. [Step 4：编写环境配置 YAML](#step-4编写环境配置-yaml)
7. [Step 5：编写训练配置 YAML](#step-5编写训练配置-yaml)
8. [关键约束与一致性检查清单](#关键约束与一致性检查清单)
9. [与 Wan / OpenSora 实现的主要差异对比](#与-wan--opensora-实现的主要差异对比)

---

## 1.1 可直接 import 的 roboscape/genie 模块

在实现 `GenieEnv` 时，以下模块和类可以直接从原仓库 import，无需重写：

```python
# ---- 世界模型核心 ----
from roboscape.genie.genie.st_mask_git import STMaskGIT        # 世界模型主干，含 maskgit_generate()
from roboscape.genie.genie.config import GenieConfig            # 加载 .json 配置文件

# ---- 视频 Tokenizer（MagViT2） ----
from roboscape.genie.magvit2.models.lfqgan import VQModel       # token → pixel 的 VAE decoder
from roboscape.genie.magvit2.config import VQConfig              # MagViT2 默认配置

# ---- 数据集 ----
from roboscape.genie.data_worldarena import RawTokenDataset      # 读取 video.bin + action.bin memmap 数据集
```

**这些模块可以直接用于：**
- `__init__`：初始化 `STMaskGIT`、`VQModel`、`GenieConfig`
- `_build_dataset`：直接返回 `RawTokenDataset` 实例
- `_infer_next_chunk_frames`：调用 `STMaskGIT.maskgit_generate()`
- `_decode_tokens`：调用 `VQModel.decode()`（**建议自己写**，不要用 `roboscape.genie.visualize.decode_latents_wrapper`，因为后者走了 PIL 转换，返回 uint8 而非 float tensor，不利于构建 `current_obs`）

**不建议 import 的部分：**
- `roboscape_env.py` 的 `RoboScapeEnv`：gym 接口与 `BaseWorldEnv` 不兼容，需按本文重写
- `roboscape.genie.data.py`：旧版数据集，用 `data_worldarena.py` 中的版本

---

## 1. 架构差异与核心挑战

GENIE 与 Wan/OpenSora 的最大差异在于**推理空间不同**：

| 特性 | Wan / OpenSora | GENIE（STMaskGIT） |
|------|----------------|-------------------|
| 推理空间 | **Pixel 空间**（生成 RGB 帧，再归一化） | **Token 空间**（生成离散 token，再 decode 到 pixel） |
| 条件帧存储 | PIL Image 列表 或 VAE latent | **MagViT2 离散 token**，shape `(B, T, H_tok, W_tok)`，`uint32` |
| 每步生成帧数 | 可批量（chunk=8）| 每次生成 **1 帧**（autoregressive 逐帧生成），需循环 `chunk` 次 |
| 动作输入格式 | 连续 float，直接拼接到模型 | 连续 float（14D），通过 `nn.Linear` embed 后注入 Transformer |
| 动作归一化 | Wan 无归一化；OpenSora 用 q01/q99 | **可选**：`encode_vla_robotwin.py` 的归一化由 `--stat_file` 参数控制；当前 `encode_vla_robotwin.sh` **未传该参数**，action 为原始值，推理时也直接透传 |
| decode 步骤 | 模型直出 pixel | 需额外调用 **MagViT2 VQModel** 将 token decode 到 pixel |
| window_size | condition + chunk 帧 | `T=16`（完整 window），含 `num_prompt_frames=8` 历史帧 + 8 待预测帧 |

**核心挑战：**
1. GENIE 每步生成 1 帧，chunk_step 需要循环调用 `model.maskgit_generate()` 共 `chunk` 次
2. 条件帧必须以 MagViT2 **token 形式**存储在 `image_queue`，而非 pixel
3. 需要额外加载 **MagViT2 VQModel**（tokenizer + decoder 两用）用于 token→pixel decode
4. 动作是否归一化取决于编码时是否传入了 `--stat_file`；当前实际使用的 `encode_vla_robotwin.sh` **未传该参数**，action 为原始值，推理时直接透传即可

---

## 2. 数据表示：Token 空间 vs Pixel 空间

```
原始 HDF5 数据 (head_camera/rgb JPEG bytes)
     │ encode_vla_robotwin.py 编码
     ▼
MagViT2 tokens  shape: (total_frame, 16, 16)  dtype: uint32
action          shape: (total_frame, 14)       dtype: float32（原始值；若 encode 时传了 --stat_file 则为 p01/p99 归一化后）
     │ 存储为 memmap (.bin 文件)
     ▼
RawTokenDataset  window_size=16, stride=1
     │ 每个样本: input_ids [T*H_tok*W_tok], actions [T*14]
     ▼
GenieEnv.image_queue  shape: (B, T_prompt, H_tok, W_tok)  dtype: int64
     │ 推理：maskgit_generate() → 生成新 token
     ▼
VQModel.decode()
     │ 输出: pixel tensor [-1, 1], shape: [B, 3, H, W]
     ▼
current_obs     shape: [B, 3, 1, T, H, W]  dtype: float32  range: [-1, 1]
     │ _wrap_obs()
     ▼
Policy 输入     shape: [B, H, W, 3]         dtype: uint8    range: [0, 255]
```

---

## Step 1：注册新环境类型

**文件：** `rlinf/envs/__init__.py`

```python
class SupportedEnvType(str, Enum):
    # ... 已有类型 ...
    WANWM      = "wan_wm"
    OPENSORAWM = "opensora_wm"
    GENIEWM    = "genie_wm"   # ← 新增
```

```python
def get_env_cls(env_type, env_cfg=None, ...):
    # ... 已有分支 ...
    elif env_type == SupportedEnvType.GENIEWM:
        from rlinf.envs.world_model.world_model_genie_env import GenieEnv
        return GenieEnv
```

---

## Step 2：实现 GenieEnv 子类

**文件位置：** `rlinf/envs/world_model/world_model_genie_env.py`

### 4.1 `__init__` 构造函数

**与 Wan 实现的关键差异：**
- 需要加载 **两个模型**：世界模型（STMaskGIT）和 VQ tokenizer（MagViT2 VQModel）
- `image_queue` 存储 **token**（int64 tensor），而非 PIL Image 或 pixel float
- 需要加载动作归一化统计文件 `stat.json`

```python
from rlinf.envs.world_model.base_world_env import BaseWorldEnv
from roboscape.genie.genie.st_mask_git import STMaskGIT, GenieConfig
from roboscape.genie.magvit2.models.lfqgan import VQModel
from roboscape.genie.magvit2.config import VQConfig

class GenieEnv(BaseWorldEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes,
                 record_metrics=True, worker_info=None):
        super().__init__(cfg, num_envs, seed_offset, total_num_processes,
                         worker_info, record_metrics)

        # --- 基本参数 ---
        self.chunk = cfg.chunk            # 每次 chunk_step 生成的帧数，= action chunk 长度
        self.num_prompt_frames = cfg.num_prompt_frames  # 历史条件帧数（对应 condition_frame_length）
        self.window_size = cfg.window_size  # = num_prompt_frames + chunk（对应 num_frames）
        assert self.window_size == self.num_prompt_frames + self.chunk
        self.image_size = tuple(cfg.image_size)   # (H, W)，输出 decode 后的图像尺寸
        self.action_dim = cfg.get("action_dim", 14)
        self.latent_side_len = cfg.get("latent_side_len", 16)  # MagViT2 spatial token grid 边长
        # token 空间的 H_tok x W_tok = latent_side_len x latent_side_len = 16x16

        # --- 动作归一化统计（可选）---
        # encode_vla_robotwin.sh 当前未传 --stat_file，action.bin 存的是原始值
        # 若将来 encode 时传了 --stat_file，则这里需加载对应 stat.json 并在推理时做同样归一化
        self.action_stat = None
        if cfg.get("action_stat_file") and os.path.exists(cfg.action_stat_file):
            import json
            with open(cfg.action_stat_file) as f:
                self.action_stat = json.load(f)
            # stat.json 结构: {"state_01": [...14D...], "state_99": [...14D...]}
            self.action_p01 = np.array(self.action_stat["state_01"], dtype=np.float32)
            self.action_p99 = np.array(self.action_stat["state_99"], dtype=np.float32)

        # --- 加载世界模型 (STMaskGIT) ---
        genie_config = GenieConfig.from_pretrained(cfg.genie_config_path)
        genie_config.T = self.window_size
        genie_config.S = self.latent_side_len ** 2
        genie_config.use_mup = False
        self.world_model = STMaskGIT(genie_config).to(self.device)
        # 从训练 checkpoint 加载权重
        ckpt = torch.load(cfg.genie_ckpt_path, map_location=self.device)
        self.world_model.load_state_dict(ckpt.get("model", ckpt))
        self.world_model.eval()

        # --- 加载 MagViT2 Tokenizer（用于 encode 初始帧 + decode 生成帧）---
        self.tokenizer = VQModel(VQConfig(), ckpt_path=cfg.magvit2_ckpt_path).to(self.device)
        self.tokenizer.eval()

        # --- 加载奖励模型 ---
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # --- 内部状态 ---
        # current_obs: pixel 空间, [B, 3, 1, T, H, W], float32 [-1,1]
        self.current_obs = None
        # image_queue: 存 MagViT2 token, 每个 env 一个 deque, 每帧 shape (H_tok, W_tok), int64
        self.image_queue = [
            collections.deque(maxlen=self.num_prompt_frames)
            for _ in range(self.num_envs)
        ]
        # action_queue: 存历史动作 buffer（与 action.bin 中格式一致，当前为原始值）
        # STMaskGIT 需要 T 帧的动作同时输入（含 padding）
        self.action_queue = [
            collections.deque(maxlen=self.num_prompt_frames)
            for _ in range(self.num_envs)
        ]
        self.task_descriptions = [""] * self.num_envs
        self._is_offloaded = False

        # GRPO 支持
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()
```

**必须加载的模型总览：**

| 模型 | 类 | 配置文件/路径 |
|------|-----|-------------|
| 世界模型（STMaskGIT）| `STMaskGIT` | `cfg.genie_config_path` + `cfg.genie_ckpt_path` |
| 视频 Tokenizer（MagViT2）| `VQModel` | `cfg.magvit2_ckpt_path`（`.ckpt` 文件）|
| 奖励模型 | `RoboTwinT5CrossAttnRewardModel` 等 | `cfg.reward_model.*` |

---

### 4.2 `_build_dataset`：数据集加载

GENIE 的编码数据存储在 memmap `.bin` 文件中，可直接复用 `roboscape_env.py` 中的 `RawTokenDataset`：

```python
def _build_dataset(self, cfg):
    from roboscape.genie.data_worldarena import RawTokenDataset
    genie_config = GenieConfig.from_pretrained(cfg.genie_config_path)
    dataset = RawTokenDataset(
        data_dir=cfg.initial_image_path,   # 编码后数据目录，含 train/video.bin, action.bin 等
        window_size=self.window_size,
        config=genie_config,
        stride=1,
        split="val",                       # reset 时用 val 集初始帧
        use_action=True,
        use_text=False,
        rollout=False,
    )
    return dataset
```

**编码数据目录结构（`encode_vla_robotwin.py` 输出）：**
```
encoded_vla_robotwin/click_bell/
├── train/
│   ├── video.bin          (total_frame, 16, 16)  uint32   ← MagViT2 token
│   ├── action.bin         (total_frame, 14)       float32  ← p01/p99 归一化后
│   ├── segment_ids_total.bin (total_frame, 1)     int32
│   └── metadata.json      {num_images, h, w, s, vocab_size, ...}
└── val/
    └── ...（同上）
```

**数据集返回格式（`dataset[idx]`）：**
```python
{
    "input_ids": torch.LongTensor,  # shape: (window_size * H_tok * W_tok,) ← 展平的 token
    "actions":   torch.FloatTensor, # shape: (window_size * 14,) ← 与训练编码时一致（当前为原始值）
}
```

---

### 4.3 `reset`：环境重置

**与 Wan 的关键差异：** reset 时需要将初始帧 pixel → token（调用 MagViT2 encode），填充 `image_queue` 为 token 格式，同时 decode 回 pixel 供 `current_obs` 使用。

```python
@torch.no_grad()
def reset(self, *, seed=None, options={}, episode_indices=None):
    self.onload()
    self.elapsed_steps = 0

    # 1. 选取 episode
    if episode_indices is None:
        episode_indices = np.random.choice(len(self.dataset), size=self.num_envs, replace=False)
    elif isinstance(episode_indices, torch.Tensor):
        episode_indices = episode_indices.cpu().numpy()

    init_token_frames = []   # 每个 env 的初始 token 帧 [window_size, H_tok, W_tok]
    init_actions_list = []   # 每个 env 的历史动作 [window_size, 14]

    for episode_idx in episode_indices:
        sample = self.dataset[episode_idx]
        # token shape: [window_size, H_tok, W_tok]
        tokens = sample["input_ids"].reshape(self.window_size,
                                              self.latent_side_len,
                                              self.latent_side_len)
        actions = sample["actions"].reshape(self.window_size, self.action_dim)
        init_token_frames.append(tokens)
        init_actions_list.append(actions)

    # 2. 填充 image_queue（token 格式）和 action_queue
    for env_idx in range(self.num_envs):
        self.image_queue[env_idx].clear()
        self.action_queue[env_idx].clear()
        for t in range(self.num_prompt_frames):
            # 每帧 token: (H_tok, W_tok), int64
            self.image_queue[env_idx].append(
                init_token_frames[env_idx][t].to(torch.int64).to(self.device)
            )
            self.action_queue[env_idx].append(
                init_actions_list[env_idx][t].to(torch.float32).to(self.device)
            )

    # 3. Decode 初始帧 token → pixel，构建 current_obs
    # 取 num_prompt_frames 帧的 token 并 decode
    # tokens_batch: [B, num_prompt_frames, H_tok, W_tok]
    tokens_batch = torch.stack([
        torch.stack(list(self.image_queue[env_idx]))
        for env_idx in range(self.num_envs)
    ]).to(self.device)   # [B, num_prompt_frames, 16, 16]

    # decode: [B * num_prompt_frames, 16, 16] → [B * num_prompt_frames, 3, H, W]
    B, T_cond, Ht, Wt = tokens_batch.shape
    flat_tokens = tokens_batch.reshape(B * T_cond, Ht, Wt)
    pixel_frames = self._decode_tokens(flat_tokens)  # [B*T, 3, H, W], float32 [-1,1]
    pixel_frames = pixel_frames.reshape(B, T_cond, 3,
                                         self.image_size[0],
                                         self.image_size[1])   # [B, T, 3, H, W]
    # 转为 current_obs 标准格式: [B, 3, 1, T, H, W]
    self.current_obs = pixel_frames.permute(0, 2, None, 1, 3, 4).unsqueeze(2)
    # 注意：permute 后 shape 是 [B, 3, T, H, W]，再 unsqueeze(2) → [B, 3, 1, T, H, W]

    self._reset_metrics()
    return self._wrap_obs(), {}

def _decode_tokens(self, tokens: torch.LongTensor) -> torch.Tensor:
    """
    调用 MagViT2 VQModel 将离散 token decode 为 pixel 帧。
    输入: (N, H_tok, W_tok) int64
    输出: (N, 3, H, W) float32, range [-1, 1]
    """
    # 参考 roboscape_env.py 中 _vis() 的 decode 逻辑
    with torch.no_grad():
        if self.tokenizer.use_ema:
            with self.tokenizer.ema_scope():
                quant = self.tokenizer.quantize.get_codebook_entry(
                    rearrange(tokens, "b h w -> b (h w)"),
                    bhwc=tokens.shape + (self.tokenizer.quantize.codebook_dim,),
                ).flip(1)
                pixel = self.tokenizer.decode(quant.to(self.device))   # [-1, 1]
        else:
            # 非 EMA 模式
            indices_flat = tokens.reshape(tokens.shape[0], -1)
            quant = self.tokenizer.quantize.get_codebook_entry(
                indices_flat,
                bhwc=tokens.shape + (self.tokenizer.quantize.codebook_dim,),
            )
            pixel = self.tokenizer.decode(quant.to(self.device))
    return pixel.float()   # (N, 3, H, W), float32 [-1,1]
```

---

### 4.4 `_infer_next_chunk_frames`：世界模型推理（核心接口）

**与 Wan/OpenSora 的最大差异：** GENIE 一次只生成 **1 帧 token**（autoregressive），需循环 `chunk` 次。每次：
1. 将 `image_queue` 中的 token 组成 `prompt_THW`，未来帧填 `mask_token_id`
2. 调用 `world_model.maskgit_generate()` 预测下一帧 token
3. 新 token 加入 `image_queue`，并 decode 回 pixel 追加到 `current_obs`

```python
@torch.no_grad()
def _infer_next_chunk_frames(self, actions: np.ndarray | torch.Tensor) -> None:
    """
    调用 GENIE 世界模型，自回归地生成 chunk 帧。
    
    输入:
        actions: shape [B, chunk, 14], float32, 策略输出的原始值（未归一化）
    副作用:
        self.current_obs 追加新生成的 chunk 帧（pixel 空间, [-1,1]）
        self.image_queue 更新为最新 token
        self.action_queue 更新为最新历史动作
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions).to(self.device)
    actions = actions.float()   # [B, chunk, 14]

    # ① 动作预处理：与编码时保持一致
    # encode_vla_robotwin.sh 当前未传 --stat_file，action.bin 为原始值，直接透传
    # 若将来 encode 时启用了 stat_file，则在此做相同的 p01/p99 归一化
    if self.action_stat is not None:
        p01 = torch.from_numpy(self.action_p01).to(self.device)  # [14]
        p99 = torch.from_numpy(self.action_p99).to(self.device)  # [14]
        actions_norm = 2.0 * (actions - p01) / (p99 - p01 + 1e-8) - 1.0
        actions_norm = torch.clamp(actions_norm, -1.0, 1.0)
    else:
        actions_norm = actions   # 当前情况：原始值直接透传

    all_new_pixel_frames = []   # 收集 chunk 帧，最终拼入 current_obs

    for step in range(self.chunk):
        # ② 构建当前 window 的 token 序列
        # prompt_THW: [B, window_size, H_tok, W_tok], int64
        # 前 num_prompt_frames 帧为已知 token，后 chunk 帧填 mask_token_id
        prompt_THW = []
        for env_idx in range(self.num_envs):
            env_tokens = torch.stack(list(self.image_queue[env_idx]))  # [num_prompt_frames, Ht, Wt]
            # 后 (window_size - num_prompt_frames) 帧填 mask
            num_to_mask = self.window_size - self.num_prompt_frames
            mask_frames = torch.full(
                (num_to_mask, self.latent_side_len, self.latent_side_len),
                self.world_model.mask_token_id,
                dtype=torch.long, device=self.device
            )
            env_prompt = torch.cat([env_tokens, mask_frames], dim=0)  # [window_size, Ht, Wt]
            prompt_THW.append(env_prompt)
        prompt_THW = torch.stack(prompt_THW, dim=0)  # [B, window_size, Ht, Wt]

        # ③ 构建动作序列 [B, window_size, 14]
        # 历史 num_prompt_frames 帧动作 + 当前步新动作 + 零 padding
        prompt_action = []
        for env_idx in range(self.num_envs):
            hist_acts = torch.stack(list(self.action_queue[env_idx]))  # [num_prompt_frames, 14]
            new_act = actions_norm[env_idx, step].unsqueeze(0)          # [1, 14]
            pad_len = self.window_size - self.num_prompt_frames - 1
            pad_acts = torch.zeros(pad_len, self.action_dim, device=self.device)
            env_acts = torch.cat([hist_acts, new_act, pad_acts], dim=0)  # [window_size, 14]
            prompt_action.append(env_acts)
        prompt_action = torch.stack(prompt_action, dim=0)  # [B, window_size, 14]

        # ④ 调用 STMaskGIT.maskgit_generate()，生成下一帧 token
        # out_t = num_prompt_frames 表示生成第 num_prompt_frames 帧（即紧接历史帧的下一帧）
        samples_HW, _ = self.world_model.maskgit_generate(
            prompt_THW,               # [B, T, H_tok, W_tok], int64
            prompt_action,            # [B, T, 14], float32
            out_t=self.num_prompt_frames,   # 生成第几帧（0-indexed）
            maskgit_steps=2,          # MaskGIT 迭代步数，越多质量越好但越慢
            temperature=0.0,          # 0=贪心采样
        )
        # samples_HW: [B, H_tok, W_tok], int64

        # ⑤ Decode token → pixel
        new_frame_pixel = self._decode_tokens(samples_HW)  # [B, 3, H, W], float32 [-1,1]
        all_new_pixel_frames.append(new_frame_pixel)

        # ⑥ 更新 image_queue 和 action_queue（滑窗）
        for env_idx in range(self.num_envs):
            self.image_queue[env_idx].append(samples_HW[env_idx])          # deque 自动弹出最老帧
            self.action_queue[env_idx].append(actions_norm[env_idx, step])  # 更新动作历史

    # ⑦ 将新生成的 chunk 帧追加到 current_obs
    # all_new_pixel_frames: list of chunk tensors, each [B, 3, H, W]
    new_frames_tensor = torch.stack(all_new_pixel_frames, dim=2)  # [B, 3, chunk, H, W]
    new_frames_tensor = new_frames_tensor.unsqueeze(2)             # [B, 3, 1, chunk, H, W]

    # 注意：此处 unsqueeze 位置需与 current_obs 的 [B,3,1,T,H,W] 格式对齐
    # 正确做法：
    new_frames_5d = torch.stack(all_new_pixel_frames, dim=1)  # [B, chunk, 3, H, W]
    new_frames_5d = new_frames_5d.permute(0, 2, 1, 3, 4)      # [B, 3, chunk, H, W]
    new_frames_6d = new_frames_5d.unsqueeze(2)                 # [B, 3, 1, chunk, H, W]

    self.current_obs = torch.cat([self.current_obs, new_frames_6d], dim=3)

    # 保持滑窗，避免内存无限增长
    max_frames = self.num_prompt_frames + self.chunk
    if self.current_obs.shape[3] > max_frames:
        self.current_obs = self.current_obs[:, :, :, -max_frames:, :, :]
```

**`maskgit_generate` 函数签名（来自 `genie/st_mask_git.py`）：**

```python
@torch.no_grad()
def maskgit_generate(
    self,
    prompt_THW: torch.LongTensor,   # (B, T, H, W) token ids，未来帧用 mask_token_id 填充
    prompt_action,                   # (B, T, action_dim) 归一化后的动作
    out_t: int,                      # 要生成的帧索引（>= 1）
    maskgit_steps: int = 1,          # MaskGIT 迭代步数
    temperature: float = 0.0,        # 采样温度（0=贪心）
    unmask_mode: str = "random",
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    # 返回: (samples_HW [B, H_tok, W_tok], factored_logits)
```

**注意事项：**
- `out_t` 的值必须等于 `num_prompt_frames`（即历史帧数量），因为每次都只生成紧接历史的下一帧
- 如果生成多帧（chunk > 1），需要用滑窗方式：生成完第 t 帧后，将其加入 `image_queue`，下一步再生成第 t+1 帧
- `train_action.sh` 中 `--window_size 16 --stride 1`，因此 `window_size=16`，`num_prompt_frames` 通常取 `8`（窗口前半）

---

### 4.5 `_infer_next_chunk_rewards`：奖励计算

与 Wan 完全相同，从 `current_obs` 取最新 chunk 帧，normalize 到 `[0,1]` 后送入奖励模型：

```python
def _infer_next_chunk_rewards(self) -> torch.Tensor:
    B, c, v, t, h, w = self.current_obs.shape
    # 取最新 chunk 帧
    frames = self.current_obs[:, :, 0, -self.chunk:, :, :]  # [B, 3, chunk, H, W]
    # [-1,1] → [0,1]
    frames_01 = (frames + 1.0) / 2.0
    # 展平: [B*chunk, 3, H, W]
    frames_flat = frames_01.permute(0, 2, 1, 3, 4).reshape(-1, 3, h, w).float()

    instructions = []
    for env_idx in range(self.num_envs):
        instructions.extend([self.task_descriptions[env_idx]] * self.chunk)

    with torch.no_grad():
        rewards = self.reward_model.compute_reward(
            frames_flat,
            task_descriptions=instructions,
        )  # [B*chunk]

    return rewards.reshape(B, self.chunk)
```

---

### 4.6 `_wrap_obs`：观测打包（策略输入接口）

**与 Wan 完全相同**（从 `current_obs` 取最后一帧，decode 结果直接在 `current_obs` 中，无需额外处理）：

```python
def _wrap_obs(self) -> dict:
    b, c, v, t, h, w = self.current_obs.shape
    last_frame = self.current_obs[:, :, 0, -1, :, :]   # [B, 3, H, W]
    full_image = last_frame.permute(0, 2, 3, 1)          # [B, H, W, 3]
    full_image = (full_image + 1.0) / 2.0 * 255.0
    full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)

    states = torch.zeros((b, self.action_dim), device=self.device, dtype=torch.float32)

    return {
        "main_images":       full_image,             # [B, H, W, 3], uint8, [0,255]
        "wrist_images":      None,
        "states":            states,                 # [B, 14], float32, 全零占位
        "task_descriptions": self.task_descriptions, # list[str]
    }
```

---

### 4.7 `chunk_step`：驱动单步推进

与 Wan 实现结构完全相同，只是内部调用的是 GENIE 版的 `_infer_next_chunk_frames`（循环生成 chunk 帧）。需注意：由于 GENIE 每帧推理都要调用一次 Transformer forward，**计算量是 Wan/OpenSora 的 chunk 倍**，务必在 chunk 较小（如 chunk=4）时先做 smoke test。

```python
@torch.no_grad()
def chunk_step(self, policy_output_action):
    self.onload()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        self._infer_next_chunk_frames(policy_output_action)

    self.elapsed_steps += self.chunk
    extracted_obs = self._wrap_obs()
    chunk_rewards = self._infer_next_chunk_rewards()
    chunk_rewards_tensors = self._calc_step_reward(chunk_rewards)

    estimated_success = self._estimate_success_from_rewards(chunk_rewards)
    chunk_terminations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool, device=self.device)
    chunk_terminations[:, -1] = estimated_success
    chunk_truncations = torch.zeros_like(chunk_terminations)
    truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps).to(self.device)
    if truncations.any():
        chunk_truncations[:, -1] = truncations

    past_terminations = chunk_terminations.any(dim=1)
    past_truncations = chunk_truncations.any(dim=1)

    infos = self._record_metrics(chunk_rewards_tensors.sum(dim=1), past_terminations, {})
    return ([extracted_obs], chunk_rewards_tensors, chunk_terminations, chunk_truncations, [infos])
```

---

### 4.8 `offload` / `onload`：显存管理

GENIE 需要管理 **三个模型**（世界模型、tokenizer、奖励模型）的显存：

```python
def offload(self):
    if self._is_offloaded:
        return
    self.world_model = self.world_model.to("cpu")
    self.tokenizer = self.tokenizer.to("cpu")
    self.reward_model = self.reward_model.to("cpu")
    if self.current_obs is not None:
        self.current_obs = self.current_obs.to("cpu")
    torch.cuda.empty_cache()
    self._is_offloaded = True

def onload(self):
    if not self._is_offloaded:
        return
    self.world_model = self.world_model.to(self.device)
    self.tokenizer = self.tokenizer.to(self.device)
    self.reward_model = self.reward_model.to(self.device)
    if self.current_obs is not None:
        self.current_obs = self.current_obs.to(self.device)
    self._is_offloaded = False
```

---

## Step 3：动作适配（prepare_actions）

**文件：** `rlinf/envs/action_utils.py`

GENIE 世界模型的动作（RobotWin 14D）直接透传，不需要额外变换（归一化在 `_infer_next_chunk_frames` 内部做）：

```python
elif env_type == SupportedEnvType.GENIEWM:
    if wm_env_type == "robotwin":
        # 直接透传，GENIE env 内部做 p01/p99 归一化
        chunk_actions = raw_chunk_actions
    else:
        raise NotImplementedError(f"Unsupported wm_env_type: {wm_env_type}")
```

---

## Step 4：编写环境配置 YAML

**文件位置：** `examples/embodiment/config/env/genie_robotwin_click_bell.yaml`

```yaml
env_type: genie_wm
wm_env_type: robotwin
task_suite_name: click_bell

total_num_envs: null      # 训练配置中覆盖
auto_reset: False
max_episode_steps: 200
max_steps_per_rollout_epoch: 200

use_rel_reward: True
reward_coef: 5.0

group_size: 1
use_fixed_reset_state_ids: True

action_dim: 14
chunk: 4                        # ← GENIE 每帧推理成本高，建议先从 chunk=4 开始
num_prompt_frames: 8            # 对应 condition_frame_length（condition_frame_length = window_size - chunk）
window_size: 12                 # = num_prompt_frames + chunk
image_size: [256, 256]

# 编码数据目录（encode_vla_robotwin.py 输出，含 train/val 子目录）
initial_image_path: /ML-vePFS/protected/tangyinzhou/RLinf/roboscape/encoded_vla_robotwin/click_bell

# GENIE 世界模型配置与权重（train_action.sh 输出目录）
genie_config_path: /ML-vePFS/protected/tangyinzhou/RLinf/roboscape/genie/genie/configs/magvit_n32_h8_d512.json
genie_ckpt_path: /manifold-obs/tangyinzhou/worldarena/worldarena_env_click_bell/checkpoint-XXXXX/model.bin

# MagViT2 Tokenizer 权重
magvit2_ckpt_path: /ML-vePFS/protected/tangyinzhou/RLinf/roboscape/genie/magvit2.ckpt

# 动作归一化统计文件（仅当 encode 时传了 --stat_file 才需填写；当前实际使用的
# encode_vla_robotwin.sh 未传该参数，action.bin 为原始值，此项可省略）
# action_stat_file: /path/to/stat.json

# token 空间分辨率（与 MagViT2 encoder 输出一致）
latent_side_len: 16

# 奖励模型
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World/model_ckpt/robotwin_click_bell/model.pth
  t5_model_name: /ML-vePFS/protected/tangyinzhou/RLinf/pretrained_models/t5-base

video_cfg:
  save_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: False
success_reward_threshold: 0.9
```

---

## Step 5：编写训练配置 YAML

**文件位置：** `examples/embodiment/config/genie_robotwin_click_bell_grpo_openpi_pi05.yaml`

```yaml
defaults:
  - _self_
  - env/train: genie_robotwin_click_bell
  - env/eval: genie_robotwin_click_bell

algorithm:
  adv_type: grpo
  loss_type: grpo_actor
  group_size: 4
  reward_coef: 5.0

env:
  train:
    total_num_envs: 4             # group_size 的整数倍 且 能被 env_world_size 整除
    group_size: ${algorithm.group_size}
    chunk: 4                      # ← 必须与 actor.num_action_chunks 一致
    action_dim: 14
    reward_coef: ${algorithm.reward_coef}
  eval:
    total_num_envs: 4
    chunk: 4
    action_dim: 14

rollout:
  model:
    model_type: openpi
    model_path: /ML-vePFS/protected/tangyinzhou/RLinf/pi05_base

actor:
  model:
    model_type: openpi
    num_action_chunks: 4          # ← 必须与 env.chunk 一致
    action_dim: 14
    openpi:
      action_chunk: 4             # ← 必须与 env.chunk 一致
      action_env_dim: 14

cluster:
  num_nodes: 1
  component_placement:
    # ... 根据 GPU 数量配置
```

---

## 关键约束与一致性检查清单

| 项目 | GENIE 特有约束 |
|------|--------------|
| **chunk 一致性** | `env.chunk` == `actor.num_action_chunks` == `actor.openpi.action_chunk` |
| **window_size 正确** | `window_size` == `num_prompt_frames + chunk`（不是 condition + chunk，而是 prompt + chunk）|
| **token 格式** | `image_queue` 中每帧为 `(H_tok, W_tok)` int64 tensor，**不是** pixel，不是 PIL Image |
| **动作归一化** | 与编码时保持一致：当前 `encode_vla_robotwin.sh` 未传 `--stat_file`，action 为原始值，推理时直接透传；若 encode 时传了 `--stat_file`，推理时需加载同一 `stat.json` 做相同归一化 |
| **maskgit_steps** | 建议 `maskgit_steps=2`（与 `generate.sh` 保持一致），1 步质量差，过多步推理慢 |
| **逐帧生成** | GENIE 每个 chunk 需循环 `chunk` 次调用 `maskgit_generate()`，计算量 = Wan × chunk 倍 |
| **decode 函数** | 必须使用 `tokenizer.ema_scope()` 上下文（若 `tokenizer.use_ema == True`），否则 decode 结果偏差大 |
| **GRPO 整除** | `total_num_envs` % `group_size` == 0 且 % `env_world_size` == 0 |
| **图像归一化链路** | token decode → pixel [-1,1] → `current_obs` [-1,1] → `_wrap_obs` uint8 [0,255] → Policy |

---

## 与 Wan / OpenSora 实现的主要差异对比

| 特性 | Wan（DiffSynth） | OpenSora（STDiT3） | **GENIE（STMaskGIT）** |
|------|-----------------|-------------------|-----------------------|
| **枚举值** | `wan_wm` | `opensora_wm` | `genie_wm` |
| **推理接口** | `WanVideoPipeline(...)` | `scheduler.sample(...)` | `model.maskgit_generate(out_t=...)` 循环 chunk 次 |
| **条件帧格式** | PIL Image 列表（pixel space）| VAE latent（latent space）| **MagViT2 token**（token space，uint32/int64）|
| **每步生成帧数** | chunk 帧一次生成 | chunk 帧一次生成 | **每次生成 1 帧**，循环 chunk 次 |
| **额外 decode 步骤** | 无（直出 pixel）| VAE decode | **必须调用 MagViT2 VQModel.decode()** |
| **动作归一化** | 无（原始值直传）| q01/q99 分位数 | **无**（当前 encode 未传 --stat_file，原始值直传；若将来启用需同步修改）|
| **模型数量** | 2（DiffSynth Pipeline = dit + vae）| 3（model + vae + scheduler）| **3**（STMaskGIT + MagViT2 + reward）|
| **推理速度（相对）** | 基准 | 类似 Wan | **最慢**（chunk 次 forward，每次都有完整 Transformer）|
| **建议 chunk** | 8 | 8 | **4**（先 smoke test，视速度决定是否增大）|
| **条件帧数** | 5 | 4 | **8**（`num_prompt_frames`，window_size=16 时推荐值）|
| **环境配置文件** | `env/wan_robotwin_click_bell.yaml` | `env/opensora_robotwin_click_bell.yaml` | `env/genie_robotwin_click_bell.yaml` |
| **训练配置文件** | `wan_robotwin_click_bell_grpo_openpi_pi05.yaml` | `opensora_robotwin_click_bell_grpo_pi05.yaml` | `genie_robotwin_click_bell_grpo_openpi_pi05.yaml` |
