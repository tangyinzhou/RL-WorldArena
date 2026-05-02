# Ctrl-World 世界模型集成 RL 环境指南

本文档是 `docs/world_model_rl_integration.md` 通用指南的 **Ctrl-World 专版**。
参照通用指南的五步结构，逐节说明 Ctrl-World 的具体实现方式及与 Wan / OpenSora 的差异。

---

## Ctrl-World 架构速览

| 属性 | Ctrl-World | Wan | OpenSora |
|------|-----------|-----|----------|
| 基础架构 | SVD（Stable Video Diffusion）| Wan2.2-TI2V-5B | STDiT3 |
| image_queue 存储 | **latent 空间**（与 OpenSora 相同）| pixel 空间（PIL Image）| latent 空间 |
| 多视角 latent | 3 路视图沿 H 方向拼接 `(4, 72, 40)` | N/A | N/A |
| history 帧数 | 6（`num_history`）| 5（条件帧）| 4（条件帧）|
| 预测帧数 (chunk) | **5**（`num_frames`）| 8 | 8 |
| 动作归一化 | p01/p99 bound norm → [-1,1] | 不归一化（原始值）| q01/q99 → [-1,1] |
| 历史采样策略 | `history_idx = [0,0,-8,-6,-4,-2]`（非均匀）| 最近 4 帧 | 最近 N 帧 |
| 图像分辨率 | 原始 (192, 320)，decode 后 resize 到 (256, 256) | 直接 (256, 256) | (256, 256) |

> **关键差异**：Ctrl-World 的 image_queue 保存的是形状 `(1, 4, 72, 40)` 的 latent 张量，
> 其中 72 = 24×3（3 个摄像头视角沿 H 方向拼接，对 RobotWin head-cam 任务 3 路相同）。
> 模型本体是 `CrtlWorld`（注意拼写）包含 SVD-UNet + SVD-VAE + CLIP 文本编码器 + Action_encoder2。

---

## 核心数据流

```
数据集初始帧 (uint8, [0,255], (H=192, W=320))
     │ reset() 用 SVD-VAE 编码
     ▼
image_queue 中的 latent   (float32, scaled, shape: [1, 4, 72, 40])
     │                    72 = 24×3（3路视图拼接）
     │ _infer_next_chunk_frames()
     │   ├── 归一化 action（p01/p99 → [-1,1]）
     │   ├── action_encoder 编码动作 → text_token [B, 11, 1024]
     │   ├── 从 image_queue 按 history_idx 采样历史 [B, 6, 4, 72, 40]
     │   └── CtrlWorldDiffusionPipeline.__call__(image, text, history, num_frames=5)
     ▼
预测 latent  (shape: [B, 5, 4, 72, 40])
     │ einops.rearrange → 分离3路视图 → 取第0路 (view_0)
     │ VAE.decode(view_0_latent / scaling_factor)
     │ → pixel (B, 5, 3, 192, 320) float32 [-1,1]
     │ resize → (B, 5, 3, 256, 256)
     ▼
current_obs 追加新帧      (float32 [-1,1], shape: [B, 3, 1, T, 256, 256])
     │
     ├──→ 奖励模型输入   float32 [0,1], shape [B*chunk, 3, 256, 256]
     │
     └──→ _wrap_obs()    uint8 [0,255], shape [B, 256, 256, 3]
               ▼
          Policy 输入 obs["main_images"]
```

---

## 直接 import 的核心组件

Ctrl-World 仓库中的以下组件**无需复制任何代码**，只需通过 `sys.path.insert` 引入即可：

| 组件 | 来源文件 | 用途 |
|------|---------|------|
| **`CrtlWorld`** | `Ctrl-World/models/ctrl_world.py` | 主模型类：封装 VAE + UNet + CLIP + Action_encoder2；`load_state_dict(ckpt)` 加载权重 |
| **`CtrlWorldDiffusionPipeline`** | `Ctrl-World/models/pipeline_ctrl_world.py` | 推理管道：`__call__(image, text, history, ...)` 完成 diffusion（设 `output_type='latent'` 仅返回 latent） |
| **`Action_encoder2`** | `Ctrl-World/models/ctrl_world.py` | 动作编码器：`forward(action)` → `[B, T, 1024]`；如需在环境侧预处理动作可直接用 |
| `UNetSpatioTemporalConditionModel` | `Ctrl-World/models/unet_spatio_temporal_condition.py` | 已嵌入 `CrtlWorld.pipeline.unet`，无需单独引用 |
| `Dataset_mix` | `Ctrl-World/dataset/dataset_droid_exp33.py` | 训练数据集（参考其加载 annotation + latent 的逻辑，环境侧实现可借鉴） |

### 标准 import 写法

```python
import sys
# 将 Ctrl-World 仓库根目录加入 Python 路径
sys.path.insert(0, cfg.ctrlworld_repo_path)

from models.ctrl_world import CrtlWorld
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

# 构造 CrtlWorld 所需的 args（它接受普通 Python object，不必须是 dataclass）
import types
args = types.SimpleNamespace(
    svd_model_path=cfg.svd_model_path,
    clip_model_path=cfg.clip_model_path,
    action_dim=cfg.action_dim,
    num_history=6,
    num_frames=5,          # Ctrl-World 的 num_frames = 预测帧数（=chunk）
    text_cond=True,
    frame_level_cond=True,
    his_cond_zero=False,
    motion_bucket_id=127,
    fps=7,
)

# 加载模型
model = CrtlWorld(args)
state_dict = torch.load(cfg.ctrlworld_ckpt_path, map_location="cpu")
model.load_state_dict(state_dict)
model.to(device).eval()

# 通过 pipeline 调用推理（输出 latent）
pipeline = model.pipeline    # CtrlWorldDiffusionPipeline 实例
vae = model.vae              # SVD VAE
action_encoder = model.action_encoder   # Action_encoder2
```

### 调用 `CtrlWorldDiffusionPipeline` 的最小示例

```python
import torch, einops

# image_cond : [B, 4, 72, 40] 当 pre_encode=True 时
# text_token : [B, 11, 1024] 动作经 action_encoder 编码
# history   : [B, 6, 4, 72, 40]

_, pred_latents = CtrlWorldDiffusionPipeline.__call__(
    pipeline,
    image=image_cond,          # 当前 latent 条件
    text=text_token,            # 动作 embedding
    height=height * 3,        # 576（3路拼接高度）
    width=width,               # 320
    num_frames=5,             # 预测 5 帧
    history=history,           # 历史 latent
    num_inference_steps=50,
    decode_chunk_size=7,
    max_guidance_scale=1.0,
    fps=7,
    motion_bucket_id=127,
    mask=None,
    output_type="latent",      # 仅返回 latent，不解码
    return_dict=False,
    frame_level_cond=True,
)
# pred_latents: [B, 5, 4, 72, 40]
```

### 不能直接 import、需要重新实现的部分

| 部分 | 原因 |
|------|------|
| `normalize_bound()`（`rollout_robotwin_click_bell.py` L111-113）| 逻辑极简：`2*(x-p01)/(p99-p01)-1`，可自行实现或直接写在环境类中 |
| `Dataset_mix` | 训练数据集含复杂随机采样逻辑；RL 环境只需顺序/指定索引加载，逻辑不同 |
| `extract_latent_robotwin.py` | 一次性数据预处理脚本，非运行时组件 |
| `config.py` / `config_eval.py` | 训练配置；RL 集成使用 Hydra YAML |

---

## Step 1：注册新环境类型

**文件：** `rlinf/envs/__init__.py`

### 1.1 枚举添加

```python
class SupportedEnvType(Enum):
    # ... 已有类型 ...
    CTRLWORLD_WM = "ctrlworld_wm"   # ← 新增
```

### 1.2 工厂函数添加

```python
elif env_type == SupportedEnvType.CTRLWORLD_WM:
    from rlinf.envs.world_model.world_model_ctrlworld_env import CtrlWorldEnv
    return CtrlWorldEnv
```

---

## Step 2：实现 CtrlWorldEnv 子类

**文件：** `rlinf/envs/world_model/world_model_ctrlworld_env.py`

### 2.1 `__init__` 构造函数

```python
import sys, json, os, collections
from typing import Optional, Union
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from rlinf.envs.world_model.base_world_env import BaseWorldEnv


class CtrlWorldEnv(BaseWorldEnv):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes,
                 record_metrics=True, worker_info=None):
        # ---------- 先设好超参数（_build_dataset 会在 super().__init__ 中被调用）----------
        self.chunk: int = cfg.chunk          # = 5 (num_frames, 预测帧数)
        self.num_history: int = cfg.num_history  # = 6 (历史帧数)
        self.image_size: tuple = tuple(cfg.image_size)   # = (256, 256)
        self.action_dim: int = cfg.get("action_dim", 14)
        self.wm_env_type: str = cfg.get("wm_env_type", "robotwin")
        # history_idx 用于从 image_queue 中采样历史帧
        # 默认 [0, 0, -8, -6, -4, -2]（与 rollout 脚本一致）
        self.history_idx = cfg.get("history_idx", [0, 0, -8, -6, -4, -2])

        super().__init__(cfg, num_envs, seed_offset, total_num_processes,
                         worker_info, record_metrics)

        # ---------- 重置状态管理（镜像 WanEnv）----------
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

        # ---------- 加载 Ctrl-World 模型 ----------
        # 需要把 Ctrl-World 目录加入 sys.path，或将 CrtlWorld 封装为可 import 的包
        sys.path.insert(0, cfg.ctrlworld_repo_path)   # e.g. /path/to/Ctrl-World
        from models.ctrl_world import CrtlWorld
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline

        # 构造 args 对象（CrtlWorld 期望 dataclass 形式）
        import types
        args = types.SimpleNamespace(
            svd_model_path=cfg.svd_model_path,
            clip_model_path=cfg.clip_model_path,
            action_dim=self.action_dim,
            num_history=self.num_history,
            num_frames=self.chunk,           # Ctrl-World 的 num_frames = 未来预测帧数
            text_cond=cfg.get("text_cond", True),
            frame_level_cond=cfg.get("frame_level_cond", True),
            his_cond_zero=cfg.get("his_cond_zero", False),
            motion_bucket_id=cfg.get("motion_bucket_id", 127),
            fps=cfg.get("fps", 7),
        )
        self.ctrlworld_model = CrtlWorld(args)
        state_dict = torch.load(cfg.ctrlworld_ckpt_path, map_location="cpu")
        self.ctrlworld_model.load_state_dict(state_dict, strict=True)
        self.ctrlworld_model.to(self.device).eval()
        self.pipeline_cls = CtrlWorldDiffusionPipeline

        # 推理参数
        self.num_inference_steps = cfg.get("num_inference_steps", 50)
        self.guidance_scale = cfg.get("guidance_scale", 1.0)
        self.decode_chunk_size = cfg.get("decode_chunk_size", 7)
        self.width = cfg.get("width", 320)       # SVD 原始宽度
        self.height = cfg.get("height", 192)      # SVD 原始高度

        # ---------- 加载动作归一化统计 ----------
        # stat.json 格式：{"state_01": [...], "state_99": [...]}
        with open(cfg.data_stat_path, "r") as f:
            stat = json.load(f)
        self.action_p01 = np.array(stat["state_01"], dtype=np.float32)[None, :]  # (1, 14)
        self.action_p99 = np.array(stat["state_99"], dtype=np.float32)[None, :]  # (1, 14)

        # ---------- 加载奖励模型 ----------
        self.reward_model = self._load_reward_model().eval().to(self.device)

        # ---------- 运行时状态 ----------
        # current_obs: float32 [-1,1], shape [B, 3, 1, T, H, W]（H,W=256,256）
        self.current_obs: Optional[torch.Tensor] = None
        self.task_descriptions: list = [""] * self.num_envs

        # image_queue[env_idx]: deque of (1, 4, 72, 40) latent 张量
        # maxlen 设为足够大以支持 history_idx 中最大的负偏移（如 -8）
        self.image_queue: list = [
            collections.deque(maxlen=max(16, abs(min(self.history_idx)) + 2))
            for _ in range(self.num_envs)
        ]

        # condition_action[env_idx]: deque of (1, action_dim) float32，历史动作
        self.action_queue: list = [
            collections.deque(maxlen=max(16, abs(min(self.history_idx)) + 2))
            for _ in range(self.num_envs)
        ]

        self._is_offloaded = False
```

---

### 2.2 `_build_dataset`

Ctrl-World 的数据集格式（由 `extract_latent_robotwin.py` 生成）：

```
{dataset_root}/
  annotation/{train,val}/{traj_id}.json       ← 包含 videos, latent_videos, states, texts
  videos/{train,val}/{traj_id}/0.mp4
  latent_videos/{train,val}/{traj_id}/0.pt    ← SVD-VAE 编码后的 latent (T, 4, 24, 40)
```

annotation JSON 格式：
```json
{
  "texts": ["click the bell"],
  "video_length": 50,
  "videos": [{"video_path": "videos/val/99/0.mp4"}],
  "latent_videos": [{"latent_video_path": "latent_videos/val/99/0.pt"}],
  "states": [[...], [...]]   // 5fps 下采样后的 14-dim states
}
```

```python
def _build_dataset(self, cfg):
    """返回 Ctrl-World 格式的数据集包装器。"""
    return CtrlWorldDataset(
        dataset_root=cfg.initial_image_path,
        data_type=cfg.get("data_type", "val"),
    )
```

```python
class CtrlWorldDataset:
    """读取 extract_latent_robotwin.py 输出格式的数据集。"""
    def __init__(self, dataset_root: str, data_type: str = "val"):
        import glob
        self.dataset_root = dataset_root
        self.data_type = data_type
        ann_dir = os.path.join(dataset_root, "annotation", data_type)
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.json")))

    def __len__(self):
        return len(self.ann_files)

    def __getitem__(self, idx):
        with open(self.ann_files[idx]) as f:
            ann = json.load(f)
        return ann  # 直接返回 annotation dict
```

---

### 2.3 `reset`

```python
@torch.no_grad()
def reset(self, *, seed=None, options=None, episode_indices=None):
    self.onload()
    self.elapsed_steps = 0

    if self.is_start:
        if self.use_fixed_reset_state_ids:
            episode_indices = self.reset_state_ids
        self._is_start = False

    if episode_indices is None:
        episode_indices = np.random.choice(len(self.dataset), size=self.num_envs, replace=False)
    elif isinstance(episode_indices, torch.Tensor):
        episode_indices = episode_indices.cpu().numpy()

    vae = self.ctrlworld_model.vae
    task_descs = []

    for env_idx, ep_idx in enumerate(episode_indices):
        ann = self.dataset[int(ep_idx)]

        # 1. 获取任务描述
        task_descs.append(ann["texts"][0] if ann["texts"] else "")

        # 2. 加载初始帧的 latent
        #    latent_videos 中存的是单路 latent (T, 4, 24, 40)
        latent_path = os.path.join(
            self.dataset.dataset_root, ann["latent_videos"][0]["latent_video_path"]
        )
        latent_single = torch.load(latent_path, map_location="cpu")  # (T, 4, 24, 40)
        first_latent = latent_single[0:1]  # (1, 4, 24, 40)

        # 3. 将单路 latent 复制为 3 路（H 方向拼接：72 = 24×3）
        #    RobotWin 只有 head_camera，3 路完全相同
        first_latent_3view = torch.cat([first_latent] * 3, dim=2)  # (1, 4, 72, 40)
        first_latent_3view = first_latent_3view.to(self.device)

        # 4. 填充 image_queue（用重复的初始帧填满历史缓冲区）
        self.image_queue[env_idx].clear()
        queue_maxlen = self.image_queue[env_idx].maxlen
        for _ in range(queue_maxlen):
            self.image_queue[env_idx].append(first_latent_3view.clone())

        # 5. 填充 action_queue（用初始动作填满历史缓冲区）
        if "states" in ann and len(ann["states"]) > 0:
            init_action = torch.tensor(ann["states"][0], dtype=torch.float32).unsqueeze(0)
        else:
            init_action = torch.zeros(1, self.action_dim)
        self.action_queue[env_idx].clear()
        for _ in range(self.action_queue[env_idx].maxlen):
            self.action_queue[env_idx].append(init_action.clone().to(self.device))

    self.task_descriptions = task_descs

    # 6. 解码初始 latent → pixel，构建 current_obs
    #    以第一个 env 的初始帧为例，批量处理所有 env
    all_first_latents = torch.cat([
        list(self.image_queue[i])[-1] for i in range(self.num_envs)
    ], dim=0)  # [B, 4, 72, 40]

    # 取第 0 路视图（latent 中 H[0:24]）
    view0_latent = all_first_latents[:, :, 0:24, :]  # [B, 4, 24, 40]
    pixel_frames = self._decode_latent(view0_latent)   # [B, 3, 192, 320] float32 [-1,1]
    pixel_frames = F.interpolate(pixel_frames, size=self.image_size, mode="bilinear", align_corners=False)
    # [B, 3, H, W] → [B, 3, 1, 1, H, W]
    self.current_obs = pixel_frames.unsqueeze(2).unsqueeze(3)

    self._reset_metrics()
    return self._wrap_obs(), {}
```

---

### 2.4 `_infer_next_chunk_frames`（核心接口）

```python
@torch.no_grad()
def _infer_next_chunk_frames(self, actions: Union[np.ndarray, torch.Tensor]) -> None:
    """
    调用 CtrlWorldDiffusionPipeline 预测下一段视频帧。
    
    输入：
        actions: [B, chunk=5, action_dim=14]，float32，策略输出原始值
    副作用：
        更新 self.current_obs，追加新生成的 chunk 帧
        更新 self.image_queue 和 self.action_queue
    """
    if isinstance(actions, np.ndarray):
        actions = torch.from_numpy(actions)
    actions = actions.float()  # [B, 5, 14]

    B = self.num_envs

    # 1. 归一化动作：p01/p99 bound normalization → [-1, 1]
    p01 = torch.from_numpy(self.action_p01).to(self.device)  # (1, 14)
    p99 = torch.from_numpy(self.action_p99).to(self.device)  # (1, 14)
    actions_np = actions.cpu().numpy()  # [B, 5, 14]
    # 对 action 中每帧做归一化
    actions_norm = 2.0 * (actions_np - self.action_p01) / (
        self.action_p99 - self.action_p01 + 1e-8) - 1.0
    actions_norm = np.clip(actions_norm, -1.0, 1.0)
    actions_tensor = torch.from_numpy(actions_norm).to(self.device).to(
        next(self.ctrlworld_model.parameters()).dtype
    )  # [B, 5, 14], bfloat16

    # 2. 构建每个 env 的 action_cond（历史动作 + 新动作）
    #    action_cond shape: [B, num_history + chunk, 14] = [B, 11, 14]
    all_action_cond = []
    for env_idx in range(B):
        # 从 action_queue 按 history_idx 采样历史动作
        q = list(self.action_queue[env_idx])  # list of (1, 14) tensors
        his_actions = []
        for idx in self.history_idx:  # [0, 0, -8, -6, -4, -2]
            his_actions.append(q[idx])  # (1, 14)
        his_tensor = torch.cat(his_actions, dim=0)  # (6, 14)
        action_cond = torch.cat([his_tensor, actions_tensor[env_idx]], dim=0)  # (11, 14)
        all_action_cond.append(action_cond)
    action_cond_batch = torch.stack(all_action_cond, dim=0)  # [B, 11, 14]

    # 3. 将动作编码为 text_token
    #    Action_encoder2 输出 [B, 11, 1024] (frame_level_cond=True)
    pipeline = self.ctrlworld_model.pipeline
    text_token = self.ctrlworld_model.action_encoder(
        action_cond_batch,
        texts=self.task_descriptions,
        text_tokinizer=self.ctrlworld_model.tokenizer,
        text_encoder=self.ctrlworld_model.text_encoder,
        frame_level_cond=True,
    )  # [B, 11, 1024]

    # 4. 构建 image_cond（当前 latent）和 history（历史 latent）
    #    image_cond: [B, 4, 72, 40]
    #    history:    [B, 6, 4, 72, 40]
    batch_image_cond = []
    batch_history = []
    for env_idx in range(B):
        q = list(self.image_queue[env_idx])  # list of (1, 4, 72, 40) tensors
        # 当前帧
        batch_image_cond.append(q[-1])  # (1, 4, 72, 40)
        # 历史帧（按 history_idx 采样）
        his_frames = [q[idx] for idx in self.history_idx]  # 6×(1, 4, 72, 40)
        history_tensor = torch.cat(his_frames, dim=0)  # (6, 4, 72, 40)
        batch_history.append(history_tensor)

    image_cond_batch = torch.cat(batch_image_cond, dim=0).to(
        self.device, text_token.dtype
    )  # [B, 4, 72, 40]
    history_batch = torch.stack(batch_history, dim=0).to(
        self.device, text_token.dtype
    )  # [B, 6, 4, 72, 40]

    # 5. 调用 CtrlWorldDiffusionPipeline
    #    注意：此处 num_frames=self.chunk（=5，仅预测未来帧）
    #    pipeline 内部会将 history 拼接到 UNet 输入
    _, pred_latents = self.pipeline_cls.__call__(
        pipeline,
        image=image_cond_batch,           # [B, 4, 72, 40]，当前 latent
        text=text_token,                  # [B, 11, 1024]，动作 embedding
        height=int(self.height * 3),      # 72 对应 3 路视图拼接高度
        width=self.width,                 # 320
        num_frames=self.chunk,            # = 5
        history=history_batch,            # [B, 6, 4, 72, 40]
        num_inference_steps=self.num_inference_steps,
        decode_chunk_size=self.decode_chunk_size,
        max_guidance_scale=self.guidance_scale,
        fps=7,
        motion_bucket_id=127,
        mask=None,
        output_type="latent",
        return_dict=False,
        frame_level_cond=True,
    )
    # pred_latents: [B, chunk, 4, 72, 40]

    # 6. 分离 3 路视图，取第 0 路解码到像素
    import einops
    # [B, chunk, 4, 72, 40] → [B*3, chunk, 4, 24, 40]
    pred_views = einops.rearrange(
        pred_latents, 'b f c (m h) w -> (b m) f c h w', m=3
    )
    view0 = pred_views[::3]  # 取每 B 组的第 0 路：[B, chunk, 4, 24, 40]

    # 7. VAE 解码 → 像素 [-1, 1]
    B_v, T_pred, C_lat, Hl, Wl = view0.shape
    flat_latents = view0.reshape(B_v * T_pred, C_lat, Hl, Wl)  # [B*chunk, 4, 24, 40]
    pixel_frames = self._decode_latent(flat_latents)  # [B*chunk, 3, 192, 320]
    pixel_frames = F.interpolate(
        pixel_frames, size=self.image_size, mode="bilinear", align_corners=False
    )  # [B*chunk, 3, 256, 256]
    pixel_frames = pixel_frames.reshape(B_v, T_pred, 3, *self.image_size)
    # [B, chunk, 3, H, W] → [B, 3, chunk, H, W] → [B, 3, 1, chunk, H, W]
    new_frames = pixel_frames.permute(0, 2, 1, 3, 4).unsqueeze(2)

    # 8. 更新 current_obs
    self.current_obs = torch.cat([self.current_obs, new_frames], dim=3)
    max_keep = self.num_history + self.chunk
    if self.current_obs.shape[3] > max_keep:
        self.current_obs = self.current_obs[:, :, :, -max_keep:, :, :]

    # 9. 更新 image_queue（追加最后一帧生成的 latent）
    for env_idx in range(B):
        last_pred_latent = pred_latents[env_idx, -1:, :, :, :]  # (1, 4, 72, 40)
        self.image_queue[env_idx].append(last_pred_latent.detach())

    # 10. 更新 action_queue（追加最后一步归一化动作）
    for env_idx in range(B):
        self.action_queue[env_idx].append(
            actions_tensor[env_idx, -1:, :].detach()  # (1, 14)
        )
```

#### 辅助方法：VAE 解码

```python
@torch.no_grad()
def _decode_latent(self, latents: torch.Tensor) -> torch.Tensor:
    """
    用 SVD-VAE 解码 latent → pixel [-1, 1]。
    
    Args:
        latents: [N, 4, H', W']，已乘以 scaling_factor 的 latent
    Returns:
        pixels: [N, 3, H, W] float32 [-1, 1]
    """
    vae = self.ctrlworld_model.vae
    dtype = next(vae.parameters()).dtype
    latents = latents.to(self.device, dtype)
    decoded = []
    for i in range(0, latents.shape[0], self.decode_chunk_size):
        chunk = latents[i:i + self.decode_chunk_size] / vae.config.scaling_factor
        decoded.append(vae.decode(chunk, num_frames=chunk.shape[0]).sample)
    pixels = torch.cat(decoded, dim=0)  # [N, 3, H, W], values in [-1, 1]
    return pixels.float()
```

---

### 2.5 `_infer_next_chunk_rewards`

Ctrl-World 使用与 Wan/OpenSora 相同的奖励模型，接口完全一致。

```python
def _infer_next_chunk_rewards(self) -> torch.Tensor:
    """对最新生成的 chunk 帧计算奖励。"""
    # current_obs: [B, 3, 1, T, H, W]，值域 [-1, 1]
    B, c, v, t, h, w = self.current_obs.shape
    # 取最后 chunk 帧
    chunk_obs = self.current_obs[:, :, 0, -self.chunk:, :, :]  # [B, 3, chunk, H, W]
    frames_flat = (
        chunk_obs.permute(0, 2, 1, 3, 4)
        .reshape(B * self.chunk, 3, h, w)
        .float()
    )  # [B*chunk, 3, H, W]

    rm_type = self.cfg.reward_model.type

    if rm_type == "ResnetRewModel":
        rewards = self.reward_model.predict_rew(frames_flat)
    elif rm_type == "TaskEmbedResnetRewModel":
        instructions = [
            self.task_descriptions[env_idx]
            for env_idx in range(B)
            for _ in range(self.chunk)
        ]
        rewards = self.reward_model.predict_rew(frames_flat, instructions)
    elif rm_type == "RoboTwinT5CrossAttn":
        # current_obs 值域 [-1,1] → 转换为 [0,1] 再送入奖励模型
        # 直接调用 compute_reward() 避免 predict_rew() 内的重复 ImageNet 归一化
        frames_01 = ((frames_flat + 1.0) / 2.0).clamp(0.0, 1.0)
        instructions = [
            self.task_descriptions[env_idx]
            for env_idx in range(B)
            for _ in range(self.chunk)
        ]
        rewards = self.reward_model.compute_reward(frames_01, task_descriptions=instructions)
    else:
        raise ValueError(f"Unknown reward model type: {rm_type}")

    return rewards.reshape(B, self.chunk)
```

---

### 2.6 `_wrap_obs`

```python
def _wrap_obs(self) -> dict:
    """
    从 current_obs 提取最后一帧，转换为 Policy 期望格式。
    
    Returns:
        {
            "main_images": Tensor [B, H, W, 3] uint8 [0,255]，head-cam 图像
            "wrist_images": None
            "states": Tensor [B, action_dim] float32，全零占位
            "task_descriptions": list[str]，长度=B
        }
    """
    # current_obs: [B, 3, 1, T, H, W]，值域 [-1, 1]
    last_frame = self.current_obs[:, :, 0, -1, :, :]   # [B, 3, H, W]
    full_image = last_frame.permute(0, 2, 3, 1)          # [B, H, W, 3]
    full_image = (full_image + 1.0) / 2.0 * 255.0
    full_image = torch.clamp(full_image, 0, 255).to(torch.uint8)

    return {
        "main_images": full_image,
        "wrist_images": None,
        "states": torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32,
                              device=self.device),
        "task_descriptions": self.task_descriptions,
    }
```

---

### 2.7 `chunk_step`

```python
def chunk_step(self, policy_output_action):
    """
    执行一个 chunk 推理步骤。
    
    Args:
        policy_output_action: [B, chunk, action_dim]，策略输出的原始动作
    Returns:
        (obs_list, chunk_rewards, chunk_terminations, chunk_truncations, info_list)
    """
    self.onload()

    actions = (
        policy_output_action.cpu().numpy()
        if isinstance(policy_output_action, torch.Tensor)
        else policy_output_action
    )

    # 推理下一段视频帧
    self._infer_next_chunk_frames(actions)

    # 计算奖励
    raw_rewards = self._infer_next_chunk_rewards()       # [B, chunk]
    chunk_rewards = self._calc_step_reward(raw_rewards)  # 差分奖励（可选）

    # 成功判定：任意帧奖励超过阈值
    success_threshold = self.cfg.get("success_reward_threshold", 0.9)
    estimated_success = raw_rewards.max(dim=1).values >= success_threshold  # [B]

    # 终止 / 截断标志（只在 chunk 最后一步置位）
    self.elapsed_steps += 1
    chunk_terminations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool, device=self.device)
    chunk_truncations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool, device=self.device)
    chunk_terminations[:, -1] = estimated_success
    chunk_truncations[:, -1] = (self.elapsed_steps >= self.cfg.max_episode_steps)

    # 指标记录
    step_reward = chunk_rewards[:, -1]
    infos = {}
    self._record_metrics(step_reward, estimated_success, infos)

    obs = self._wrap_obs()

    if self.cfg.get("enable_offload", False):
        self.offload()

    return [obs], chunk_rewards, chunk_terminations, chunk_truncations, [infos]
```

---

### 2.8 `offload` / `onload`

```python
def offload(self):
    """将 Ctrl-World 模型和奖励模型移至 CPU。"""
    if not self._is_offloaded:
        self.ctrlworld_model.to("cpu")
        self.reward_model.to("cpu")
        torch.cuda.empty_cache()
        self._is_offloaded = True

def onload(self):
    """将模型移回 GPU。"""
    if self._is_offloaded:
        self.ctrlworld_model.to(self.device)
        self.reward_model.to(self.device)
        self._is_offloaded = False
```

---

## Step 3：动作适配

**文件：** `rlinf/envs/action_utils.py`

在 `prepare_actions()` 的 `elif` 链中添加：

```python
elif env_type == SupportedEnvType.CTRLWORLD_WM:
    if wm_env_type == "robotwin":
        # Ctrl-World RobotWin：直接透传，不做变换
        chunk_actions = raw_chunk_actions
    else:
        raise NotImplementedError(f"Ctrl-World wm_env_type={wm_env_type} not implemented")
```

---

## Step 4：编写环境配置 YAML

**文件：** `examples/embodiment/config/env/ctrlworld_robotwin_click_bell.yaml`

```yaml
# Ctrl-World 世界模型环境配置（RobotWin click_bell 任务）

env_type: ctrlworld_wm
task_suite_name: robotwin_click_bell
wm_env_type: robotwin

total_num_envs: null   # 由训练配置覆盖

auto_reset: False
ignore_terminations: False
max_episode_steps: 200
max_steps_per_rollout_epoch: 200

use_rel_reward: True
reward_coef: 1.0

seed: 0
group_size: 1
use_fixed_reset_state_ids: True

is_eval: False

success_reward_threshold: 0.9

# ──────────────────────────────────────────────────────────────────────────
# Ctrl-World 生成参数
# ──────────────────────────────────────────────────────────────────────────
# chunk 必须与 actor.num_action_chunks 和 actor.openpi.action_chunk 保持一致
chunk: 5                # Ctrl-World num_frames（预测帧数），注意与 Wan/OpenSora 的 8 不同
num_history: 6          # 历史帧数
image_size: [256, 256]  # 解码后 resize 到的分辨率

# Ctrl-World 推理尺寸（SVD 原始分辨率）
width: 320
height: 192

# 动作维度（RobotWin 14-DoF）
action_dim: 14

# history_idx：非均匀历史采样（与 rollout_robotwin_click_bell.py 保持一致）
history_idx: [0, 0, -8, -6, -4, -2]

# SVD 推理步数
num_inference_steps: 50
guidance_scale: 1.0
decode_chunk_size: 7
motion_bucket_id: 127
fps: 7
text_cond: True
frame_level_cond: True
his_cond_zero: False

# ──────────────────────────────────────────────────────────────────────────
# 模型路径
# ──────────────────────────────────────────────────────────────────────────

# Ctrl-World 代码仓库路径（用于 sys.path 导入）
ctrlworld_repo_path: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World

# SVD 预训练模型路径（CrtlWorld.__init__ 需要）
svd_model_path: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World/stable-video-diffusion-img2vid

# CLIP 模型路径
clip_model_path: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World/clip-vit-base-patch32

# 已训练好的 Ctrl-World checkpoint（.pt state_dict）
ctrlworld_ckpt_path: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World/model_ckpt/robotwin_click_bell/checkpoint-20000.pt

# 动作归一化统计文件（extract_latent_robotwin.py 生成）
data_stat_path: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World/dataset_meta_info/robotwin_click_bell/stat.json

# 数据集路径（annotation + latent_videos + videos）
initial_image_path: /ML-vePFS/protected/tangyinzhou/RLinf/Ctrl-World/dataset_example/robotwin_click_bell
data_type: val

# ──────────────────────────────────────────────────────────────────────────
# 奖励模型
# ──────────────────────────────────────────────────────────────────────────
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: /manifold-obs/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth
  t5_model_name: /ML-vePFS/protected/tangyinzhou/RLinf/pretrained_models/t5-base

# ──────────────────────────────────────────────────────────────────────────
# 视频录制
# ──────────────────────────────────────────────────────────────────────────
video_cfg:
  save_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: False
```

---

## Step 5：编写训练配置 YAML

**文件：** `examples/embodiment/config/ctrlworld_robotwin_click_bell_grpo_openpi_pi05.yaml`

以 `wan_robotwin_click_bell_grpo_openpi_pi05.yaml` 为基础，修改以下部分：

```yaml
defaults:
  - _self_
  - env/ctrlworld_robotwin_click_bell   # ← 使用上面创建的环境配置

algorithm:
  adv_type: grpo
  loss_type: grpo_actor
  group_size: 4
  reward_coef: 1.0

env:
  train:
    total_num_envs: 4
    group_size: ${algorithm.group_size}
    chunk: 5              # ← Ctrl-World 预测 5 帧（注意不是 8）
    action_dim: 14
    reward_coef: ${algorithm.reward_coef}

rollout:
  model:
    model_path: /ML-vePFS/protected/tangyinzhou/RLinf/pi05_base

actor:
  model:
    num_action_chunks: 5    # ← 必须与 env.chunk 一致（Ctrl-World 是 5，非 8）
    action_dim: 14
    openpi:
      action_chunk: 5       # ← 同上
      action_env_dim: 14
```

> **⚠️ 关键差异**：Ctrl-World 的 `chunk=5`（`num_frames=5`），而 Wan 和 OpenSora 是 `chunk=8`。
> 因此 `actor.num_action_chunks` 和 `actor.openpi.action_chunk` 都必须改为 **5**。

---

## 关键约束与一致性检查清单

| 项目 | Ctrl-World 的值 | 检查点 |
|------|----------------|--------|
| chunk 一致性 | **5** | `env.chunk` == `actor.num_action_chunks` == `actor.openpi.action_chunk` |
| action_dim 一致性 | 14 | `env.action_dim` == `actor.action_dim` == `actor.openpi.action_env_dim` |
| action_cond 总帧数 | 11 | `num_history + chunk = 6 + 5 = 11`，须与 Action_encoder2 期望一致 |
| 动作归一化 | p01/p99 → [-1,1] | `data_stat_path` 来自 `dataset_meta_info/robotwin_click_bell/stat.json` |
| image_queue 格式 | latent `(1, 4, 72, 40)` | 不是 PIL Image（与 Wan 不同） |
| 多视图 latent | 3 路 H 拼接 | RobotWin 3 路完全相同；decode 时取 `H[0:24]` 即第 0 路 |
| GRPO 整除 | `total_num_envs % group_size == 0` | 同其他世界模型 |
| 图像归一化链路 | uint8 → latent → decode [-1,1] → _wrap_obs uint8 → 奖励模型 [0,1] | |

---

## 与其他世界模型的对比（Ctrl-World 视角）

| 特性 | Ctrl-World | Wan | OpenSora |
|------|-----------|-----|----------|
| **枚举值** | `ctrlworld_wm` | `wan_wm` | `opensora_wm` |
| **推理接口** | `CtrlWorldDiffusionPipeline.__call__(image, text, history, ...)` | `WanVideoPipeline(input_image, input_image4, action, ...)` | `scheduler.sample(model, z, y, mask, ...)` |
| **条件帧存储** | latent `(1,4,72,40)`，3路视图拼接 | PIL Image 列表（pixel space）| VAE latent（deque，latent space）|
| **历史帧采样** | `history_idx=[0,0,-8,-6,-4,-2]` 非均匀 | 最近 4 帧 | 最近 N 帧（deque）|
| **动作历史** | 6 帧，存 action_queue | 5 帧，`condition_action` Tensor | 不保留历史 |
| **动作归一化** | p01/p99 bound norm | 无（原始值） | q01/q99 分位数归一化 |
| **chunk 大小** | **5 帧** | 8 帧 | 8 帧 |
| **输入图像分辨率** | (192, 320) → resize 到 (256, 256) | 直接 (256, 256) | (256, 256) |
| **模型加载方式** | `CrtlWorld(args)` + `load_state_dict` | HuggingFace `from_pretrained` | 分组件加载 |

---

## 快速 Smoke Test

用单 GPU 验证环境初始化和基本流程：

```python
from omegaconf import OmegaConf
from rlinf.envs import get_env_cls

cfg = OmegaConf.load("examples/embodiment/config/env/ctrlworld_robotwin_click_bell.yaml")
cfg.total_num_envs = 1
cfg.group_size = 1

EnvCls = get_env_cls("ctrlworld_wm")
env = EnvCls(cfg, num_envs=1, seed_offset=0, total_num_processes=1)

# reset
obs, _ = env.reset()
print("main_images shape:", obs["main_images"].shape)   # 期望 (1, 256, 256, 3) uint8
print("states shape:", obs["states"].shape)              # 期望 (1, 14)

# 模拟 chunk_step
import numpy as np
fake_actions = np.zeros((1, 5, 14), dtype=np.float32)   # chunk=5
obs_list, rewards, terms, truncs, infos = env.chunk_step(fake_actions)
print("chunk rewards:", rewards.shape)                   # 期望 (1, 5)
print("terminations:", terms.shape)                      # 期望 (1, 5)
```

---

## 额外注意事项

### Ctrl-World 代码路径问题

由于 `CrtlWorld` 在独立的 `Ctrl-World/` 仓库中，需要通过 `cfg.ctrlworld_repo_path` 动态
加入 `sys.path`。若需更干净的集成，建议将 `Ctrl-World/models/` 拷贝或软链接到
`rlinf/models/embodiment/ctrlworld/` 并修改 import 路径。

### VAE scaling_factor

Ctrl-World 使用 SVD-VAE，其 `vae.config.scaling_factor` 非 1.0（通常 ≈ 0.18215）。
decode 时必须：`latents / vae.config.scaling_factor`，与 OpenSora 和 Wan 均相同。

### 动作重排顺序

原始 RobotWin 14-dim 动作顺序：`[L_joint(6), L_gripper(1), R_joint(6), R_gripper(1)]`。
`extract_latent_robotwin.py` 会将其重排为：`[L_joint(6), R_joint(6), L_gripper(1), R_gripper(1)]`
（`state_reordered`）存入 annotation 的 `states` 字段。使用 `states` 时注意此重排。

若使用 Policy（Pi05）输出的动作，则需确认 Policy 训练数据所用的动作顺序与 Ctrl-World
训练数据的 `states` 字段顺序一致。
