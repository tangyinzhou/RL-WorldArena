# Wan World Model as Environment for VLA Training - RobotWin实战指南

本文档提供使用Wan世界模型作为环境训练Vision-Language-Action (VLA) 策略的完整技术指南，**专注于RobotWin环境的实际部署**。

## 项目背景与目标

### 当前进展

**已完成**:
- ✅ Wan世界模型训练完成 (4K轨迹，320步/轨迹)
  - 训练数据: `/manifold-obs/wzl/vla_robotwin_4k_320`
  - 编码数据: `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/encoded_data/click_bell`
  - 输出目录: `diffsynth-studio/outputs/click_bell`
  - 训练命令: 使用`accelerate launch`在GPU 4-7上训练
  
- ✅ RobotWin奖励模型训练完成
  - 训练脚本: `/ML-vePFS/protected/tangyinzhou/RLinf/debug_robotwin_reward_model.py`
  - 模型类型: ResnetRewModel (基于图像预测奖励)
  
- ✅ Policy SFT训练完成
  - 训练命令: `bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05_click_bell`
  - 模型: PI05单视角策略
  
- ✅ LIBERO示例数据已下载
  - 路径: `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-LIBERO-Spatial`
  - 用途: 参考数据格式和目录结构

### 目标

在RobotWin环境上复现Wan世界模型训练Policy的完整流程：
1. 使用Wan世界模型替代真实仿真器进行快速采样
2. 通过RL (PPO/GRPO) 提升Policy性能
3. 在真实RobotWin仿真器上评估训练效果

### 核心挑战

| 维度 | LIBERO (已实现) | RobotWin (待实现) | 差异点 |
|------|-----------------|-------------------|--------|
| Action维度 | 7 (xyz+rpy+gripper) | **14 (双臂关节+夹爪)** | ⚠️ 需要代码适配 |
| 相机视角 | 单视角/三视角 | **单视角(head)** | ✅ 已支持 |
| 夹爪控制 | -1表示打开 | **绝对位置控制** | ⚠️ 需要修改逻辑 |
| 奖励模型 | Resnet (通用) | **RobotWin专属** | ✅ 已训练 |
| 初始化数据 | LIBERO轨迹 | **RobotWin 4K轨迹** | ⚠️ 需要转换格式 |
| 环境类型 | `wm_env_type: libero` | **`wm_env_type: robotwin`** | ⚠️ 需要配置 |

## 完整任务清单与实施步骤

### 任务1: 准备Wan世界模型输出和初始化数据集 🔄 IN PROGRESS

#### 1.1 导出训练好的Wan模型权重

**当前状态**: Wan模型训练输出在 `diffsynth-studio/outputs/click_bell`

**需要做的**:
```bash
# 1. 检查训练输出目录
ls -lh /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/outputs/click_bell/

# 2. 找到最新的checkpoint
# 通常格式: outputs/click_bell/epoch_XXX/ 或类似结构

# 3. 导出为标准格式 (参考LIBERO的结构)
# 需要包含:
#   - DiT模型权重 (safetensors格式)
#   - VAE权重 (可使用预训练的Wan2.2_VAE.pth)
#   - 配置文件
```

**目标目录结构**:
```
/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/
├── dit_model.safetensors        # DiT模型权重
├── Wan2.2_VAE.pth               # VAE权重
├── resnet_rm.pth                # 奖励模型
└── dataset/                     # 初始化数据集
    ├── traj0.npy
    ├── traj1.npy
    ├── ...
    ├── traj0_kir.npy
    └── trajN_kir.npy
```

#### 1.2 生成RobotWin初始化数据集

**数据源**: `/manifold-obs/wzl/vla_robotwin_4k_320`

**转换脚本** (需要创建):
```python
# 位置: rlinf/data/datasets/world_model/convert_robotwin_to_npy.py
import numpy as np
import torch
from pathlib import Path

def convert_robotwin_trajectory(src_dir, dst_dir, enable_kir=True):
    """
    将RobotWin轨迹转换为Wan环境所需的npy格式
    
    Args:
        src_dir: 原始RobotWin数据路径
        dst_dir: 输出npy文件路径
        enable_kir: 是否生成KIR版本
    """
    # 1. 加载RobotWin数据
    # 2. 提取初始帧 (start_items)
    # 3. 提取前4个关键帧 (target_items) - 用于KIR
    # 4. 保存为npy格式
    
    for traj_idx, trajectory in enumerate(trajectories):
        data = {
            'start_items': [
                {
                    'image': trajectory[0]['head_camera'],  # [3, 256, 256]
                    'observation.state': trajectory[0]['qpos']  # [14]
                }
            ],
            'target_items': [
                {
                    'image': trajectory[i]['head_camera'],
                    'action': trajectory[i]['action']  # [14] 注意是14维!
                }
                for i in range(1, 5)  # 前4个关键帧
            ],
            'task': "click the bell"
        }
        
        np.save(dst_dir / f"traj{traj_idx}.npy", data)
        
        if enable_kir:
            np.save(dst_dir / f"traj{traj_idx}_kir.npy", data)
```

**关键注意事项**:
- ⚠️ RobotWin action是**14维**，不是7维
- 图像需要resize到 `[3, 256, 256]`
- 图像值范围 `[0, 1]` (会在WanEnv中归一化到[-1, 1])

#### 1.3 验证数据集格式

```python
# 验证脚本
import numpy as np
from pathlib import Path

def verify_dataset(dataset_dir):
    for npy_file in Path(dataset_dir).glob("*.npy"):
        data = np.load(npy_file, allow_pickle=True).item()
        
        # 检查必需字段
        assert 'start_items' in data, f"Missing start_items in {npy_file}"
        assert 'target_items' in data, f"Missing target_items in {npy_file}"
        assert 'task' in data, f"Missing task in {npy_file}"
        
        # 检查图像格式
        img = data['start_items'][0]['image']
        assert img.shape == (3, 256, 256), f"Wrong image shape: {img.shape}"
        
        # 检查action维度 (RobotWin是14维)
        if data['target_items']:
            action = data['target_items'][0]['action']
            assert action.shape == (14,), f"Wrong action dim: {action.shape}"
        
        print(f"✓ {npy_file.name} validated")

verify_dataset("/path/to/dataset/")
```

---

### 任务2: 修改WanEnv支持RobotWin 14维Action ⚠️ CRITICAL

#### 2.1 修改 `world_model_wan_env.py`

**文件位置**: `/ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py`

**需要修改的地方**:

```python
# 修改1: condition_action初始化 (第95-99行)
# 原代码:
self.condition_action = torch.zeros(
    self.num_envs,
    self.condition_frame_length,
    7,  # ❌ 硬编码7维
)

# 修改为:
self.action_dim = cfg.get("action_dim", 7)  # ✅ 可配置
self.condition_action = torch.zeros(
    self.num_envs,
    self.condition_frame_length,
    self.action_dim,
)

# 修改2: 环境类型判断 (第102-106行)
# 原代码:
self.is_libero_env = cfg.get("wm_env_type", "libero") == "libero"
if self.reset_gripper_open and self.is_libero_env:
    self.condition_action[:, :, -1] = -1

# 修改为:
self.wm_env_type = cfg.get("wm_env_type", "libero")
self.is_libero_env = self.wm_env_type == "libero"
self.is_robotwin_env = self.wm_env_type == "robotwin"

# LIBERO夹爪打开设为-1
if self.reset_gripper_open and self.is_libero_env:
    self.condition_action[:, :, -1] = -1

# RobotWin不需要特殊处理 (使用绝对位置控制)
# 如果RobotWin有特殊的初始化需求，在这里添加

# 修改3: reset时的condition_action (第342-348行)
# 原代码:
env_condition_action = np.zeros(
    (self.condition_frame_length, 7), dtype=np.float32
)

# 修改为:
env_condition_action = np.zeros(
    (self.condition_frame_length, self.action_dim), dtype=np.float32
)
```

#### 2.2 添加配置项

在WanEnv的config中需要支持:
```yaml
env:
  train:
    action_dim: 14  # RobotWin使用14维action
    wm_env_type: robotwin  # 标识环境类型
```

---

### 任务3: 创建Wan环境配置文件 📝

#### 3.1 创建 `env/wan_robotwin_click_bell.yaml`

**文件位置**: `examples/embodiment/config/env/wan_robotwin_click_bell.yaml`

**完整配置**:
```yaml
# RobotWin Wan世界模型环境配置
env_type: wan_wm
task_suite_name: robotwin_click_bell
wm_env_type: robotwin  # 关键: 标识为RobotWin环境

total_num_envs: null

auto_reset: False
ignore_terminations: False
max_steps_per_rollout_epoch: 400
max_episode_steps: 400

use_rel_reward: True
reward_coef: 1.0

# Wan环境特定设置
reset_gripper_open: False  # RobotWin不使用-1表示打开
is_eval: False

seed: 0
group_size: 1
use_fixed_reset_state_ids: True
use_ordered_reset_state_ids: False
specific_reset_id: null

video_cfg:
  save_video: True
  info_on_video: True
  video_base_dir: ${runner.logger.log_path}/video/train

enable_offload: True  # 防止OOM

### Wan模型参数 ###

wan_wm_hf_ckpt_path: /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell

# 模型路径 (相对于wan_wm_hf_ckpt_path)
VAE_path: ${env.train.wan_wm_hf_ckpt_path}/Wan2.2_VAE.pth
model_path: ${env.train.wan_wm_hf_ckpt_path}/dit_model.safetensors

# 初始化数据集
enable_kir: True  # 启用关键帧初始化
initial_image_path: ${env.train.wan_wm_hf_ckpt_path}/dataset/

# Wan推理参数
num_inference_steps: 5  # 推理步数 (5为质量和速度的平衡)
chunk: 8  # action chunk长度
condition_frame_length: 5  # 条件帧长度
image_size: [256, 256]  # 输出图像尺寸
num_frames: 13  # 总帧数 = condition_frame_length + chunk

# Action维度配置
action_dim: 14  # RobotWin 14维action (双臂6+1 + 6+1)

# 奖励模型配置
reward_model:
  type: ResnetRewModel
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
```

---

### 任务4: 创建完整RL训练配置 🚀

#### 4.1 创建 `wan_robotwin_click_bell_ppo_openpi_pi05.yaml`

**文件位置**: `examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml`

**完整配置**:
```yaml
defaults:
  - env/wan_robotwin_click_bell@env.train  # 训练用Wan环境
  - env/robotwin_click_bell@env.eval       # 评估用真实环境
  - model/pi0_5@actor.model
  - training_backend/fsdp@actor.fsdp_config
  - override hydra/job_logging: stdout

hydra:
  run:
    dir: .
  output_subdir: null
  searchpath:
    - file://${oc.env:EMBODIED_PATH}/config/

cluster:
  num_nodes: 1
  component_placement:
    actor, env, rollout: 0-7  # 8张GPU

runner:
  task_type: embodied
  logger:
    log_path: "${oc.env:REPO_PATH}/logs/wan-robotwin-rl"
    project_name: "rlinf_wan_robotwin"
    experiment_name: "click_bell_wan_ppo_pi05"
    logger_backends: ["wandb", "tensorboard"]

  max_epochs: 1000
  max_steps: -1

  only_eval: False
  val_check_interval: 10
  save_interval: 10

  resume_dir: null
  ckpt_path: null

algorithm:
  normalize_advantages: True
  kl_penalty: kl
  group_size: 8  # 分组大小
  reward_coef: 1.0

  rollout_epoch: 1
  eval_rollout_epoch: 1

  reward_type: chunk_level
  logprob_type: chunk_level
  entropy_type: token_level

  update_epoch: 5
  adv_type: gae
  loss_type: actor_critic
  loss_agg_func: "token-mean"
  kl_beta: 0.0
  entropy_bonus: 0
  clip_ratio_high: 0.2
  clip_ratio_low: 0.2
  clip_ratio_c: 3.0
  value_clip: 0.2
  huber_delta: 10.0

  gamma: 0.99
  gae_lambda: 0.95

  filter_rewards: False
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9

  sampling_params:
    do_sample: True
    temperature_train: 1.0
    temperature_eval: 0.6
    top_k: 50
    top_p: 1.0
    repetition_penalty: 1.0
    add_BOS: False

  length_params:
    max_new_token: null
    max_length: 1024
    min_length: 1

env:
  group_name: "EnvGroup"
  enable_offload: True
  
  # 训练环境 (Wan世界模型)
  train:
    total_num_envs: 64
    reward_coef: ${algorithm.reward_coef}
    max_episode_steps: 400
    max_steps_per_rollout_epoch: 400
    group_size: ${algorithm.group_size}
    
    # Wan世界模型路径
    wan_wm_hf_ckpt_path: /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell
    
    enable_offload: True
    num_inference_steps: 5
  
  # 评估环境 (真实RobotWin仿真器)
  eval:
    total_num_envs: 64
    auto_reset: True
    ignore_terminations: True
    max_episode_steps: 400
    reward_coef: ${algorithm.reward_coef}
    max_steps_per_rollout_epoch: 400
    group_size: 1
    use_fixed_reset_state_ids: True
    is_eval: True
    assets_path: "/ML-vePFS/protected/tangyinzhou/RLinf/RoboTwin"
    seeds_path: ${oc.env:REPO_PATH}/rlinf/envs/robotwin/seeds/eval_seeds.json
    video_cfg:
      save_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval
    center_crop: False
    task_config:
      embodiment: [aloha-agilex]
      camera:
        collect_head_camera: true
        collect_wrist_camera: false
      domain_randomization:
        random_background: false
        cluttered_table: false
        clean_background_rate: 1
        random_head_camera_dis: 0
        random_table_height: 0
        random_light: false
        crazy_random_light_rate: 0

rollout:
  group_name: "RolloutGroup"
  backend: "huggingface"
  recompute_logprobs: False
  enable_offload: True
  pipeline_stage_num: 1
  model:
    model_path: ${actor.model.model_path}
    precision: ${actor.model.precision}

actor:
  group_name: "ActorGroup"
  training_backend: "fsdp"
  micro_batch_size: 32
  global_batch_size: 512
  seed: 1234
  enable_offload: True

  model:
    # SFT训练好的Policy路径
    model_path: "/path/to/sft_checkpoint/actor"
    num_action_chunks: 50
    action_dim: 14  # RobotWin 14维
    add_value_head: True
    num_steps: 5
    openpi:
      config_name: "pi05_aloha_robotwin_head"
      num_images_in_input: 1
      noise_level: 0.3
      detach_critic_input: True

  optim:
    lr: 5.0e-06
    value_lr: 1.0e-04
    adam_beta1: 0.9
    adam_beta2: 0.95
    adam_eps: 1.0e-08
    weight_decay: 0.01
    clip_grad: 1.0
    critic_warmup_steps: 0

  fsdp_config:
    strategy: "fsdp"
    gradient_checkpointing: False
    mixed_precision:
      param_dtype: ${actor.model.precision}
      reduce_dtype: ${actor.model.precision}
      buffer_dtype: ${actor.model.precision}

reward:
  use_reward_model: False

critic:
  use_critic_model: False
```

---

### 任务5: 测试Wan环境初始化 🧪

#### 5.1 单元测试脚本

```bash
# 测试Wan环境
cd /ML-vePFS/protected/tangyinzhou/RLinf
python -m rlinf.envs.world_model.world_model_wan_env
```

**测试检查清单**:
- [ ] Wan模型加载成功
- [ ] VAE模型加载成功
- [ ] 奖励模型加载成功
- [ ] 初始化数据集加载成功
- [ ] Reset返回正确格式的观测
- [ ] chunk_step生成视频帧
- [ ] 奖励预测正常

#### 5.2 手动测试脚本

创建测试脚本 `test_wan_robotwin.py`:
```python
import os
os.environ['EMBODIED_PATH'] = 'examples/embodiment'

from hydra import compose, initialize_config_dir
from rlinf.envs.world_model.world_model_wan_env import WanEnv

config_dir = "examples/embodiment/config"
config_name = "wan_robotwin_click_bell_ppo_openpi_pi05"

with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
    cfg = compose(config_name=config_name)
    env_cfg = cfg['env']['train']

# 创建环境
env = WanEnv(env_cfg, env_cfg.total_num_envs, seed_offset=0, total_num_processes=1)

# 测试reset
obs, info = env.reset()
print(f"✓ Reset成功")
print(f"  obs keys: {obs.keys()}")
print(f"  main_images shape: {obs['main_images'].shape}")
print(f"  task_descriptions: {obs['task_descriptions'][:2]}")

# 测试chunk_step
import numpy as np
dummy_actions = np.zeros((env_cfg.total_num_envs, 8, 14))  # [batch, chunk, 14]
obs_list, rewards, terminations, truncations, infos = env.chunk_step(dummy_actions)
print(f"✓ Chunk step成功")
print(f"  rewards shape: {rewards.shape}")
print(f"  rewards: {rewards[0]}")
```

---

### 任务6: 启动RL训练 🚀

#### 6.1 启动命令

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf

# 设置环境变量
export REPO_PATH=/ML-vePFS/protected/tangyinzhou/RLinf
export EMBODIED_PATH=examples/embodiment

# 启动训练
bash examples/embodiment/run_embodiment.sh wan_robotwin_click_bell_ppo_openpi_pi05
```

#### 6.2 监控训练

```bash
# TensorBoard
tensorboard --logdir logs/wan-robotwin-rl --port 6006

# 浏览器打开
# http://localhost:6006
```

**关键监控指标**:
- `env/success_once`: 任务成功率 (最重要)
- `train/actor/policy_loss`: Policy损失
- `rollout/rewards`: 环境奖励
- `train/actor/approx_kl`: KL散度 (应 < 0.01)
- `train/actor/clip_fraction`: 裁剪比例

#### 6.3 常见问题排查

**OOM问题**:
```yaml
# 解决: 减小环境数或启用offload
env:
  train:
    total_num_envs: 32  # 从64降到32
    enable_offload: True
```

**奖励稀疏**:
```yaml
# 解决: 调整奖励系数和阈值
algorithm:
  reward_coef: 5.0  # 增大奖励权重
  filter_rewards: True
  rewards_lower_bound: 0.1
  rewards_upper_bound: 0.9
```

**生成质量差**:
```yaml
# 解决: 增加推理步数
env:
  train:
    num_inference_steps: 10  # 从5增加到10
```

---

### 任务7: 评估与验证 📊

#### 7.1 定期评估

```bash
# 创建评估配置 (基于训练配置，但env.eval使用真实环境)
bash examples/embodiment/eval_embodiment.sh wan_robotwin_click_bell_ppo_openpi_pi05_eval
```

#### 7.2 成功率对比

记录训练进度:
```
Step 0 (SFT基线): ?%
Step 1000: ?%
Step 5000: ?%
Step 10000: ?%
```

#### 7.3 视频可视化

```yaml
# 在评估配置中启用视频保存
env:
  eval:
    video_cfg:
      save_video: True
      video_base_dir: ${runner.logger.log_path}/video/eval
```

查看生成的视频，直观判断策略质量。

---

## 核心架构

Wan世界模型允许在无需真实机器人或物理仿真器的情况下，通过视觉生成模型模拟环境动态变化，为VLA策略的强化学习训练提供闭环环境。

## 核心架构

```
Policy (PI05/OpenVLA) → Action Chunk → Wan World Model → Generated Frames → Reward Model → RL Update
```

## 1. Policy Action 输出格式

### 输出形状
```
[num_envs, num_action_chunks, action_dim]
```

### Action维度说明

**LIBERO环境 (7维)**:
- `[0:3]`: 末端执行器3D位置 (x, y, z)
- `[3:6]`: 3D旋转 (roll, pitch, yaw)
- `[6]`: 夹爪控制 (开/合)

**RoboTwin双臂环境 (14维)**:
- `[0:6]`: 左臂6关节
- `[6]`: 左臂夹爪
- `[7:13]`: 右臂6关节
- `[13]`: 右臂夹爪

### 归一化状态

**Policy内部处理**:
- 训练和推理时使用归一化action（基于`norm_stats.json`统计信息）
- 归一化方法: 基于训练数据的分位数(q01, q99)或均值方差

**Policy输出**:
- 通过`AlohaOutputs._encode_actions()`进行反归一化
- **输出到环境的是原始物理空间的action，未经归一化**

### 配置示例
```yaml
actor:
  model:
    num_action_chunks: 50  # 每次输出的action chunk数量
    action_dim: 14         # RoboTwin
    # action_dim: 7        # LIBERO
    openpi:
      config_name: "pi05_aloha_robotwin_head"
      action_chunk: 5      # 每个chunk包含的步数
      action_env_dim: 7    # 环境动作维度
```

## 2. Wan世界模型Action输入

### 输入格式
```
[num_envs, chunk_length, action_dim]
```

对于LIBERO环境：
- `chunk_length = 8` (配置项`cfg.chunk`)
- `action_dim = 7`

### 归一化状态

**Wan环境不对action进行归一化处理**：

```python
# world_model_wan_env.py:501-506
# Normalize actions (仅做类型转换，无归一化)
actions_tensor = (
    torch.from_numpy(actions).to(self.device)
    if isinstance(actions, np.ndarray)
    else actions.to(self.device)
)
```

**Action流转**:
```
Policy输出(原始空间) → Wan环境(直接使用) → 生成视频帧
```

### 条件Action机制

Wan维护`condition_action`用于自回归生成：
```python
# 初始化时设置
self.condition_action = torch.zeros(
    self.num_envs,
    self.condition_frame_length,  # 默认5
    7,
)

# 如果夹爪打开(LIBERO)
if self.reset_gripper_open and self.is_libero_env:
    self.condition_action[:, :, -1] = -1  # gripper维度设为-1
```

生成时拼接：
```python
if self.retain_action:
    # [condition_action, new_actions] → [B, 5+8, 7] = [B, 13, 7]
    actions_tensor = torch.cat([self.condition_action, actions_tensor], dim=1)
```

## 3. 环境初始化

### 数据结构

Wan环境使用`.npy`文件存储轨迹数据：

```
dataset/
├── traj0.npy          # 仅含初始帧的轨迹
├── traj1.npy
├── ...
├── traj0_kir.npy      # 含关键帧的轨迹(KIR模式)
└── trajN_kir.npy
```

每个`.npy`文件包含：
- `start_items`: 初始帧列表
- `target_items`: 关键帧列表（用于KIR）
- `task`: 任务描述字符串

### 初始化流程

```python
# 1. 选择episode
episode_indices = np.random.choice(len(self.dataset), size=num_envs, replace=False)

# 2. 加载初始帧
for env_idx, episode_idx in enumerate(episode_indices):
    episode_data = self.dataset[episode_idx]
    first_frame = episode_data["start_items"][0]
    img_tensor = first_frame["image"]  # [3, H, W], [0,1]
    
# 3. 图像归一化到[-1, 1]
img_tensor = self.trans_norm(img_tensor)  # mean=0.5, std=0.5

# 4. 填充条件帧 (重复condition_frame_length次)
env_img_tensor = img_tensor.unsqueeze(1).repeat(1, 5, 1, 1)

# 5. KIR模式: 使用前4个关键帧替换条件帧
if len(target_items) == 4:  # condition_frame_length - 1
    for target_idx, target_frame in enumerate(target_items):
        env_img_tensor[:, target_idx + 1] = target_frame["image"]
        env_condition_action[target_idx + 1] = target_frame["action"]

# 6. 初始化image_queue
for env_idx in range(num_envs):
    frames = [self.current_obs[env_idx, :, 0, t_idx:t_idx+1, :, :] 
              for t_idx in range(5)]
    self.image_queue[env_idx] = frames
```

### KIR (KeyFrame-Init Rollout)

```yaml
env:
  train:
    enable_kir: True  # 启用关键帧初始化
    # True:  从所有npy文件(含kir)等概率初始化
    # False: 仅从不含kir的npy文件初始化
```

**优势**: KIR提供更丰富的初始状态分布，避免仅从轨迹起点开始训练。

## 4. 多环境并行机制

### 架构设计

```
┌─────────────────────────────────────┐
│     WanVideoPipeline (单例)         │
│  - 共享DiT模型                      │
│  - 共享VAE模型                      │
│  - 批量推理 (batch_size=num_envs)   │
└─────────────────────────────────────┘
           ↗           ↖
    ┌──────┴─────┐ ┌───┴──────┐
    │  Env 0     │ │ Env N    │
    │ - queue    │ │ - queue  │
    │ - obs      │ │ - obs    │
    │ - task     │ │ - task   │
    └────────────┘ └──────────┘
```

### 关键特性

**共享组件**:
- `WanVideoPipeline`: 世界模型推理引擎
- `ResnetRewModel`: 奖励模型

**独立状态** (每个env维护):
- `image_queue`: 最近帧队列 `[condition_frame_length]`
- `current_obs`: 当前观测 `[C, 1, T, H, W]`
- `task_descriptions`: 任务描述
- `condition_action`: 条件动作

### 配置示例
```yaml
env:
  train:
    total_num_envs: 64     # 总环境数
    group_size: 8          # 分组大小
    # 实际组数: 64 / 8 = 8组
```

### 批量推理
```python
# 一次性处理所有环境
B = num_envs
kwargs = {
    "input_image": batch_input_image,      # List[PIL], len=B
    "input_image4": batch_input_image4,    # List[List[PIL]], B×4
    "action": actions_tensor,              # [B, T, A], T=13
    "batch_size": B,
    ...
}
output = self.pipe(**kwargs)  # 返回List[PIL], len=B
```

## 5. 奖励机制

### 奖励模型类型

#### ResnetRewModel
```python
# 输入: 图像 [batch, 3, H, W], 范围[-1, 1]
# 架构: ResNet → GlobalAvgPool → FC(512→1) → Sigmoid
# 输出: [0, 1], 二值化后为 0 或 1

class ResnetRewModel(nn.Module):
    def predict_rew(self, obs):
        obs = obs.clamp(-1.0, 1.0)
        x = self.net(obs.to(dtype=torch.float32))
        x = torch.round(x)  # 二值化
        return x
```

**特点**:
- 仅使用图像输入
- 不区分任务，通用奖励

#### TaskEmbedResnetRewModel (推荐)
```python
# 输入: 
#   - 图像 [batch, 3, H, W]
#   - 任务描述 (自然语言) → task_id (0-9) → embedding
# 架构: 
#   - 视觉分支: ResNet → [batch, 512]
#   - 任务分支: Embedding(10, 64) → [batch, 64]
#   - 融合层: Linear(576→256→1) → Sigmoid
# 输出: [0, 1], 二值化后为 0 或 1
```

**任务映射示例** (LIBERO-Spatial):
```python
LIBERO_SPATIAL_TASKS = {
    0: "pick up the black bowl between the plate and the ramekin and place it on the plate",
    1: "pick up the black bowl from table center and place it on the plate",
    # ... 共10个任务
}
```

### 奖励计算流程

```python
# 1. 提取生成的chunk帧
extract_chunk_obs = self.current_obs[:, -self.chunk:, :, :, :, :]
# [num_envs, chunk, 3, v, h, w]

# 2. 预测奖励
if reward_model_type == "ResnetRewModel":
    rewards = self.reward_model(extract_chunk_obs)
elif reward_model_type == "TaskEmbedResnetRewModel":
    instructions = [task_desc for _ in range(self.chunk) 
                    for task_desc in self.task_descriptions]
    rewards = self.reward_model(extract_chunk_obs, instructions)

# 3. 计算step reward (差分奖励)
reward_diffs = reward_coef * chunk_rewards - prev_step_reward
prev_step_reward = reward_coef * chunk_rewards

# 4. Termination判定
success_threshold = 0.9
max_reward_in_chunk = chunk_rewards.max(dim=1)[0]
success_estimated = max_reward_in_chunk >= success_threshold
```

### 配置示例
```yaml
env:
  train:
    reward_model:
      type: TaskEmbedResnetRewModel
      from_pretrained: /path/to/resnet_rm.pth
    reward_coef: 1.0
    success_reward_threshold: 0.9
```

## 完整训练流程

```yaml
# 1. Reset
obs, info = env.reset()
# obs = {main_images, wrist_images=None, states, task_descriptions}

# 2. Policy预测
actions, result = policy.predict_action_batch(obs, mode="train")
# actions: [num_envs, num_action_chunks, action_dim]

# 3. 环境step
obs_list, rewards, terminations, truncations, infos = env.chunk_step(actions)
# rewards: [num_envs, chunk]

# 4. RL更新
advantages, returns = compute_gae(rewards, values, dones)
actor_loss = compute_ppo_loss(logprobs, advantages, ...)
```

## 关键配置参考

```yaml
actor:
  model:
    model_type: "openpi"
    num_action_chunks: 50
    action_dim: 7  # LIBERO: 7, RoboTwin: 14
    unnorm_key: "libero_spatial_no_noops"

env:
  train:
    wm_env_type: libero
    task_suite_name: libero_spatial
    enable_kir: True
    chunk: 8
    condition_frame_length: 5
    num_inference_steps: 5
    initial_image_path: /path/to/dataset
    wan_wm_hf_ckpt_path: /path/to/wan_weights
    reward_model:
      type: TaskEmbedResnetRewModel
      from_pretrained: /path/to/reward_model.pth

rollout:
  model:
    model_type: ${actor.model.model_type}
    num_action_chunks: ${actor.model.num_action_chunks}
```

## 关键文件路径汇总

### 已完成的组件

| 组件 | 路径 | 状态 |
|------|------|------|
| Wan世界模型训练输出 | `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/outputs/click_bell` | ✅ 完成 |
| RobotWin原始数据 | `/manifold-obs/wzl/vla_robotwin_4k_320` | ✅ 完成 |
| 编码数据 | `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/encoded_data/click_bell` | ✅ 完成 |
| RobotWin奖励模型 | 使用 `debug_robotwin_reward_model.py` 训练 | ✅ 完成 |
| Policy SFT模型 | SFT训练输出 | ✅ 完成 |
| LIBERO示例数据 | `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-LIBERO-Spatial` | ✅ 完成 |

### 需要创建的文件

| 文件 | 路径 | 状态 |
|------|------|------|
| Wan环境配置 | `examples/embodiment/config/env/wan_robotwin_click_bell.yaml` | ⏳ 待创建 |
| RL训练配置 | `examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml` | ⏳ 待创建 |
| 评估配置 | `examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05_eval.yaml` | ⏳ 待创建 |
| 数据集转换脚本 | `rlinf/data/datasets/world_model/convert_robotwin_to_npy.py` | ⏳ 待创建 |
| 测试脚本 | `test_wan_robotwin.py` | ⏳ 待创建 |

### 需要修改的文件

| 文件 | 修改内容 | 状态 |
|------|---------|------|
| `rlinf/envs/world_model/world_model_wan_env.py` | 支持14维action，添加robotwin环境类型 | ⏳ 待修改 |

---

## 实施检查清单

### 阶段1: 数据准备 (1-2天)

- [ ] 导出Wan模型权重为标准格式
- [ ] 创建数据集转换脚本
- [ ] 生成RobotWin初始化数据集 (npy格式)
- [ ] 验证数据集格式正确
- [ ] 组织完整目录结构

### 阶段2: 代码适配 (1天)

- [ ] 修改WanEnv支持14维action
- [ ] 添加robotwin环境类型判断
- [ ] 更新condition_action初始化逻辑
- [ ] 创建Wan环境配置文件
- [ ] 创建RL训练配置文件

### 阶段3: 测试验证 (1天)

- [ ] 单元测试通过
- [ ] 环境reset正常
- [ ] chunk_step生成视频
- [ ] 奖励模型预测正常
- [ ] 观测格式正确

### 阶段4: RL训练 (持续)

- [ ] 启动训练
- [ ] 监控训练指标
- [ ] 解决训练问题
- [ ] 定期评估
- [ ] 记录实验结果

---

## 常见问题FAQ

### RobotWin特有问题

### Q: RobotWin的action维度是14维，Wan环境能处理吗？
**A**: 需要修改代码。当前WanEnv硬编码了7维action，需要改为可配置的`action_dim`参数。详见任务2的代码修改说明。

### Q: RobotWin的夹爪控制与LIBERO有什么不同？
**A**: 
- **LIBERO**: 使用-1表示夹爪打开，是归一化后的值
- **RobotWin**: 使用绝对位置控制 (关节角度)，范围约为 `[0, 0.04]`
- 因此RobotWin不需要在初始化时特殊设置夹爪值

### Q: 如何验证Wan生成的视频质量？
**A**: 
1. 保存训练时的视频到`video_cfg.video_base_dir`
2. 定期人工检查视频质量
3. 对比Wan生成视频与真实RobotWin视频
4. 如果生成质量差，可以尝试:
   - 增加`num_inference_steps` (5→10)
   - 检查训练数据质量
   - 调整模型超参数

### Q: 奖励模型准确率如何保证？
**A**:
1. 使用RobotWin仿真器的reward函数生成训练标签
2. 确保训练数据覆盖成功和失败案例
3. 验证奖励模型在测试集上的准确率
4. 定期对比奖励模型预测与真实reward

### Q: 训练时OOM怎么办？
**A**: 按优先级尝试:
1. 启用offload: `enable_offload: True`
2. 减少环境数: `total_num_envs: 64 → 32`
3. 减少推理步数: `num_inference_steps: 5 → 3`
4. 使用梯度检查点 (但OpenPI不支持)

### Q: 如何判断训练是否在进步？
**A**: 关注以下指标:
1. **`env/success_once`**: 最直接的成功率指标
2. **`rollout/rewards`**: 环境奖励趋势
3. **`train/actor/policy_loss`**: 应逐渐下降
4. **视频可视化**: 人工检查动作质量
5. **定期评估**: 在真实RobotWin环境测试

### Q: Wan世界模型 vs 真实仿真器训练有什么差异？
**A**:
| 维度 | Wan世界模型 | 真实仿真器 |
|------|-------------|-----------|
| 速度 | 快 (GPU推理) | 慢 (物理仿真) |
| 准确性 | 近似 (生成模型) | 精确 |
| 并行度 | 高 (批量推理) | 受限 |
| 适用场景 | 快速探索 | 最终评估 |

**建议**: 用Wan训练，用真实仿真器评估。

---

## 通用问题

### Q: Policy输出的action是否需要额外归一化？
**A**: 不需要。Policy已经输出原始物理空间的action，Wan环境直接使用。

### Q: 为什么LIBERO的gripper动作是-1表示打开？
**A**: 这是LIBERO环境的约定。初始化时设置`condition_action[:, :, -1] = -1`表示夹爪打开状态。

### Q: 多环境并行是否会影响生成质量？
**A**: 不会。批量推理仅提升效率，每个环境的生成是独立的（通过不同seed控制）。

### Q: 奖励稀疏导致训练困难怎么办？
**A**: 
1. 使用`enable_kir=True`提供更丰富的初始状态
2. 增加`total_num_envs`提高采样效率
3. 调整`reward_coef`和`success_reward_threshold`
4. 考虑使用相对奖励(`use_rel_reward: True`)

## 相关文件

### RobotWin实战相关

- **Wan环境实现**: `rlinf/envs/world_model/world_model_wan_env.py` (需要修改)
- **奖励模型训练**: `/ML-vePFS/protected/tangyinzhou/RLinf/debug_robotwin_reward_model.py`
- **RobotWin环境配置**: `examples/embodiment/config/env/robotwin_click_bell.yaml`
- **Policy配置**: `examples/embodiment/config/model/pi0_5.yaml`
- **数据转换参考**: `rlinf/data/datasets/world_model/` (需要创建转换脚本)

### LIBERO参考实现

- **LIBERO Wan配置**: `examples/embodiment/config/env/wan_libero_spatial.yaml`
- **LIBERO RL配置**: `examples/embodiment/config/wan_libero_spatial_grpo_openvlaoft.yaml`
- **LIBERO环境**: `rlinf/envs/libero/libero_env.py`
- **奖励模型**: `rlinf/models/embodiment/reward/wan_reward_model.py`

### Policy实现

- **OpenPI模型**: `rlinf/models/embodiment/openpi/openpi_action_model.py`
- **Aloha策略**: `rlinf/models/embodiment/openpi/policies/aloha_policy.py`
- **RobotWin数据配置**: `rlinf/models/embodiment/openpi/dataconfig/robotwin_aloha_head_dataconfig.py`

---

## 参考资料

- Wan世界模型: 基于[DiffSynth-Studio](https://github.com/RLinf/diffsynth-studio)框架
- OpenPI策略: Physical Intelligence的VLA架构
- LIBERO基准: 机器人操作任务套件
- RobotWin平台: 双臂机器人仿真环境

---

## 下一步行动

**立即需要做的**:

1. **导出Wan模型权重** - 检查`outputs/click_bell`目录，找到最佳checkpoint
2. **创建数据集转换脚本** - 将RobotWin 4K轨迹转为npy格式
3. **修改WanEnv代码** - 支持14维action和robotwin环境类型
4. **创建配置文件** - 基于LIBERO模板创建RobotWin配置

**预计时间线**:
- 第1-2天: 数据准备
- 第3天: 代码适配和配置
- 第4天: 测试验证
- 第5天+: RL训练

**成功标准**:
- ✅ Wan环境能正常初始化
- ✅ chunk_step生成合理视频
- ✅ 奖励模型预测准确
- ✅ RL训练稳定收敛
- ✅ 评估成功率提升
