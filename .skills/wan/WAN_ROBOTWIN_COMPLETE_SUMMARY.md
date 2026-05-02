# Wan世界模型RobotWin完整配置总结

## 📋 任务列表完成情况

| 任务 | 状态 | 说明 |
|------|------|------|
| **任务1**: 准备Wan模型和数据集 | ✅ 完成 | 数据转换脚本 + 验证脚本 |
| **任务2**: 修改WanEnv支持14维action | ✅ 完成 | action_dim可配置 + action_key支持 |
| **任务3**: 创建Wan环境配置 | ✅ 完成 | env_type修复 + reward配置 |
| **任务4**: 集成Reward模型 | ✅ 完成 | predict_rew + from_pretrained |
| **任务5**: 创建完整RL训练配置 | ✅ 完成 | GRPO配置 + Pi05路径 |

---

## 🎯 完整配置体系

### 1. 环境配置

**文件**: [env/wan_robotwin_click_bell.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)

```yaml
env_type: wan_wm                        # ✅ 与SupportedEnvType匹配
wm_env_type: robotwin                   # ✅ 内部类型判断
task_suite_name: robotwin_click_bell    # 对T5模型无影响

# Wan模型路径
wan_wm_hf_ckpt_path: null               # 由主配置设置
VAE_path: ${env.train.wan_wm_hf_ckpt_path}/Wan2.2_VAE.pth
model_path: ${env.train.wan_wm_hf_ckpt_path}/dit_model.safetensors

# 数据集路径
initial_image_path: ${env.train.wan_wm_hf_ckpt_path}/dataset/

# 视频生成参数
chunk: 8
condition_frame_length: 5
image_size: [256, 256]
num_frames: 13

# RobotWin特定配置
action_dim: 14                          # ✅ 14维绝对action
action_key: abs_action                  # ✅ 数据集字段名
reset_gripper_open: False               # ✅ 绝对位置控制

# Reward模型
reward_model:
  type: RoboTwinT5CrossAttn             # ✅ 新类型
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth

# KIR
enable_kir: False                       # ✅ 无KIR数据
```

### 2. 训练配置

**文件**: [wan_robotwin_click_bell_ppo_openpi_pi05.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml)

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: 0                # ✅ 单卡配置

algorithm:
  adv_type: grpo                        # ✅ GRPO算法
  group_size: 8                         # ✅ Group大小
  reward_coef: 5.0
  filter_rewards: True

env:
  train:
    total_num_envs: 8                   # ✅ 8个环境（匹配group_size）
    group_size: ${algorithm.group_size} # 8
    max_episode_steps: 256
    enable_offload: True                # ✅ 节省显存
    num_inference_steps: 5              # ✅ 与LIBERO一致
    enable_kir: False                   # ✅ 无KIR数据
    wan_wm_hf_ckpt_path: "/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell"

actor:
  model:
    model_path: "/ML-vePFS/protected/tangyinzhou/RLinf/logs/pi05-clickbell/click_bell_headcam/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt"
    model_type: "openpi_pi05"

rollout:
  model:
    model_path: "/ML-vePFS/protected/tangyinzhou/RLinf/logs/pi05-clickbell/click_bell_headcam/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt"

runner:
  max_epochs: 1000
  save_interval: 5
  val_check_interval: -1                # ✅ 暂时不验证
```

---

## 🔧 代码修改总结

### 修改的文件

| 文件 | 修改内容 | 行号 |
|------|---------|------|
| **world_model_wan_env.py** | action_dim可配置 | L96 |
| | condition_action使用action_dim | L98-101 |
| | 添加wm_env_type判断 | L104-106 |
| | _build_dataset支持action_key | L123-131 |
| | _load_reward_model支持新类型 | L148-167 |
| | reset使用action_dim | L348 |
| **robotwin_reward_model.py** | 添加predict_rew方法 | L333-363 |
| | 添加from_pretrained方法 | L365-425 |
| **wan_robotwin_click_bell.yaml** | env_type改为wan_wm | L1 |
| | 启用reward_model配置 | L58-61 |
| **wan_robotwin_click_bell_ppo_openpi_pi05.yaml** | 完整GRPO配置 | 全部 |

---

## 📊 数据流完整流程

### 训练时的数据流

```
1. 初始化
   ├─ 加载Wan模型 (dit_model.safetensors + VAE)
   ├─ 加载Reward模型 (resnet_rm.pth)
   ├─ 加载Policy模型 (Pi05 checkpoint)
   └─ 加载数据集 (dataset/*.npy)

2. Reset (环境重置)
   ├─ 从数据集采样trajectory
   ├─ 提取start_frame (第1帧)
   ├─ 设置task_description (instruction)
   └─ 返回观测: {main_images, states, task_descriptions}

3. Policy Rollout (生成action)
   ├─ Policy接收观测
   ├─ 生成14维action chunk (8步)
   └─ 返回: actions [num_envs, 8, 14]

4. WanEnv chunk_step (世界模型步进)
   ├─ _infer_next_chunk_frames(actions)
   │   └─ Wan生成8帧视频 [num_envs, 3, 1, 13, 256, 256]
   ├─ _infer_next_chunk_rewards()
   │   └─ Reward模型计算reward
   │       ├─ 值域转换: [-1, 1] → [0, 1]
   │       ├─ preprocess: 256→224 + ImageNet normalize
   │       ├─ T5编码instruction
   │       └─ 返回: rewards [num_envs, 8]
   └─ _calc_step_reward()
       └─ 奖励差分计算
           └─ 返回: chunk_rewards_tensors [num_envs, 8]

5. GRPO更新
   ├─ 计算advantage (group内归一化)
   ├─ 计算policy loss
   ├─ 反向传播
   └─ 更新Policy权重

6. 循环
   └─ 重复步骤2-5，直到epoch结束
```

### Reward计算详细流程

```
WanEnv.current_obs [num_envs, 3, 1, 13, 256, 256]  (值域: [-1, 1])
    ↓
提取最后8帧
    ↓
[num_envs*8, 3, 256, 256]
    ↓
RoboTwinT5CrossAttnRewardModel.predict_rew()
    ↓
值域检测: obs.min() < 0?
    ├─ 是 → obs = (obs + 1.0) / 2.0  (转换为[0, 1])
    └─ 否 → 保持原样
    ↓
preprocess_images()
    ├─ Resize: 256x256 → 224x224
    └─ Normalize: ImageNet mean/std
    ↓
T5编码instruction
    ↓
Cross-Attention (visual × text)
    ↓
MLP Head
    ↓
Sigmoid → rewards [num_envs*8]  (值域: [0, 1])
    ↓
reshape回 [num_envs, 8]
```

---

## 🚀 运行指南

### 前置检查

```bash
# 1. 检查Wan模型文件
ls -lh /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/
# 应该有:
#   - dit_model.safetensors
#   - Wan2.2_VAE.pth
#   - resnet_rm.pth
#   - dataset/ (包含*.npy文件)

# 2. 检查Pi05模型
ls -lh /ML-vePFS/protected/tangyinzhou/RLinf/logs/pi05-clickbell/click_bell_headcam/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt

# 3. 检查数据集
ls /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/dataset/ | head -5
# 应该有: traj0.npy, traj1.npy, ...
```

### 测试配置

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment

# 查看配置信息
python train_embodied_agent.py \
    --config-name wan_robotwin_click_bell_ppo_openpi_pi05 \
    --info
```

### 开始训练

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment

# 单卡训练
MUJOCO_GL=egl \
python train_embodied_agent.py \
    --config-name wan_robotwin_click_bell_ppo_openpi_pi05
```

### 监控训练

```bash
# TensorBoard
tensorboard --logdir ../logs --port 6006

# 访问: http://localhost:6006
```

---

## ⚠️ 注意事项

### 显存需求

| 组件 | 显存估算 |
|------|---------|
| Wan模型 (DiT + VAE) | ~8-10 GB |
| Reward模型 (T5 + ResNet) | ~2-3 GB |
| Policy (Pi05) | ~4-6 GB |
| 8个环境 + 激活 | ~4-6 GB |
| **总计** | **~18-25 GB** |

**建议**: 使用A100 40GB或更高显存的GPU

### 如果OOM

1. **减少num_inference_steps**:
```yaml
env:
  train:
    num_inference_steps: 3  # 从5降到3
```

2. **启用更多offload**:
```yaml
# 已在配置中启用
env.train.enable_offload: True
actor.enable_offload: True
rollout.enable_offload: True
```

3. **减少total_num_envs** (但要保持能被group_size整除):
```yaml
env:
  train:
    total_num_envs: 8  # 最小值
```

### 训练时间估算

- 每步推理 (Wan生成8帧): ~2-3秒
- 每epoch (8环境 × 256步 / 8 chunk): ~512步
- 每epoch时间: ~15-25分钟
- 1000 epochs: ~10-17天

**建议**: 先用少量epochs测试 (如10-20)

---

## 📁 生成的文件

### 配置文件
- [examples/embodiment/config/env/wan_robotwin_click_bell.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)
- [examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml)

### 数据脚本
- [rlinf/data/datasets/world_model/convert_robotwin_to_libero_format.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/data/datasets/world_model/convert_robotwin_to_libero_format.py)
- [rlinf/data/datasets/world_model/verify_libero_format_dataset.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/data/datasets/world_model/verify_libero_format_dataset.py)

### 代码修改
- [rlinf/envs/world_model/world_model_wan_env.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py)
- [rlinf/models/embodiment/reward/robotwin_reward_model.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/models/embodiment/reward/robotwin_reward_model.py)

### 文档
- [.skills/STEP1_COMPLETION_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/STEP1_COMPLETION_SUMMARY.md)
- [.skills/STEP2_FINAL_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/STEP2_FINAL_SUMMARY.md)
- [.skills/STEP3_COMPLETION_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/STEP3_COMPLETION_SUMMARY.md)
- [.skills/STEP4_COMPLETION_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/STEP4_COMPLETION_SUMMARY.md)
- [.skills/STEP5_COMPLETION_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/STEP5_COMPLETION_SUMMARY.md)
- [.skills/REWARD_AND_ISSUES_ANALYSIS.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/REWARD_AND_ISSUES_ANALYSIS.md)
- [.skills/REWARD_MODEL_INTEGRATION.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/REWARD_MODEL_INTEGRATION.md)
- [.skills/WAN_ROBOTWIN_COMPLETE_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/WAN_ROBOTWIN_COMPLETE_SUMMARY.md) (本文档)

---

## 🎉 完成状态

**所有5个任务已完成！**

现在可以:
1. ✅ 检查前置条件
2. ✅ 测试配置加载
3. ✅ 开始训练

祝训练顺利！🚀

---

**文档版本**: 1.0  
**完成时间**: 2026-04-22  
**状态**: ✅ 全部完成
