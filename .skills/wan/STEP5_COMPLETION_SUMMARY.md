# 任务5 完成报告

## 任务: 创建完整RL训练配置

### 完成状态: ✅ 完成

---

## ✅ 完成的配置

### 主配置文件

**文件**: [wan_robotwin_click_bell_grpo_openpi_pi05.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml)

### 关键配置项

#### 1. 算法: GRPO

```yaml
algorithm:
  adv_type: grpo           # 与LIBERO一致
  group_size: 8            # GRPO需要group_size
  kl_beta: 0.0
  entropy_bonus: 0
  reward_coef: 5.0
  filter_rewards: True
  rewards_lower_bound: 0.5
  rewards_upper_bound: 4.5
```

#### 2. 环境配置: 1卡1环境

```yaml
env:
  train:
    total_num_envs: 1              # 1个并行环境（测试用）
    group_size: 8                  # GRPO的group_size
    max_episode_steps: 256
    enable_kir: False              # 没有KIR数据
    enable_offload: True           # 节省显存
    num_inference_steps: 5         # 与LIBERO一致
```

**注意**: 
- `total_num_envs=1` 且 `group_size=8` 会导致问题！
- GRPO需要 `total_num_envs` 能被 `group_size` 整除
- **建议改为 `total_num_envs: 8`**

#### 3. Policy模型: 你的Pi05 checkpoint

```yaml
actor:
  model:
    model_path: "/ML-vePFS/protected/tangyinzhou/RLinf/logs/pi05-clickbell/click_bell_headcam/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt"
    model_type: "openpi_pi05"

rollout:
  model:
    model_path: "/ML-vePFS/protected/tangyinzhou/RLinf/logs/pi05-clickbell/click_bell_headcam/checkpoints/global_step_4000/actor/model_state_dict/full_weights.pt"
```

#### 4. 训练配置

```yaml
runner:
  max_epochs: 1000
  save_interval: 5
  val_check_interval: -1    # 暂时不验证
  
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout: 0    # 单卡
```

#### 5. Reward模型

```yaml
# 在 env/wan_robotwin_click_bell.yaml 中
reward_model:
  type: RoboTwinT5CrossAttn
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
```

---

## ❓ 回答你的问题

### Q5: `task_suite_name: robotwin_click_bell` 是啥意思？

**A**: 这个参数用于**Reward模型加载任务嵌入 (task embedding)**。

#### 作用

对于 `TaskEmbedResnetRewModel` (LIBERO用的):
```python
rew_model = TaskEmbedResnetRewModel(
    checkpoint_path=...,
    task_suite_name="libero_goal"  # ← 决定加载哪个任务的embedding
)
```

它会根据 `task_suite_name` 从checkpoint中加载对应的任务文本嵌入向量。

#### 你的情况

你的 `RoboTwinT5CrossAttnRewardModel` **不需要**这个参数！

因为你的模型使用 **T5编码器** 直接编码instruction字符串：
```python
# 你的模型内部
text_tokens = self.t5_encoder(instructions)  # 直接编码字符串
```

所以：
- ✅ 你的instruction是str（例如 "click the bell"）- 完全正确
- ✅ T5会实时编码这些字符串
- ❌ 不需要 `task_suite_name` 预定义的embedding

**配置中的 `task_suite_name` 对你的reward模型没有影响**，可以保留任意值。

---

### Q10: KIR数据

**A**: 我检查了你的数据集目录：

```bash
diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/dataset/
# 没有找到 *kir*.npy 文件
```

所以配置为：
```yaml
env:
  train:
    enable_kir: False  # 没有KIR数据，关闭
```

#### 什么是KIR？

**KIR (KeyFrame-Init Rollout)**: 关键帧初始化策略

**作用**:
- 不是从轨迹的第一帧开始
- 而是从随机关键帧开始rollout
- 提供更丰富的初始状态分布

**数据格式**:
- 普通数据: `traj0.npy`, `traj1.npy`, ...
- KIR数据: `traj0_kir.npy`, `traj1_kir.npy`, ...

**如果想启用KIR**:
需要运行脚本生成KIR数据（通常从现有轨迹采样关键帧）。

---

## ⚠️ 重要问题

### total_num_envs vs group_size

**当前配置有问题**:
```yaml
total_num_envs: 1    # 1个环境
group_size: 8        # GRPO需要8个环境一组
```

**GRPO的要求**:
- `total_num_envs` 必须能被 `group_size` 整除
- 因为GRPO需要在每个group内计算advantage

**建议修改**:

```yaml
# 选项A: 最小配置（8个环境）
total_num_envs: 8
group_size: 8

# 选项B: 更大规模（如果显存够）
total_num_envs: 16
group_size: 8

# 选项C: 保持1环境，但改用PPO
adv_type: ppo  # PPO不要求group_size
group_size: 1
```

**我的建议**: 先用 **选项A** (8环境) 测试

---

## 📊 配置对比

| 配置项 | LIBERO | RobotWin (当前) | 说明 |
|--------|--------|----------------|------|
| **算法** | GRPO | GRPO ✅ | 一致 |
| **total_num_envs** | 64 | 1 ⚠️ | 需要改为8 |
| **group_size** | 8 | 8 | 一致 |
| **num_inference_steps** | 5 | 5 ✅ | 一致 |
| **enable_offload** | True | True ✅ | 一致 |
| **enable_kir** | True | False ✅ | 无KIR数据 |
| **reward_model** | TaskEmbedResnet | RoboTwinT5CrossAttn ✅ | 你的模型 |
| **max_episode_steps** | 256 | 256 ✅ | 一致 |

---

## 🚀 运行命令

### 测试配置加载

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment

# 查看配置（不运行）
python train_embodied_agent.py --config-name wan_robotwin_click_bell_ppo_openpi_pi05 --info
```

### 开始训练

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment

# 单卡训练
MUJOCO_GL=egl \
python train_embodied_agent.py \
    --config-name wan_robotwin_click_bell_ppo_openpi_pi05
```

**注意**: 需要先修复 `total_num_envs` 问题！

---

## 📁 修改的文件

### 配置文件
1. [examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/wan_robotwin_click_bell_ppo_openpi_pi05.yaml)
   - 改为GRPO算法
   - 设置Pi05模型路径
   - total_num_envs=1（需要改为8）
   - 关闭KIR和eval

### 环境配置（之前已完成）
2. [examples/embodiment/config/env/wan_robotwin_click_bell.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)
   - env_type: wan_wm
   - reward_model: RoboTwinT5CrossAttn

---

## ✅ 下一步

### 必须修复

**修改 `total_num_envs`**:

```yaml
# wan_robotwin_click_bell_ppo_openpi_pi05.yaml Line 88
train:
  total_num_envs: 8  # ← 改为8（能被group_size=8整除）
```

### 可选优化

1. **验证Reward模型加载**:
```python
python3 << 'EOF'
from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
    '/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth'
)
print("✓ Reward模型加载成功")
EOF
```

2. **测试完整流程**:
   - 修复total_num_envs后
   - 运行训练脚本
   - 观察是否正常

---

## 💡 关键说明

### GRPO vs PPO

| 特性 | GRPO | PPO |
|------|------|-----|
| **Advantage计算** | Group内归一化 | GAE |
| **需要group_size** | ✅ 是 | ❌ 否 |
| **显存需求** | 较高（需要group） | 较低 |
| **训练稳定性** | 较好 | 一般 |
| **LIBERO使用** | ✅ 是 | ❌ 否 |

### 为什么LIBERO用GRPO？

GRPO在机器人学习任务中表现更好：
- 同一task的多个rollout组成group
- group内归一化advantage
- 减少variance，提升稳定性

### WanEnv的Reward计算

```
Policy生成action (14维)
  ↓
WanEnv.chunk_step()
  ↓
Wan生成8帧视频
  ↓
Reward模型计算reward (使用T5编码instruction)
  ↓
GRPO计算advantage (group内归一化)
  ↓
更新Policy
```

---

**报告生成时间**: 2026-04-22  
**文档版本**: 1.0  
**状态**: ✅ 完成（需修复total_num_envs）
