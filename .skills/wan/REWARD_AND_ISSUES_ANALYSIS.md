# 问题1、2和Reward流程分析

## 🔴 问题1、2: env_type差异的影响

### 差异1: `env_type`

```yaml
# LIBERO
env_type: wan_wm

# RobotWin
env_type: world_model_wan
```

### 影响位置

**文件**: [rlinf/envs/__init__.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/__init__.py#L31)

```python
class SupportedEnvType(Enum):
    # ...
    WANWM = "wan_wm"  # ← 只注册了 wan_wm
    # 没有 world_model_wan
```

**影响**:
- 如果使用 `env_type: world_model_wan`，`get_env_cls()` 会找不到这个类型
- **会导致启动失败！**

### ✅ 解决方案

**选项A**: 统一使用 `wan_wm`（推荐）
```yaml
# RobotWin配置改为
env_type: wan_wm  # 与LIBERO一致
```

**选项B**: 在SupportedEnvType中添加新类型
```python
class SupportedEnvType(Enum):
    WANWM = "wan_wm"
    WORLD_MODEL_WAN = "world_model_wm"  # 新增
```

**建议**: 使用选项A，因为代码中只用 `wan_wm`。

---

### 差异2: `model_path` 文件名

```yaml
# LIBERO
model_path: ${env.train.wan_wm_hf_ckpt_path}/model-00001.safetensors

# RobotWin
model_path: ${env.train.wan_wm_hf_ckpt_path}/dit_model.safetensors
```

### 影响

**文件**: [world_model_wan_env.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py#L134)

```python
def _build_pipeline(self):
    pipe = WanVideoPipeline.from_pretrained(
        model_configs=[
            ModelConfig(path=self.cfg.model_path, offload_device="cpu"),  # ← 使用配置的路径
            ModelConfig(path=self.cfg.VAE_path, offload_device="cpu"),
        ],
    )
```

**影响**:
- 只是文件路径不同，不影响逻辑
- 确保你的目录中有 `dit_model.safetensors` 文件即可

---

## 🎯 Reward模型完整流程分析

### 📊 Reward计算流程图

```
Policy生成Action (14维)
    ↓
WanEnv.chunk_step(policy_output_action)
    ↓
[1] 调用 _infer_next_chunk_frames(actions)
    → Wan世界模型生成8帧视频
    → 更新 self.current_obs [num_envs, 3, 1, 13, 256, 256]
    
[2] 调用 _infer_next_chunk_rewards()
    → 从 current_obs 提取最近8帧
    → 调用 reward_model.predict_rew(observations, instructions)
    → 返回 rewards [num_envs, chunk]  例如: [64, 8]
    
[3] 调用 _calc_step_reward(chunk_rewards)
    → 计算奖励差分: reward_diff = reward_coef * chunk_reward - prev_step_reward
    → 返回 chunk_rewards_tensors [num_envs, chunk]
    
[4] 返回给RL训练循环
    return (
        [extracted_obs],              # 观测
        chunk_rewards_tensors,         # 奖励 [num_envs, chunk]
        chunk_terminations,            # 终止信号
        chunk_truncations,             # 截断信号
        [infos]                        # 额外信息
    )
```

### 🔍 详细代码解析

#### 步骤1: Reward模型预测

**位置**: `_infer_next_chunk_rewards()` (Line 452-503)

```python
def _infer_next_chunk_rewards(self):
    # 1. 从current_obs提取chunk观测
    # current_obs shape: [num_envs, 3, 1, 13, 256, 256]
    extract_chunk_obs = self.current_obs.permute(0, 3, 1, 2, 4, 5)
    # 变成: [num_envs, 13, 3, 1, 256, 256]
    
    # 2. 只取最近chunk帧 (最后8帧)
    extract_chunk_obs = extract_chunk_obs[:, -self.chunk:, :, :, :, :]
    # 变成: [num_envs, 8, 3, 1, 256, 256]
    
    # 3. reshape用于reward模型
    extract_chunk_obs = extract_chunk_obs.reshape(
        self.num_envs * self.chunk, 3, 1, 256, 256
    )
    # 变成: [64*8, 3, 1, 256, 256] = [512, 3, 1, 256, 256]
    
    extract_chunk_obs = extract_chunk_obs.squeeze(2)
    # 变成: [512, 3, 256, 256]  (去掉view维度)
    
    # 4. 准备instructions (每个env重复chunk次)
    instructions = []
    for env_idx in range(self.num_envs):  # 64个环境
        task_desc = self.task_descriptions[env_idx]  # "click the bell"
        instructions.extend([task_desc] * self.chunk)  # 每个重复8次
    # instructions长度: 64 * 8 = 512
    
    # 5. 调用reward模型
    rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions)
    # rewards shape: [512]
    
    # 6. reshape回 [num_envs, chunk]
    rewards = rewards.reshape(self.num_envs, self.chunk)
    # rewards shape: [64, 8]
    
    return rewards
```

#### 步骤2: Reward差分计算

**位置**: `_calc_step_reward()` (Line 207-217)

```python
def _calc_step_reward(self, chunk_rewards):
    """
    计算奖励差分 (reward shaping)
    
    目的: 将绝对奖励转换为增量奖励
    - 如果奖励增加 → 正奖励
    - 如果奖励减少 → 负奖励
    - 鼓励持续进步
    """
    reward_diffs = torch.zeros(
        (self.num_envs, self.chunk), dtype=torch.float32, device=self.device
    )
    
    for i in range(self.chunk):  # 遍历8个时间步
        reward_diffs[:, i] = (
            self.cfg.reward_coef * chunk_rewards[:, i]  # 当前步奖励 * 系数
            - self.prev_step_reward  # 减去上一步的奖励
        )
        # 更新prev_step_reward供下一步使用
        self.prev_step_reward = self.cfg.reward_coef * chunk_rewards[:, i]
    
    return reward_diffs  # [num_envs, chunk]
```

**示例**:
```python
# 假设某个env的原始reward: [0.1, 0.2, 0.35, 0.5, 0.65, 0.75, 0.85, 0.95]
# reward_coef = 5.0

# 计算过程:
# step 0: 5.0 * 0.1 - 0 = 0.5
# step 1: 5.0 * 0.2 - 0.5 = 0.5
# step 2: 5.0 * 0.35 - 1.0 = 0.75
# step 3: 5.0 * 0.5 - 1.75 = 0.75
# ...

# 最终返回增量奖励
```

#### 步骤3: 成功判定

**位置**: `_estimate_success_from_rewards()` (需要查看实现)

```python
# 基于奖励估计是否成功
estimated_success = self._estimate_success_from_rewards(chunk_rewards)

# 创建terminations (只在chunk的最后一步标记)
raw_chunk_terminations = torch.zeros(self.num_envs, self.chunk, dtype=torch.bool)
raw_chunk_terminations[:, -1] = estimated_success  # 只在第8步标记
```

#### 步骤4: 返回给RL循环

**位置**: `chunk_step()` 返回值 (Line 739-745)

```python
return (
    [extracted_obs],              # 观测: dict with 'main_images', 'states', etc.
    chunk_rewards_tensors,         # 奖励: [num_envs, chunk] 差分奖励
    chunk_terminations,            # 终止: [num_envs, chunk] 布尔值
    chunk_truncations,             # 截断: [num_envs, chunk] 布尔值
    [infos]                        # 额外信息
)
```

---

## 🎯 你的Reward模型配置建议

### 你的Reward模型信息

**训练脚本**: [debug_robotwin_reward_model.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/debug_robotwin_reward_model.py)  
**模型路径**: `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth`  
**模型类型**: `robotwin_t5_crossattn` (使用T5文本编码 + Cross-Attention)

### ⚠️ 关键问题

**WanEnv不直接支持 `robotwin_t5_crossattn` 类型！**

WanEnv只支持两种reward模型类型 (Line 148-151):
```python
if self.cfg.reward_model.type == "ResnetRewModel":
    rew_model = ResnetRewModel(...)
elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
    rew_model = TaskEmbedResnetRewModel(...)
```

但你训练的是 `RoboTwinT5CrossAttnRewardModel`，这是不同的类！

### 🔍 模型对比

| 特性 | TaskEmbedResnetRewModel (LIBERO) | RoboTwinT5CrossAttnRewardModel (你的) |
|------|----------------------------------|---------------------------------------|
| **文本编码** | Task embedding (简单) | T5-base (强大) |
| **视觉编码** | ResNet | ResNet |
| **融合方式** | 拼接后MLP | Cross-Attention |
| **图像尺寸** | 可能需要224x224 | 224x224 |
| **预测接口** | `predict_rew(obs, instructions)` | `predict_rew(obs, instructions)` |

### 💡 解决方案

#### 方案A: 修改WanEnv支持你的模型类型 ✅ (推荐)

**步骤1**: 在WanEnv中添加新类型支持

修改 [world_model_wan_env.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py#L148-L157):

```python
def _load_reward_model(self):
    if self.cfg.reward_model.type == "ResnetRewModel":
        from diffsynth.models.reward_model import ResnetRewModel
        rew_model = ResnetRewModel(self.cfg.reward_model.from_pretrained)
    elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
        from diffsynth.models.reward_model import TaskEmbedResnetRewModel
        rew_model = TaskEmbedResnetRewModel(
            checkpoint_path=self.cfg.reward_model.from_pretrained,
            task_suite_name=self.cfg.task_suite_name,
        )
    elif self.cfg.reward_model.type == "RoboTwinT5CrossAttn":
        # 新增: 支持RobotWin reward模型
        from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
        rew_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
            self.cfg.reward_model.from_pretrained
        )
    else:
        raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")
    
    return rew_model
```

**步骤2**: 配置Wan环境

```yaml
# wan_robotwin_click_bell.yaml
reward_model:
  type: RoboTwinT5CrossAttn  # 新类型
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
```

**步骤3**: 确认接口兼容

需要检查 `RoboTwinT5CrossAttnRewardModel` 是否有 `predict_rew()` 方法，以及签名是否与WanEnv调用一致。

---

## 💡 建议的操作方案

### 方案1: 使用你训练的Reward模型 ✅ (推荐)

**步骤**:

1. **取消注释并修改配置**:
```yaml
# wan_robotwin_click_bell.yaml
reward_model:
  type: ResnetRewModel  # 或 TaskEmbedResnetRewModel，取决于训练时的类型
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
```

2. **确认模型类型**:
   - 如果训练时用了T5文本编码 → `TaskEmbedResnetRewModel`
   - 如果只用图像 → `ResnetRewModel`

3. **验证路径**:
```bash
ls -lh /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth
```

### 方案2: 暂时不用Reward模型

如果想先测试WanEnv是否能跑通：

```yaml
# wan_robotwin_click_bell.yaml
# 保持reward_model注释
```

WanEnv代码会检测:
```python
if self.reward_model is None:
    raise ValueError("Reward model is not loaded")
```

**所以要么配置reward_model，要么修改代码跳过。**

### 方案3: 修改代码支持可选Reward模型

修改 `_infer_next_chunk_rewards()`:
```python
def _infer_next_chunk_rewards(self):
    if self.reward_model is None:
        # 返回零奖励或基于其他指标的奖励
        return torch.zeros(self.num_envs, self.chunk, device=self.device)
    
    # ... 正常逻辑
```

---

## 📝 总结

### Reward流程回答你的问题:

**Q**: reward是在世界模型接口返回，还是世界模型返回之后处理？

**A**: **世界模型返回之后处理**

具体流程:
1. Wan世界模型生成视频帧 (`_infer_next_chunk_frames`)
2. 从生成的视频中提取帧 (`_infer_next_chunk_rewards`)
3. 调用独立的reward模型计算奖励
4. 对奖励做差分处理 (`_calc_step_reward`)
5. 返回给RL训练循环

**所以**:
- Wan世界模型**只负责生成视频**
- Reward模型是**独立的组件**
- 两者在WanEnv中**串联使用**

### 你需要做的:

1. **修复env_type**: 改为 `wan_wm`
2. **配置reward_model**: 取消注释并设置正确路径
3. **确认reward模型类型**: 查看训练配置确定是 `ResnetRewModel` 还是 `TaskEmbedResnetRewModel`

需要我帮你检查和修改配置吗？
