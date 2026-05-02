# RobotWin Reward模型集成方案

## 📊 问题分析

### WanEnv的Reward调用方式

**位置**: [world_model_wan_env.py Line 475, 498](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py#L475)

```python
# WanEnv期望reward模型有 predict_rew() 方法
rewards = self.reward_model.predict_rew(extract_chunk_obs, instructions)
# 输入: obs [B, 3, H, W], instructions list[str]
# 输出: rewards [B]
```

### 你的Reward模型接口

**类**: `RoboTwinT5CrossAttnRewardModel`  
**位置**: [robotwin_reward_model.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/models/embodiment/reward/robotwin_reward_model.py)

```python
# 你的是 forward() 方法
def forward(
    self,
    input_data: torch.Tensor,      # [B, C, H, W]
    labels: Optional[torch.Tensor] = None,
    instructions: Optional[list[str]] = None,
) -> dict[str, Any]:
    """返回: {'loss': ..., 'acc': ..., 'prob': ..., 'logits': ...}"""
```

**没有 `predict_rew()` 方法！**

---

## 💡 解决方案

### 方案A: 添加predict_rew包装方法 ✅ (最简单)

在 `RoboTwinT5CrossAttnRewardModel` 中添加一个包装方法：

```python
class RoboTwinT5CrossAttnRewardModel(BaseImageRewardModel):
    # ... 现有代码 ...
    
    @torch.no_grad()
    def predict_rew(
        self, 
        obs: torch.Tensor, 
        instructions: Optional[list[str]] = None
    ) -> torch.Tensor:
        """
        WanEnv兼容的reward预测接口。
        
        Args:
            obs: [B, 3, H, W] 图像输入，范围任意 (会自动预处理)
            instructions: list[str] 任务描述
            
        Returns:
            rewards: [B] 奖励值，范围 [0, 1]
        """
        # 1. 预处理图像
        obs = self.preprocess_images(obs)
        
        # 2. 调用forward
        output = self.forward(obs, instructions=instructions)
        
        # 3. 返回概率作为reward
        return output['prob']  # 或 output['logits']
```

### 方案B: 修改WanEnv适配你的模型

修改WanEnv的 `_infer_next_chunk_rewards()` 方法：

```python
def _infer_next_chunk_rewards(self):
    # ... 前面的代码相同 ...
    
    if self.cfg.reward_model.type == "RoboTwinT5CrossAttn":
        from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
        
        # 使用compute_reward接口
        observations = {
            'images': extract_chunk_obs,
            'main_images': extract_chunk_obs,
        }
        rewards = self.reward_model.compute_reward(
            observations, 
            task_descriptions=instructions
        )
        rewards = rewards.reshape(self.num_envs, self.chunk)
    
    # ... 其他类型 ...
```

### 方案C: 使用DiffSynth的TaskEmbedResnetRewModel ✅ (推荐)

如果你的reward模型架构与LIBERO的类似，可以尝试：

```yaml
# wan_robotwin_click_bell.yaml
reward_model:
  type: TaskEmbedResnetRewModel
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
  task_suite_name: robotwin_click_bell
```

**前提条件**:
- 你的模型checkpoint的state_dict结构与`TaskEmbedResnetRewModel`兼容
- 需要验证模型能否正确加载

---

## 🔧 推荐实施步骤

### Step 1: 验证模型checkpoint结构

```python
import torch

# 加载你的模型
checkpoint = torch.load(
    '/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth',
    map_location='cpu',
    weights_only=False
)

# 查看结构
print("Checkpoint keys:", checkpoint.keys())
if 'model_state_dict' in checkpoint:
    print("Model keys (前10个):")
    for i, key in enumerate(checkpoint['model_state_dict'].keys()):
        if i < 10:
            print(f"  {key}: {checkpoint['model_state_dict'][key].shape}")
```

### Step 2: 选择并实施解决方案

**我推荐方案A** (添加predict_rew)，因为：
1. ✅ 最小改动
2. ✅ 不影响现有代码
3. ✅ 保持WanEnv的通用性
4. ✅ 符合其他reward模型的设计模式

### Step 3: 更新配置

```yaml
# wan_robotwin_click_bell.yaml (Line 54-61)
reward_model:
  type: RoboTwinT5CrossAttn  # 新类型
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
```

### Step 4: 修改WanEnv加载逻辑

```python
# world_model_wan_env.py Line 148-162
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
        from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
        rew_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
            self.cfg.reward_model.from_pretrained
        )
    else:
        raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")
    
    return rew_model
```

---

## 📝 注意事项

### 图像尺寸差异

| 模型 | 输入图像尺寸 | 说明 |
|------|-------------|------|
| Wan世界模型输出 | 256x256 | 生成的视频帧 |
| Reward模型期望 | 224x224 | 训练时的尺寸 |

**解决**: `preprocess_images()` 会自动resize，不需要担心。

### 图像值域

| 来源 | 值域 | 说明 |
|------|------|------|
| WanEnv current_obs | [-1, 1] | 归一化后的tensor |
| Reward模型期望 | [0, 1] 或 uint8 | preprocess会处理 |

**解决**: `preprocess_images()` 会处理不同的输入值域。

### 调用流程中的值域转换

```python
# WanEnv._infer_next_chunk_rewards() Line 459-486
extract_chunk_obs = self.current_obs.permute(...)  # [-1, 1]
extract_chunk_obs = extract_chunk_obs.squeeze(2)   # [B, 3, 256, 256]
extract_chunk_obs = extract_chunk_obs.to(self.device)

# 如果reward模型内部调用preprocess_images:
# [-1, 1] → clamp到[0,1] → normalize → [0, 1]标准化
```

**需要检查**: 你的reward模型的`predict_rew`或`compute_reward`是否正确处理了[-1, 1]的输入。

---

## 🎯 最终建议

### 立即可行的方案

1. **在 `RoboTwinT5CrossAttnRewardModel` 中添加 `predict_rew()` 方法**
2. **在 WanEnv 中添加新类型支持**
3. **更新配置文件**

### 需要确认的问题

1. **你的reward模型checkpoint格式**:
   - 包含 `model_state_dict` 还是直接是state_dict?
   - 是否包含训练optimizer等其他信息?

2. **接口兼容性**:
   - `predict_rew()` 应该返回 `[B]` 还是 `[B, 1]`?
   - WanEnv期望的是 `[B]` (Line 476, 499 reshape)

3. **是否需要图像预处理**:
   - WanEnv传入的是 `[-1, 1]` 的tensor
   - 你的reward模型期望什么输入?

需要我帮你实施这些修改吗？
