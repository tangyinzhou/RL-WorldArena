# 任务4 完成报告

## 任务: 修复env_type并集成Reward模型

### 完成状态: ✅ 完成

---

## ✅ 完成的修改

### 1. 修复env_type配置

**文件**: [wan_robotwin_click_bell.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/env/wan_robotwin_click_bell.yaml#L1)

```yaml
# 修改前:
env_type: world_model_wan  # ❌ 未注册，会导致启动失败

# 修改后:
env_type: wan_wm           # ✅ 与SupportedEnvType.WANWM匹配
```

**影响**: 
- 现在可以正确通过`get_env_cls()`找到WanEnv类
- 与LIBERO配置保持一致

---

### 2. 配置Reward模型

**文件**: [wan_robotwin_click_bell.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/env/wan_robotwin_click_bell.yaml#L58-L61)

```yaml
# 修改前 (注释掉):
# reward_model:
#   type: TaskEmbedResnetRewModel
#   from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/taskemb_resnet_rm.pth

# 修改后 (启用):
reward_model:
  type: RoboTwinT5CrossAttn    # ✅ RobotWin专用类型
  from_pretrained: ${env.train.wan_wm_hf_ckpt_path}/resnet_rm.pth
```

**配置说明**:
- `type: RoboTwinT5CrossAttn` - 新增的reward模型类型
- `from_pretrained` - 指向你训练的checkpoint路径
- 路径解析: `/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth`

---

### 3. 添加predict_rew方法

**文件**: [robotwin_reward_model.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/models/embodiment/reward/robotwin_reward_model.py#L333-L363)

```python
@torch.no_grad()
def predict_rew(
    self,
    obs: torch.Tensor,
    instructions: Optional[list[str]] = None,
) -> torch.Tensor:
    """WanEnv兼容的reward预测接口。
    
    - WanEnv传入: obs [B, 3, H, W] (值域可能是[-1, 1]或[0, 1])
    - WanEnv期望返回: rewards [B]
    """
    # 1. 处理输入值域 [-1, 1] → [0, 1]
    if obs.min() < 0:
        obs = (obs + 1.0) / 2.0
    
    # 2. 使用BaseImageRewardModel的预处理 (resize + ImageNet normalize)
    obs = self.preprocess_images(obs)
    
    # 3. 调用compute_reward
    return self.compute_reward(obs, task_descriptions=instructions)
```

**关键特性**:
- ✅ 自动处理值域转换 (WanEnv的[-1, 1] → 训练时的[0, 1])
- ✅ 复用preprocess_images (resize到224x224 + ImageNet normalize)
- ✅ 调用现有的compute_reward方法
- ✅ 返回格式与WanEnv期望一致 [B]

---

### 4. 添加from_pretrained类方法

**文件**: [robotwin_reward_model.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/models/embodiment/reward/robotwin_reward_model.py#L365-L425)

```python
@classmethod
def from_pretrained(
    cls,
    checkpoint_path: str,
    config: Optional[dict] = None,
) -> "RoboTwinT5CrossAttnRewardModel":
    """从checkpoint加载预训练模型。"""
    # 1. 使用默认配置 (与训练时一致)
    default_config = {
        "model_type": "robotwin_t5_crossattn",
        "t5_model_name": "t5-base",
        "freeze_t5": True,
        "max_text_length": 64,
        "num_attn_heads": 8,
        "attn_dropout": 0.0,
        "hidden_dim": 256,
        "head_dropout": 0.1,
        "image_size": [3, 224, 224],
        "normalize": True,
        # ...
    }
    
    # 2. 创建模型实例
    model = cls(cfg)
    
    # 3. 加载checkpoint (支持直接state_dict格式)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    
    return model
```

**关键特性**:
- ✅ 使用与训练时相同的默认配置
- ✅ 支持你的checkpoint格式 (直接state_dict)
- ✅ 自动处理T5模型加载
- ✅ 打印加载进度信息

---

### 5. WanEnv支持新Reward类型

**文件**: [world_model_wan_env.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py#L148-L167)

```python
def _load_reward_model(self):
    if self.cfg.reward_model.type == "ResnetRewModel":
        from diffsynth.models.reward_model import ResnetRewModel
        rew_model = ResnetRewModel(self.cfg.reward_model.from_pretrained)
    elif self.cfg.reward_model.type == "TaskEmbedResnetRewModel":
        from diffsynth.models.reward_model import TaskEmbedResnetRewModel
        rew_model = TaskEmbedResnetRewModel(...)
    elif self.cfg.reward_model.type == "RoboTwinT5CrossAttn":
        # ✅ 新增: RobotWin T5 Cross-Attention Reward模型
        from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel
        rew_model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
            self.cfg.reward_model.from_pretrained
        )
    else:
        raise ValueError(f"Unknown reward model type: {self.cfg.reward_model.type}")
    
    return rew_model
```

---

## 📊 数据流完整梳理

### Reward计算流程

```
1. WanEnv.chunk_step(policy_output_action)
   ↓
2. _infer_next_chunk_frames(actions)
   → Wan生成8帧视频
   → self.current_obs [num_envs, 3, 1, 13, 256, 256]  值域: [-1, 1]
   ↓
3. _infer_next_chunk_rewards()
   → 提取最后8帧: [num_envs*8, 3, 256, 256]
   → 调用: reward_model.predict_rew(obs, instructions)
   ↓
4. RoboTwinT5CrossAttnRewardModel.predict_rew()
   → 值域转换: [-1, 1] → [0, 1]
   → preprocess_images: 
     * Resize: 256x256 → 224x224
     * Normalize: ImageNet mean/std
   → compute_reward(obs, instructions)
   → 返回: rewards [num_envs*8]  值域: [0, 1]
   ↓
5. reshape回 [num_envs, 8]
   ↓
6. _calc_step_reward(chunk_rewards)
   → 计算奖励差分
   → 返回给RL循环
```

### 图像值域转换

| 阶段 | 值域 | 说明 |
|------|------|------|
| Wan生成 | [-1, 1] | Tanh归一化 |
| predict_rew输入 | [-1, 1] | 来自current_obs |
| 值域转换后 | [0, 1] | `(obs + 1) / 2` |
| preprocess_images | 标准化 | ImageNet mean/std |
| 最终输入模型 | ~[-2, 2] | ImageNet标准化后 |

---

## 🔍 技术细节

### Checkpoint格式

你的checkpoint结构:
```python
{
    'visual_encoder.0.weight': torch.Size([64, 3, 7, 7]),
    'visual_encoder.1.weight': torch.Size([64]),
    # ... ResNet权重
    't5_encoder.encoder.block.0.layer.0.SelfAttention.q.weight': ...,
    # ... T5权重
    'cross_attn.in_proj_weight': ...,
    'reward_head.0.weight': torch.Size([256, 512]),
    # ... 直接是state_dict
}
```

`from_pretrained`会正确处理这个格式。

### 图像尺寸处理

| 来源 | 尺寸 | 处理 |
|------|------|------|
| Wan输出 | 256x256 | 世界模型配置 |
| Reward期望 | 224x224 | 训练时配置 |
| 自动处理 | ✅ | preprocess_images会resize |

### 值域处理逻辑

```python
# predict_rew中的值域检测
if obs.min() < 0:
    # 检测到[-1, 1]范围，转换到[0, 1]
    obs = (obs + 1.0) / 2.0

# 如果已经是[0, 1]，不做转换
# 然后preprocess_images会统一处理
```

---

## 📁 修改的文件

### 配置文件
1. [examples/embodiment/config/env/wan_robotwin_click_bell.yaml](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/config/env/wan_robotwin_click_bell.yaml)
   - Line 1: `env_type: wan_wm`
   - Line 58-61: 启用reward_model配置

### 模型代码
2. [rlinf/models/embodiment/reward/robotwin_reward_model.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/models/embodiment/reward/robotwin_reward_model.py)
   - Line 333-363: 添加`predict_rew()`方法
   - Line 365-425: 添加`from_pretrained()`类方法

### 环境代码
3. [rlinf/envs/world_model/world_model_wan_env.py](file:///ML-vePFS/protected/tangyinzhou/RLinf/rlinf/envs/world_model/world_model_wan_env.py)
   - Line 148-167: 添加`RoboTwinT5CrossAttn`类型支持

### 文档
4. [.skills/STEP4_COMPLETION_SUMMARY.md](file:///ML-vePFS/protected/tangyinzhou/RLinf/.skills/STEP4_COMPLETION_SUMMARY.md) (本文档)

---

## ✅ 检查清单

- [x] env_type改为`wan_wm` (与SupportedEnvType匹配)
- [x] reward_model配置启用并设置为`RoboTwinT5CrossAttn`
- [x] 添加`predict_rew()`方法 (WanEnv兼容接口)
- [x] 添加`from_pretrained()`方法 (checkpoint加载)
- [x] WanEnv支持新reward类型
- [x] 值域转换处理 ([-1, 1] → [0, 1])
- [x] 图像尺寸处理 (256x256 → 224x224)
- [x] ImageNet normalization (通过preprocess_images)
- [x] 返回格式正确 ([B] tensor)

---

## 🚀 下一步

所有代码修改已完成！现在可以:

1. **测试Reward模型加载**:
```python
from rlinf.models.embodiment.reward import RoboTwinT5CrossAttnRewardModel

model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
    '/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth'
)
print("✓ Reward模型加载成功")
```

2. **测试WanEnv初始化** (需要完整配置)

3. **继续任务5**: 创建训练脚本和运行指南

---

## 💡 注意事项

1. **T5模型路径**: 
   - 如果运行环境没有网络，需要设置本地T5路径
   - 可以通过config覆盖: `config={"t5_model_name": "/path/to/local/t5-base"}`

2. **设备管理**:
   - `predict_rew`内部会调用`compute_reward`
   - `compute_reward`会自动将输入移到模型设备
   - 不需要手动处理device

3. **性能**:
   - T5模型较大，首次加载可能需要一些时间
   - 建议启用`enable_offload: True`节省显存

---

**报告生成时间**: 2026-04-22  
**文档版本**: 1.0  
**状态**: ✅ 完成
