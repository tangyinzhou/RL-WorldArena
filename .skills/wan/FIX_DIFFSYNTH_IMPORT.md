# 修复diffsynth模块导入问题

## ❌ 错误信息

```
ModuleNotFoundError: No module named 'diffsynth'
```

## 🔍 问题原因

WanEnv需要从diffsynth-studio导入模块：
```python
from diffsynth.models.reward_model import ResnetRewModel, TaskEmbedResnetRewModel
from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline
```

但diffsynth-studio没有安装到Python环境中。

## ✅ 已完成的修复

### 1. 修改run_embodiment.sh

**文件**: [examples/embodiment/run_embodiment.sh](file:///ML-vePFS/protected/tangyinzhou/RLinf/examples/embodiment/run_embodiment.sh#L10-L13)

```bash
# 添加diffsynth-studio到PYTHONPATH
export DIFFSYNTH_PATH="${REPO_PATH}/diffsynth-studio"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${DIFFSYNTH_PATH}:$PYTHONPATH"
```

这样Ray worker子进程也能找到diffsynth模块。

### 2. 安装缺失依赖

```bash
# 已安装seaborn (diffsynth的依赖)
cd /ML-vePFS/protected/tangyinzhou/RLinf/.venv_wan_openpi
pip install seaborn
```

## 🚀 现在可以重新运行

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf

# 重新运行训练
bash examples/embodiment/run_embodiment.sh wan_robotwin_click_bell_grpo_openpi_pi05
```

## 📋 如果还有其他缺失依赖

如果运行时还报其他ModuleNotFoundError，安装对应的包：

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf/.venv_wan_openpi

# 安装diffsynth的所有依赖
pip install -r /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/requirements.txt

# 或者单独安装
pip install <missing_package>
```

## 🔧 备选方案

如果PYTHONPATH方案仍有问题，可以考虑：

### 方案A: 安装diffsynth-studio为可编辑包

```bash
cd /ML-vePFS/protected/tangyinzhou/RLinf/.venv_wan_openpi
pip install -e /ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/
```

### 方案B: 使用.venv_wan_openpi环境

检查是否有专门的Wan环境：
```bash
ls -la /ML-vePFS/protected/tangyinzhou/RLinf/.venv_wan*/
```

---

**修复时间**: 2026-04-22  
**状态**: ✅ 已修复
