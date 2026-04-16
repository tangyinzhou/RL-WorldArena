"""
Debug script for RoboTwin Reward Model training on single GPU.
This script is designed for line-by-line debugging with debugger (e.g., pdb, debugpy).

Usage:
    # Run with pdb
    python debug_robotwin_reward_model.py
    
    # Run with debugpy (for VSCode debugging)
    python -m debugpy --listen 5678 --wait-for-client debug_robotwin_reward_model.py
    
    # Run directly
    python debug_robotwin_reward_model.py
    
    # With custom data paths
    python debug_robotwin_reward_model.py \
        data.train_data_paths=logs/robotwin_reward_data/train.pt \
        data.val_data_paths=logs/robotwin_reward_data/val.pt
    
    # With local T5 model path (for offline environments)
    python debug_robotwin_reward_model.py \
        actor.model.t5_model_name=/path/to/local/t5-base
"""
import os
import sys

# Disable NCCL for single-GPU debugging
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Set Ray temp directory
RAY_ROOT_DIR = "/ML-vePFS/protected/tangyinzhou/tmp/ray"
os.environ["RAY_ROOT_DIR"] = RAY_ROOT_DIR
os.environ["TMPDIR"] = "/ML-vePFS/protected/tangyinzhou/tmp"

# Clear RAY_ADDRESS to force local Ray cluster auto-start
os.environ.pop("RAY_ADDRESS", None)
os.environ.pop("RAY_CLUSTER_ADDRESS", None)

# Remove cached cluster address to prevent auto-connecting to old cluster
RAY_CURRENT_CLUSTER_FILE = os.path.join(RAY_ROOT_DIR, "ray_current_cluster")
if os.path.exists(RAY_CURRENT_CLUSTER_FILE):
    print(f"Removing cached Ray cluster address: {RAY_CURRENT_CLUSTER_FILE}")
    os.remove(RAY_CURRENT_CLUSTER_FILE)

# Kill existing Ray processes to avoid version mismatch
import subprocess
try:
    subprocess.run(["ray", "stop"], capture_output=True, timeout=10)
    print("Stopped existing Ray cluster")
except Exception as e:
    print(f"Note: Could not stop Ray cluster: {e}")

# Set CUDA visible devices to single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set wandb API key (replace with your actual key)
# Get your API key from: https://wandb.ai/settings
os.environ["WANDB_API_KEY"] = "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"

# Set paths
REPO_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf"
ROBOTWIN_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf/RoboTwin"

# Add to Python path
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, ROBOTWIN_PATH)
os.environ["REPO_PATH"] = REPO_PATH
os.environ["ROBOTWIN_PATH"] = ROBOTWIN_PATH

# Set PYTHONPATH so Ray worker subprocesses can also find modules
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    f"{ROBOTWIN_PATH}:{REPO_PATH}:{existing_pythonpath}"
    if existing_pythonpath
    else f"{ROBOTWIN_PATH}:{REPO_PATH}"
)

import json
import hydra
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.sft_runner import SFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.reward.reward_worker import FSDPTextCondRewardWorker

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


# Hydra configuration path
CONFIG_PATH = os.path.join(REPO_PATH, "examples/reward/config")
CONFIG_NAME = "robotwin_reward_training"

# Local T5 model path (downloaded from hf-mirror.com)
LOCAL_T5_PATH = os.path.join(REPO_PATH, "pretrained_models", "t5-base")


@hydra.main(version_base="1.1", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    """Main training function for debugging."""
    
    # Override T5 model path to local directory if it exists
    if os.path.exists(LOCAL_T5_PATH):
        print(f"\n✓ Using local T5 model: {LOCAL_T5_PATH}")
        cfg.actor.model.t5_model_name = LOCAL_T5_PATH
    else:
        print(f"\n⚠ Local T5 not found at {LOCAL_T5_PATH}, will download from HuggingFace")
    
    # Validate configuration
    cfg = validate_cfg(cfg)
    
    # Print resolved configuration for debugging
    print("=" * 80)
    print("RoboTwin Reward Model Training Configuration:")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    print("=" * 80)
    
    # Create cluster (single node for debugging)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    
    # Component placement
    component_placement = HybridComponentPlacement(cfg, cluster)
    
    # Create reward worker group (actor in this context is the reward model)
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPTextCondRewardWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    
    # Create runner (SFTRunner drives the training loop)
    runner = SFTRunner(
        cfg=cfg,
        actor=actor_group,
    )
    
    # Initialize workers and run training loop
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    # You can add pdb here to start debugging from the beginning
    # import pdb; pdb.set_trace()
    
    main()
