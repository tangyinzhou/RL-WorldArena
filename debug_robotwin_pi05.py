"""
Debug script for Pi05 RobotWin PPO training on single GPU.
This script is designed for line-by-line debugging with debugger (e.g., pdb, debugpy).

Usage:
    # Run with pdb
    python debug_robotwin_pi05.py
    
    # Run with debugpy (for VSCode debugging)
    python -m debugpy --listen 5678 --wait-for-client debug_robotwin_pi05.py
    
    # Run directly
    python debug_robotwin_pi05.py
"""
import os
import sys

# Disable NCCL for single-GPU debugging
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["RAY_ROOT_DIR"] = "/ML-vePFS/protected/tangyinzhou/tmp/ray"
os.environ["TMPDIR"] = "/ML-vePFS/protected/tangyinzhou/tmp"
# Set CUDA visible devices to single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set environment variables required by RobotWin
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Set paths
REPO_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf"
ROBOTWIN_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf/RoboTwin"
EMBODIED_PATH = REPO_PATH  # EMBODIED_PATH points to the RLinf repo root

# Add to Python path
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, ROBOTWIN_PATH)
os.environ["REPO_PATH"] = REPO_PATH
os.environ["ROBOTWIN_PATH"] = ROBOTWIN_PATH
os.environ["EMBODIED_PATH"] = EMBODIED_PATH
# Set PYTHONPATH so Ray worker subprocesses can also find robotwin module
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
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor

import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)


# Hydra configuration path
CONFIG_PATH = os.path.join(REPO_PATH, "examples/embodiment/config")
CONFIG_NAME = "robotwin_adjust_bottle_ppo_openpi_pi05_debug"  # Use debug config


@hydra.main(version_base="1.1", config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg):
    """Main training function for debugging."""
    
    # Validate configuration
    cfg = validate_cfg(cfg)
    
    # Print resolved configuration for debugging
    print("=" * 80)
    print("Configuration:")
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    print("=" * 80)
    
    # Create cluster (single node for debugging)
    cluster = Cluster(
        cluster_cfg=cfg.cluster, 
        distributed_log_dir=cfg.runner.per_worker_log_path
    )
    
    # Component placement
    component_placement = HybridComponentPlacement(cfg, cluster)
    
    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    
    # Create reward worker group (if needed)
    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )
    
    # Create runner
    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )
    
    # Initialize workers and run training loop
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    # You can add pdb here to start debugging from the beginning
    # import pdb; pdb.set_trace()
    
    main()
