"""
Debug script for Genie World Model PPO training on single GPU.
This script is designed for line-by-line debugging with debugger (e.g., pdb, debugpy).

Usage:
    # Run directly
    python debug_genie_wm.py

    # Run with pdb
    python debug_genie_wm.py

    # Run with debugpy (for VSCode debugging)
    python -m debugpy --listen 5678 --wait-for-client debug_genie_wm.py
"""
import os
import sys

# Disable NCCL for single-GPU debugging
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
# Use basic attention in Genie (avoids xformers/flash_attn CUDA binary issues)
os.environ["XFORMERS_DISABLED"] = "true"
os.environ["RAY_ROOT_DIR"] = "/ML-vePFS/protected/tangyinzhou/tmp/ray"
os.environ["TMPDIR"] = "/ML-vePFS/protected/tangyinzhou/tmp"
# RAY_ADDRESS is intentionally NOT hardcoded here.
# Ray will auto-discover the local cluster, or you can set it externally:
#   export RAY_ADDRESS="<head_ip>:6379"
# Set CUDA visible devices to single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Set paths
REPO_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf"
EMBODIED_PATH = REPO_PATH  # EMBODIED_PATH points to the RLinf repo root

# Genie module lives in roboscape/genie; inject it into PYTHONPATH so that
# both the main process and Ray worker subprocesses can find it.
GENIE_PATH = "/ML-vePFS/protected/tangyinzhou/roboscape/roboscape/genie"
# The genie sub-package itself is one level below GENIE_PATH
GENIE_MODULE_PATH = os.path.join(GENIE_PATH, "genie")

# Add to Python path (main process only)
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, GENIE_PATH)  # enables: from genie.config import ...
                                 #           from magvit2.config import ...

os.environ["REPO_PATH"] = REPO_PATH
os.environ["EMBODIED_PATH"] = EMBODIED_PATH

# Propagate to Ray worker subprocesses
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = (
    f"{GENIE_PATH}:{REPO_PATH}:{existing_pythonpath}"
    if existing_pythonpath
    else f"{GENIE_PATH}:{REPO_PATH}"
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
CONFIG_NAME = "genie_wm_adjust_bottle_ppo_openpi_pi05_debug"


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
        distributed_log_dir=cfg.runner.per_worker_log_path,
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
    # Uncomment to start debugging from the beginning:
    # import pdb; pdb.set_trace()

    main()
