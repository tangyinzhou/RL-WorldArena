"""
Inference script for RoboTwin Reward Model.

Loads a trained reward model checkpoint and runs inference on validation data,
outputting images with predicted reward values overlaid.

Usage:
    # Default: run on validation set with default checkpoint
    python infer_robotwin_reward_model.py

    # Specify checkpoint and data path
    python infer_robotwin_reward_model.py \
        --checkpoint /path/to/resnet_rm.pth \
        --data-path logs/robotwin_reward_data_click_bell/val.pt

    # Limit number of samples and select output dir
    python infer_robotwin_reward_model.py \
        --max-samples 100 \
        --output-dir logs/reward_inference_output

    # Run on training set
    python infer_robotwin_reward_model.py \
        --data-path logs/robotwin_reward_data_click_bell/train.pt
"""
import argparse
import os
import sys

REPO_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf"
sys.path.insert(0, REPO_PATH)
os.environ.setdefault("REPO_PATH", REPO_PATH)

import torch
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

from rlinf.models.embodiment.reward.robotwin_reward_model import RoboTwinT5CrossAttnRewardModel
from rlinf.data.datasets.reward_model import TextCondRewardBinaryDataset


def parse_args():
    parser = argparse.ArgumentParser(description="RoboTwin Reward Model Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth",
        help="Path to trained reward model checkpoint (.pth)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="logs/robotwin_reward_data_click_bell/val.pt",
        help="Path to .pt dataset file (val.pt or train.pt)",
    )
    parser.add_argument(
        "--t5-model-name",
        type=str,
        default="/ML-vePFS/protected/tangyinzhou/RLinf/pretrained_models/t5-base",
        help="Path to T5 model (local or HuggingFace id)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs/reward_inference_output",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum number of samples to run inference on (0 = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save individual images with reward overlay (slow, large output)",
    )
    parser.add_argument(
        "--grid-mode",
        action="store_true",
        help="Save a grid image per batch instead of individual images",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print(f"Loading model from: {args.checkpoint}")
    config_overrides = {
        "t5_model_name": args.t5_model_name,
    }
    model = RoboTwinT5CrossAttnRewardModel.from_pretrained(
        args.checkpoint, config=config_overrides
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")

    # ------------------------------------------------------------------
    # 2. Load dataset
    # ------------------------------------------------------------------
    print(f"Loading dataset from: {args.data_path}")
    dataset = TextCondRewardBinaryDataset(args.data_path)
    total = len(dataset)
    if args.max_samples > 0:
        total = min(total, args.max_samples)
    print(f"  Total samples in dataset: {len(dataset)}, running on: {total}")

    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    all_rewards = []
    all_labels = []
    all_instructions = []

    # Stats accumulators
    correct = 0
    total_processed = 0
    tp = 0  # true positive: label=1, pred>0.5
    fp = 0  # false positive: label=0, pred>0.5
    tn = 0  # true negative: label=0, pred<=0.5
    fn = 0  # false negative: label=1, pred<=0.5

    print(f"\nRunning inference (batch_size={args.batch_size})...")
    with torch.no_grad():
        for start in tqdm(range(0, total, args.batch_size), desc="Inference"):
            end = min(start + args.batch_size, total)
            batch_images = []
            batch_instructions = []
            batch_labels = []

            for i in range(start, end):
                img, instr, label = dataset[i]
                batch_images.append(img)
                batch_instructions.append(instr)
                batch_labels.append(label.item() if isinstance(label, torch.Tensor) else label)

            # Stack images: list of [C, H, W] → [B, C, H, W]
            # Dataset images are [0, 1] float32, matching the training distribution.
            # compute_reward() internally calls _encode_visual() → preprocess_images()
            # which applies resize + ImageNet normalisation once.
            images_tensor = torch.stack(batch_images, dim=0).to(device)
            rewards = model.compute_reward(images_tensor, task_descriptions=batch_instructions)
            rewards = rewards.cpu().numpy()

            # Accumulate stats
            for i, (reward, label, instr) in enumerate(zip(rewards, batch_labels, batch_instructions)):
                pred = 1 if reward >= 0.5 else 0
                if label == 1 and pred == 1:
                    tp += 1
                elif label == 0 and pred == 1:
                    fp += 1
                elif label == 0 and pred == 0:
                    tn += 1
                elif label == 1 and pred == 0:
                    fn += 1

                if pred == label:
                    correct += 1
                total_processed += 1

                all_rewards.append(float(reward))
                all_labels.append(int(label))
                all_instructions.append(instr)

            # Save images (optional)
            if args.save_images or args.grid_mode:
                vis_images = images_tensor.cpu().clone().clamp(0, 1)

                if args.grid_mode:
                    from torchvision.utils import make_grid
                    grid = make_grid(vis_images, nrow=8, padding=2)
                    grid_path = os.path.join(args.output_dir, f"grid_{start:06d}.png")
                    save_image(grid, grid_path)

                    ann_path = os.path.join(args.output_dir, f"grid_{start:06d}.txt")
                    with open(ann_path, "w") as f:
                        for idx, (r, l, instr) in enumerate(zip(
                            rewards, batch_labels, batch_instructions
                        )):
                            mark = "✓" if (r >= 0.5) == (l == 1) else "✗"
                            f.write(
                                f"[{idx}] reward={r:.4f} label={l} "
                                f"pred={'succ' if r >= 0.5 else 'fail'} "
                                f"{mark} | {instr[:50]}\n"
                            )
                elif args.save_images:
                    for idx in range(end - start):
                        global_idx = start + idx
                        r = rewards[idx]
                        l = batch_labels[idx]
                        instr = batch_instructions[idx][:30].replace(" ", "_")

                        img_dir = os.path.join(
                            args.output_dir,
                            f"label_{l}",
                            f"pred_{'succ' if r >= 0.5 else 'fail'}",
                        )
                        os.makedirs(img_dir, exist_ok=True)
                        img_path = os.path.join(img_dir, f"{global_idx:06d}_r{r:.3f}_{instr}.png")
                        save_image(vis_images[idx], img_path)

    # ------------------------------------------------------------------
    # 4. Print summary statistics
    # ------------------------------------------------------------------
    all_rewards = np.array(all_rewards)
    all_labels = np.array(all_labels)

    accuracy = correct / total_processed if total_processed > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "=" * 70)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total samples:        {total_processed}")
    print(f"  Accuracy:             {accuracy:.4f} ({correct}/{total_processed})")
    print(f"  Precision:            {precision:.4f} (TP={tp}, FP={fp})")
    print(f"  Recall:               {recall:.4f} (TP={tp}, FN={fn})")
    print(f"  Confusion:            TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print()

    pos_rewards = all_rewards[all_labels == 1]
    neg_rewards = all_rewards[all_labels == 0]

    print(f"  Label=1 (success):    N={len(pos_rewards)}, "
          f"mean={pos_rewards.mean():.4f}, std={pos_rewards.std():.4f}, "
          f"min={pos_rewards.min():.4f}, max={pos_rewards.max():.4f}")
    print(f"  Label=0 (fail):       N={len(neg_rewards)}, "
          f"mean={neg_rewards.mean():.4f}, std={neg_rewards.std():.4f}, "
          f"min={neg_rewards.min():.4f}, max={neg_rewards.max():.4f}")
    print()

    print("  Reward distribution (label=1):")
    _print_histogram(pos_rewards)

    print("\n  Reward distribution (label=0):")
    _print_histogram(neg_rewards)

    # ------------------------------------------------------------------
    # 5. Save detailed results to file
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "inference_results.txt")
    with open(results_path, "w") as f:
        f.write("idx\treward\tlabel\tpred\tcorrect\tinstruction\n")
        for i, (r, l, instr) in enumerate(zip(all_rewards, all_labels, all_instructions)):
            pred = 1 if r >= 0.5 else 0
            f.write(f"{i}\t{r:.6f}\t{l}\t{pred}\t{int(pred == l)}\t{instr}\n")

    np.savez(
        os.path.join(args.output_dir, "inference_results.npz"),
        rewards=all_rewards,
        labels=all_labels,
        instructions=np.array(all_instructions),
    )

    print(f"\n✓ Detailed results saved to: {results_path}")
    print(f"✓ Numpy arrays saved to: {os.path.join(args.output_dir, 'inference_results.npz')}")
    if args.save_images or args.grid_mode:
        print(f"✓ Images saved to: {args.output_dir}")
    print("=" * 70)


def _print_histogram(values: np.ndarray, bins: int = 10, width: int = 50):
    """Print a simple ASCII histogram of reward values."""
    if len(values) == 0:
        print("    (no data)")
        return
    counts, edges = np.histogram(values, bins=bins, range=(0, 1))
    max_count = counts.max() if counts.max() > 0 else 1
    for i in range(bins):
        bar_len = int(counts[i] / max_count * width)
        bar = "█" * bar_len
        print(f"    [{edges[i]:.2f}-{edges[i+1]:.2f}) {counts[i]:>6d} {bar}")


if __name__ == "__main__":
    main()
