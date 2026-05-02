"""
Inference script for RoboTwin Reward Model on PNG images.

Loads a trained reward model and runs inference on PNG image files,
outputting predicted reward values.

Usage:
    # Single image
    python infer_robotwin_reward_from_png.py \
        --image /path/to/image.png \
        --instruction "click the bell"

    # Multiple images in a directory
    python infer_robotwin_reward_from_png.py \
        --image-dir /path/to/images/ \
        --instruction "click the bell"

    # Batch inference with different instructions
    python infer_robotwin_reward_from_png.py \
        --image-dir /path/to/images/ \
        --instruction-file instructions.txt \
        --batch-size 16

    # Save results with visualizations
    python infer_robotwin_reward_from_png.py \
        --image-dir /path/to/images/ \
        --instruction "click the bell" \
        --output-dir logs/reward_png_output \
        --save-vis
"""
import argparse
import os
import sys
from pathlib import Path

REPO_PATH = "/ML-vePFS/protected/tangyinzhou/RLinf"
sys.path.insert(0, REPO_PATH)
os.environ.setdefault("REPO_PATH", REPO_PATH)

import torch
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm

from rlinf.models.embodiment.reward.robotwin_reward_model import RoboTwinT5CrossAttnRewardModel


def parse_args():
    parser = argparse.ArgumentParser(description="RoboTwin Reward Model Inference from PNG")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/ML-vePFS/protected/tangyinzhou/RLinf/diffsynth-studio/RLinf-Wan-RobotWin-ClickBell/resnet_rm.pth",
        help="Path to trained reward model checkpoint (.pth)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single PNG image",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing PNG images",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Task instruction (used for all images if not using --instruction-file)",
    )
    parser.add_argument(
        "--instruction-file",
        type=str,
        default=None,
        help="Text file with one instruction per line (must match image count)",
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
        default="logs/reward_png_output",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save images with reward overlay",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size to resize to [H, W]",
    )
    return parser.parse_args()


def load_image(image_path: str, target_size: tuple = (256, 256)) -> torch.Tensor:
    """Load and preprocess a single PNG image.
    
    Returns:
        Tensor of shape [C, H, W] in [0, 1] range
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0  # [H, W, C] in [0, 1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # [C, H, W]
    return img_tensor


def load_images_from_directory(image_dir: str, target_size: tuple = (256, 256)):
    """Load all PNG images from a directory.
    
    Returns:
        List of (image_path, image_tensor) tuples
    """
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    if len(image_paths) == 0:
        raise ValueError(f"No PNG/JPG images found in {image_dir}")
    
    images = []
    for img_path in image_paths:
        try:
            img_tensor = load_image(str(img_path), target_size)
            images.append((str(img_path), img_tensor))
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")
    
    return images


def draw_reward_on_image(image: np.ndarray, reward: float, instruction: str = "") -> np.ndarray:
    """Draw reward value and instruction on image.
    
    Args:
        image: numpy array [H, W, C] in [0, 255]
        reward: predicted reward value
        instruction: task instruction
    
    Returns:
        Image with overlay
    """
    try:
        from rlinf.envs.utils import put_info_on_image
        
        # Create info dict for overlay
        color = "green" if reward >= 0.5 else "red"
        info = {
            "reward": f"{reward:.4f}",
            "prediction": "success" if reward >= 0.5 else "fail",
            "color": color,
        }
        if instruction:
            info["instruction"] = instruction[:50]
        
        # Use put_info_on_image to draw text
        return put_info_on_image(image, info)
    except Exception as e:
        # Fallback: just return original image
        print(f"Warning: Failed to draw overlay: {e}")
        return image


def main():
    args = parse_args()
    
    # Validate inputs
    if args.image is None and args.image_dir is None:
        raise ValueError("Either --image or --image-dir must be specified")
    
    if args.instruction is None and args.instruction_file is None:
        raise ValueError("Either --instruction or --instruction-file must be specified")
    
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
    # 2. Load images
    # ------------------------------------------------------------------
    if args.image is not None:
        # Single image mode
        print(f"Loading image: {args.image}")
        img_tensor = load_image(args.image, tuple(args.image_size))
        images = [(args.image, img_tensor)]
    else:
        # Directory mode
        print(f"Loading images from: {args.image_dir}")
        images = load_images_from_directory(args.image_dir, tuple(args.image_size))
    
    total = len(images)
    print(f"✓ Loaded {total} images")
    
    # Load instructions
    if args.instruction_file is not None:
        with open(args.instruction_file, "r") as f:
            instructions = [line.strip() for line in f.readlines()]
        if len(instructions) != total:
            raise ValueError(
                f"Instruction file has {len(instructions)} lines but found {total} images"
            )
    else:
        instructions = [args.instruction] * total
    
    # ------------------------------------------------------------------
    # 3. Run inference
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_rewards = []
    all_paths = []
    all_instructions = []
    
    print(f"\nRunning inference (batch_size={args.batch_size})...")
    with torch.no_grad():
        for start in tqdm(range(0, total, args.batch_size), desc="Inference"):
            end = min(start + args.batch_size, total)
            batch_images = []
            batch_paths = []
            batch_instructions = []
            
            for i in range(start, end):
                img_path, img_tensor = images[i]
                batch_images.append(img_tensor)
                batch_paths.append(img_path)
                batch_instructions.append(instructions[i])
            
            # Stack images: [B, C, H, W]
            # Images are already [0, 1] float32, matching training distribution
            images_tensor = torch.stack(batch_images, dim=0).to(device)
            
            # Compute rewards (single ImageNet normalization via compute_reward)
            rewards = model.compute_reward(
                images_tensor, task_descriptions=batch_instructions
            )
            rewards = rewards.cpu().numpy()
            
            # Accumulate results
            for i, (reward, img_path, instr) in enumerate(zip(
                rewards, batch_paths, batch_instructions
            )):
                all_rewards.append(float(reward))
                all_paths.append(img_path)
                all_instructions.append(instr)
            
            # Save visualizations (optional)
            if args.save_vis:
                for i, (reward, img_path, instr) in enumerate(zip(
                    rewards, batch_paths, batch_instructions
                )):
                    # Convert tensor to numpy [H, W, C] in [0, 255]
                    img_np = (images_tensor[i].cpu().numpy() * 255.0).transpose(1, 2, 0)
                    img_np = img_np.astype(np.uint8)
                    
                    # Draw reward on image
                    img_with_reward = draw_reward_on_image(img_np, reward, instr)
                    
                    # Save with reward in filename
                    img_name = Path(img_path).stem
                    reward_str = f"r{reward:.3f}"
                    pred_str = "succ" if reward >= 0.5 else "fail"
                    output_path = os.path.join(
                        args.output_dir,
                        f"{img_name}_{reward_str}_{pred_str}.png"
                    )
                    save_image(
                        torch.from_numpy(img_with_reward).permute(2, 0, 1).float() / 255.0,
                        output_path
                    )
    
    # ------------------------------------------------------------------
    # 4. Print summary statistics
    # ------------------------------------------------------------------
    all_rewards = np.array(all_rewards)
    
    print("\n" + "=" * 70)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total images:         {total}")
    print(f"  Mean reward:          {all_rewards.mean():.4f}")
    print(f"  Std reward:           {all_rewards.std():.4f}")
    print(f"  Min reward:           {all_rewards.min():.4f}")
    print(f"  Max reward:           {all_rewards.max():.4f}")
    print()
    
    # Success/fail statistics
    success_mask = all_rewards >= 0.5
    num_success = success_mask.sum()
    num_fail = total - num_success
    
    print(f"  Success (r >= 0.5):   {num_success}/{total} ({num_success/total*100:.1f}%)")
    print(f"  Fail (r < 0.5):       {num_fail}/{total} ({num_fail/total*100:.1f}%)")
    print()
    
    # Print per-image results
    print("PER-IMAGE RESULTS:")
    print("-" * 70)
    for i, (reward, img_path, instr) in enumerate(zip(all_rewards, all_paths, all_instructions)):
        pred = "success" if reward >= 0.5 else "fail"
        print(f"  [{i:3d}] r={reward:.4f} ({pred:7s}) | {Path(img_path).name} | {instr[:50]}")
    
    print("=" * 70)
    
    # ------------------------------------------------------------------
    # 5. Save detailed results to file
    # ------------------------------------------------------------------
    results_path = os.path.join(args.output_dir, "reward_results.txt")
    with open(results_path, "w") as f:
        f.write("idx\treward\tprediction\timage_path\tinstruction\n")
        for i, (reward, img_path, instr) in enumerate(zip(all_rewards, all_paths, all_instructions)):
            pred = "success" if reward >= 0.5 else "fail"
            f.write(f"{i}\t{reward:.6f}\t{pred}\t{img_path}\t{instr}\n")
    
    np.savez(
        os.path.join(args.output_dir, "reward_results.npz"),
        rewards=all_rewards,
        paths=np.array(all_paths),
        instructions=np.array(all_instructions),
    )
    
    print(f"\n✓ Detailed results saved to: {results_path}")
    print(f"✓ Numpy arrays saved to: {os.path.join(args.output_dir, 'reward_results.npz')}")
    if args.save_vis:
        print(f"✓ Visualizations saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
