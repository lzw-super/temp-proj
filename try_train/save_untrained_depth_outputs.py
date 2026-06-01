"""
Save depth outputs from an initialized, untrained HeadOnlyModel.

This script mirrors the depth visualization style in validate_model_v2.py, but
it deliberately does not load a trained checkpoint. It is useful for saving a
fixed "before training" baseline to compare against trained depth outputs.

Examples:
  python try_train/save_untrained_depth_outputs.py \
    --data_root /path/to/Replica/room0 \
    --output_dir try_train/untrained_depth_outputs

  # Reuse architecture settings from a trained checkpoint, without loading
  # the model weights from that checkpoint.
  python try_train/save_untrained_depth_outputs.py \
    --data_root /path/to/Replica/room0 \
    --config_checkpoint try_train/checkpoints/v2_fixed/checkpoint_final.pt \
    --output_dir try_train/untrained_depth_outputs_v2_fixed
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from replica_dataset import ReplicaDataset
from head_only_model import HeadOnlyModel


BACKBONE_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_model_config(args) -> Dict[str, object]:
    """Resolve architecture config without loading trained model weights."""
    config = {
        "backbone": "dinov2_vits14",
        "img_size": 224,
        "freeze_backbone": True,
        "train_pose": True,
        "num_views": 2,
    }

    if args.config_checkpoint is not None:
        ckpt = torch.load(args.config_checkpoint, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args", {})
        config.update({
            "backbone": ckpt_args.get("backbone", config["backbone"]),
            "img_size": ckpt_args.get("img_size", config["img_size"]),
            "freeze_backbone": ckpt_args.get("freeze_backbone", config["freeze_backbone"]),
            "train_pose": ckpt_args.get("train_pose", config["train_pose"]),
            "num_views": ckpt_args.get("min_views", ckpt_args.get("num_views", config["num_views"])),
        })
        print(f"[config] Loaded architecture settings from {args.config_checkpoint}")

    if args.backbone is not None:
        config["backbone"] = args.backbone
    if args.img_size is not None:
        config["img_size"] = args.img_size
    if args.freeze_backbone is not None:
        config["freeze_backbone"] = args.freeze_backbone
    if args.train_pose is not None:
        config["train_pose"] = args.train_pose
    if args.num_views is not None:
        config["num_views"] = args.num_views

    return config


def build_untrained_model(config: Dict[str, object], device: str) -> HeadOnlyModel:
    backbone = str(config["backbone"])
    if backbone not in BACKBONE_EMBED_DIMS:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = HeadOnlyModel(
        backbone_name=backbone,
        freeze_backbone=bool(config["freeze_backbone"]),
        img_size=int(config["img_size"]),
        patch_size=14,
        embed_dim=BACKBONE_EMBED_DIMS[backbone],
        train_depth_head=True,
        train_pose_head=bool(config["train_pose"]),
        num_views=int(config["num_views"]),
    )
    model = model.to(device).eval()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[model] Initialized model without trained checkpoint weights")
    print(f"  - backbone: {backbone}")
    print(f"  - img_size: {config['img_size']}, num_views: {config['num_views']}")
    print(f"  - total params: {total:,}, trainable params: {trainable:,}")
    return model


def create_eval_dataset(data_root: str, config: Dict[str, object], seed: int) -> ReplicaDataset:
    """Create a deterministic, augmentation-free Replica dataset."""
    return ReplicaDataset(
        data_root=data_root,
        num_views=int(config["num_views"]),
        max_dim=int(config["img_size"]),
        temporal_window=30,
        shuffle_view_order=False,
        sampler_type="temporal_nearby",
        color_jitter_prob=0.0,
        grayscale_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
        co_jitter=True,
        seed=seed,
    )


def squeeze_depth(depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5:
        depth = depth.squeeze(-1)
    return depth


def resize_to_shape(depth: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if depth.shape == target_shape:
        return depth
    return cv2.resize(depth, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def compute_depth_stats(depth_pred: np.ndarray, depth_gt: np.ndarray, valid_mask: np.ndarray) -> Dict[str, float]:
    valid = valid_mask & np.isfinite(depth_pred) & np.isfinite(depth_gt) & (depth_gt > 0)
    if valid.sum() == 0:
        return {
            "pred_min": float("nan"),
            "pred_max": float("nan"),
            "pred_mean": float("nan"),
            "gt_min": float("nan"),
            "gt_max": float("nan"),
            "gt_mean": float("nan"),
            "mean_abs_error": float("nan"),
            "mean_rel_error": float("nan"),
            "valid_pixels": 0,
        }

    pred_valid = depth_pred[valid]
    gt_valid = depth_gt[valid]
    abs_error = np.abs(pred_valid - gt_valid)
    rel_error = abs_error / (gt_valid + 1e-3)

    return {
        "pred_min": float(pred_valid.min()),
        "pred_max": float(pred_valid.max()),
        "pred_mean": float(pred_valid.mean()),
        "gt_min": float(gt_valid.min()),
        "gt_max": float(gt_valid.max()),
        "gt_mean": float(gt_valid.mean()),
        "mean_abs_error": float(abs_error.mean()),
        "mean_rel_error": float(rel_error.mean()),
        "valid_pixels": int(valid.sum()),
    }


def masked_for_vis(array: np.ndarray, valid_mask: np.ndarray):
    safe = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ma.masked_where(~valid_mask, safe)


def save_sample_visualization(
    sample_idx: int,
    frame_ids,
    images: np.ndarray,
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_masks: np.ndarray,
    stats,
    vis_dir: Path,
    vmax: Optional[float],
    error_vmax: float,
):
    num_views = depth_pred.shape[0]
    fig, axes = plt.subplots(4, num_views, figsize=(4.5 * num_views, 14), squeeze=False)

    gt_valid_values = depth_gt[valid_masks]
    if vmax is None:
        depth_vmax = max(float(gt_valid_values.max()) if gt_valid_values.size else 0.0, 10.0)
    else:
        depth_vmax = vmax

    for v in range(num_views):
        rgb = np.clip(images[v].transpose(1, 2, 0), 0.0, 1.0)
        gt_v = depth_gt[v]
        pred_v = depth_pred[v]
        mask_v = valid_masks[v]
        err_v = np.abs(np.nan_to_num(pred_v, nan=0.0, posinf=0.0, neginf=0.0) - gt_v)

        axes[0, v].imshow(rgb)
        axes[0, v].set_title(f"RGB view {v}\nframe {frame_ids[v]}")
        axes[0, v].axis("off")

        im_gt = axes[1, v].imshow(masked_for_vis(gt_v, mask_v), cmap="gray", vmin=0, vmax=depth_vmax)
        axes[1, v].set_title(f"GT Depth\nrange {stats[v]['gt_min']:.3f}-{stats[v]['gt_max']:.3f}m")
        axes[1, v].axis("off")
        plt.colorbar(im_gt, ax=axes[1, v], fraction=0.046, pad=0.04)

        im_pred = axes[2, v].imshow(masked_for_vis(pred_v, mask_v), cmap="gray", vmin=0, vmax=depth_vmax)
        axes[2, v].set_title(
            f"Untrained Depth\nrel_err {stats[v]['mean_rel_error']:.3f}"
        )
        axes[2, v].axis("off")
        plt.colorbar(im_pred, ax=axes[2, v], fraction=0.046, pad=0.04)

        im_err = axes[3, v].imshow(masked_for_vis(err_v, mask_v), cmap="hot", vmin=0, vmax=error_vmax)
        axes[3, v].set_title(f"Abs Error\nmean {stats[v]['mean_abs_error']:.3f}m")
        axes[3, v].axis("off")
        plt.colorbar(im_err, ax=axes[3, v], fraction=0.046, pad=0.04)

    plt.suptitle(f"Initialized Untrained Depth Outputs - Sample {sample_idx}", fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / f"untrained_depth_sample_{sample_idx:03d}.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[vis] Saved {save_path}")
    return save_path


def save_raw_outputs(
    raw_dir: Path,
    sample_idx: int,
    frame_ids,
    images: np.ndarray,
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    valid_masks: np.ndarray,
):
    sample_dir = raw_dir / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(sample_dir / "images.npy", images)
    np.save(sample_dir / "depth_untrained.npy", depth_pred)
    np.save(sample_dir / "depth_gt.npy", depth_gt)
    np.save(sample_dir / "valid_masks.npy", valid_masks)
    with open(sample_dir / "frame_ids.json", "w") as f:
        json.dump([int(fid) for fid in frame_ids], f, indent=2)


def run(args):
    set_seed(args.seed)
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print(f"[device] Requested {device}, but CUDA is unavailable. Falling back to CPU.")
        device = "cpu"

    output_dir = Path(args.output_dir)
    vis_dir = output_dir / "vis"
    raw_dir = output_dir / "raw"
    vis_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    config = resolve_model_config(args)
    with open(output_dir / "model_config.json", "w") as f:
        json.dump(config, f, indent=2)

    model = build_untrained_model(config, device)
    dataset = create_eval_dataset(args.data_root, config, args.seed)

    results = []

    use_cuda_amp = args.use_amp and device.startswith("cuda") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16
    if use_cuda_amp and torch.cuda.get_device_capability()[0] < 8:
        amp_dtype = torch.float16

    with torch.no_grad():
        for sample_idx in range(min(args.num_samples, len(dataset))):
            sample = dataset[sample_idx]
            images = sample["images"].unsqueeze(0).to(device)
            depths_gt = sample["depths"].cpu().numpy()
            valid_masks = sample["valid_masks"].cpu().numpy().astype(bool)
            frame_ids = sample["frame_ids"]

            if use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    predictions = model(images)
            else:
                predictions = model(images)

            depth_pred = squeeze_depth(predictions["depth"])[0].detach().cpu().float().numpy()
            depth_pred = np.stack([
                resize_to_shape(depth_pred[v], depths_gt[v].shape)
                for v in range(depth_pred.shape[0])
            ], axis=0)

            view_stats = []
            for view_idx in range(depth_pred.shape[0]):
                stats = compute_depth_stats(depth_pred[view_idx], depths_gt[view_idx], valid_masks[view_idx])
                stats.update({
                    "sample_idx": sample_idx,
                    "view_idx": view_idx,
                    "frame_id": int(frame_ids[view_idx]),
                })
                view_stats.append(stats)
                results.append(stats)
                print(
                    f"[sample {sample_idx} view {view_idx}] "
                    f"frame={frame_ids[view_idx]} "
                    f"pred=[{stats['pred_min']:.3f}, {stats['pred_max']:.3f}] "
                    f"gt=[{stats['gt_min']:.3f}, {stats['gt_max']:.3f}] "
                    f"rel_err={stats['mean_rel_error']:.3f}"
                )

            images_np = sample["images"].cpu().numpy()
            vis_path = save_sample_visualization(
                sample_idx=sample_idx,
                frame_ids=frame_ids,
                images=images_np,
                depth_pred=depth_pred,
                depth_gt=depths_gt,
                valid_masks=valid_masks,
                stats=view_stats,
                vis_dir=vis_dir,
                vmax=args.vmax,
                error_vmax=args.error_vmax,
            )
            for stats in view_stats:
                stats["visualization"] = str(vis_path)

            if not args.no_save_raw:
                save_raw_outputs(raw_dir, sample_idx, frame_ids, images_np, depth_pred, depths_gt, valid_masks)

    summary = {
        "config": config,
        "num_samples": min(args.num_samples, len(dataset)),
        "mean_rel_error": float(np.nanmean([r["mean_rel_error"] for r in results])) if results else float("nan"),
        "mean_abs_error": float(np.nanmean([r["mean_abs_error"] for r in results])) if results else float("nan"),
        "results": results,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[summary] Saved {output_dir / 'summary.json'}")
    print(f"[done] Outputs saved under {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Save initialized untrained HeadOnlyModel depth outputs for comparison."
    )
    parser.add_argument("--data_root", type=str, required=True, help="Replica data root directory")
    parser.add_argument("--output_dir", type=str, default="try_train/untrained_depth_outputs")
    parser.add_argument(
        "--config_checkpoint",
        type=str,
        default=None,
        help="Optional trained checkpoint to read architecture args from; weights are not loaded.",
    )

    parser.add_argument("--backbone", type=str, default=None, choices=list(BACKBONE_EMBED_DIMS.keys()))
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--num_views", type=int, default=None)
    parser.add_argument("--freeze_backbone", dest="freeze_backbone", action="store_true", default=None)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--train_pose", dest="train_pose", action="store_true", default=None)
    parser.add_argument("--no_train_pose", dest="train_pose", action="store_false")

    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--vmax", type=float, default=None, help="Depth visualization vmax. Default: max(GT max, 10m).")
    parser.add_argument("--error_vmax", type=float, default=2.0)
    parser.add_argument("--no_save_raw", action="store_true", help="Only save PNG visualizations and summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
