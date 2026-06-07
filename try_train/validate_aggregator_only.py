"""Validate original, random-init, and trained GCT aggregator outputs."""

import argparse
import gc
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from replica_dataset import ReplicaDataset
from lingbot_map.utils.load_fn import load_and_preprocess_images
from train_aggregator_only import (
    align_depth_to_gt,
    forward_aggregator_heads,
    load_depth_gct_model,
    random_init_selected_blocks,
)


def autocast_context(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def median_align(pred, gt, mask):
    aligned = pred.clone()
    for batch_idx in range(pred.shape[0]):
        for view_idx in range(pred.shape[1]):
            valid = mask[batch_idx, view_idx].bool()
            if valid.sum() < 10:
                continue
            pred_median = pred[batch_idx, view_idx][valid].median().clamp(min=1e-6)
            gt_median = gt[batch_idx, view_idx][valid].median()
            aligned[batch_idx, view_idx] = pred[batch_idx, view_idx] * (
                gt_median / pred_median
            )
    return aligned


def scale_align_pose_translation(pose_enc, pose_gt):
    """Anchor at frame 0 and fit one positive translation scale per batch."""
    aligned = pose_enc.clone()
    pred_center = pose_enc[:, :, :3] - pose_enc[:, :1, :3]
    gt_center = pose_gt[:, :, :3, 3]

    for batch_idx in range(pose_enc.shape[0]):
        pred = pred_center[batch_idx, 1:].float()
        gt = gt_center[batch_idx, 1:].float()
        denominator = pred.square().sum().clamp(min=1e-8)
        scale = (pred * gt).sum() / denominator
        scale = scale.clamp(min=1e-6, max=1e6)
        aligned[batch_idx, :, :3] = (
            pred_center[batch_idx] * scale.to(pred_center.dtype)
        )
    return aligned


def evaluate_model(model, dataset, data_root, device, num_samples, num_vis, label):
    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()
    metrics = {
        "metric_depth_loss": [],
        "scale_aligned_depth_loss": [],
        "scale_aligned_abs_pose_loss": [],
        "scale_aligned_rel_pose_loss": [],
    }
    visual_predictions = {"metric": [], "aligned": []}
    model.eval()

    for sample_idx in range(min(num_samples, len(dataset))):
        sample = dataset[sample_idx]
        frame_ids = sample["frame_ids"]
        image_paths = [
            str(Path(data_root) / "results" / f"frame{frame_id:06d}.jpg")
            for frame_id in frame_ids
        ]
        images = load_and_preprocess_images(
            image_paths,
            mode="crop",
            image_size=518,
            patch_size=14,
        ).unsqueeze(0).to(device)
        depths = sample["depths"].unsqueeze(0).to(device)
        masks = sample["valid_masks"].unsqueeze(0).to(device)
        poses = sample["poses"].unsqueeze(0).to(device)

        with torch.no_grad(), autocast_context(device):
            predictions = forward_aggregator_heads(model, images)
        depth_pred, masks = align_depth_to_gt(predictions["depth"], depths, masks)
        depth_pred = depth_pred.float().clamp(min=1e-6)
        depths = depths.float()
        aligned = median_align(depth_pred, depths, masks)
        aligned_pose = scale_align_pose_translation(
            predictions["pose_enc"].float(),
            poses.float(),
        )

        depth_loss = depth_loss_fn(depth_pred, depths, masks)
        aligned_loss = depth_loss_fn(aligned, depths, masks)
        abs_pose_loss = pose_loss_fn(aligned_pose, poses.float())
        rel_pose_loss = rel_pose_loss_fn(aligned_pose, poses.float())

        metrics["metric_depth_loss"].append(float(depth_loss))
        metrics["scale_aligned_depth_loss"].append(float(aligned_loss))
        metrics["scale_aligned_abs_pose_loss"].append(float(abs_pose_loss))
        metrics["scale_aligned_rel_pose_loss"].append(float(rel_pose_loss))

        if sample_idx < num_vis:
            visual_predictions["metric"].append(
                depth_pred[0, 0].detach().cpu().numpy()
            )
            visual_predictions["aligned"].append(
                aligned[0, 0].detach().cpu().numpy()
            )
        print(
            f"  [{label} {sample_idx + 1}/{min(num_samples, len(dataset))}] "
            f"depth={float(depth_loss):.4f} "
            f"depth_aligned={float(aligned_loss):.4f} "
            f"abs_pose_aligned={float(abs_pose_loss):.4f} "
            f"rel_pose_aligned={float(rel_pose_loss):.4f}"
        )

    averaged = {key: float(np.mean(values)) for key, values in metrics.items()}
    return averaged, visual_predictions


def load_trained_blocks(model, checkpoint):
    missing, unexpected = model.load_state_dict(
        checkpoint["aggregator_state_dict"], strict=False
    )
    trained_keys = set(checkpoint["aggregator_state_dict"])
    missing_trained = [key for key in missing if key in trained_keys]
    if missing_trained or unexpected:
        raise RuntimeError(
            f"Trained aggregator state mismatch: missing={missing_trained[:5]}, "
            f"unexpected={unexpected[:5]}"
        )


def collect_gt_visuals(dataset, num_vis):
    visuals = []
    frame_ids = []
    for idx in range(min(num_vis, len(dataset))):
        sample = dataset[idx]
        visuals.append(sample["depths"][0].numpy())
        frame_ids.append(int(sample["frame_ids"][0]))
    return visuals, frame_ids


def plot_depth_comparison(predictions, gt_visuals, frame_ids, output_path, scale_key):
    model_names = ["Original", "Random Init", "Trained", "Ground Truth"]
    num_rows = len(gt_visuals)
    fig, axes = plt.subplots(
        num_rows,
        len(model_names),
        figsize=(18, 4.2 * num_rows),
        squeeze=False,
    )

    for row, gt in enumerate(gt_visuals):
        valid_gt = gt[np.isfinite(gt) & (gt > 0)]
        vmin = float(np.percentile(valid_gt, 2)) if valid_gt.size else 0.0
        vmax = float(np.percentile(valid_gt, 98)) if valid_gt.size else 1.0
        row_images = [
            predictions["Original"][scale_key][row],
            predictions["Random Init"][scale_key][row],
            predictions["Trained"][scale_key][row],
            gt,
        ]
        for col, (name, depth) in enumerate(zip(model_names, row_images)):
            if depth.shape != gt.shape:
                depth = cv2.resize(depth, (gt.shape[1], gt.shape[0]))
            image = axes[row, col].imshow(
                depth,
                cmap="gray",
                vmin=vmin,
                vmax=vmax,
            )
            axes[row, col].set_title(
                f"{name} | frame {frame_ids[row]}\nGT scale {vmin:.2f}-{vmax:.2f} m"
            )
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    scale_title = "Metric Scale" if scale_key == "metric" else "Median-Aligned Scale"
    fig.suptitle(f"GCT Aggregator-Only Depth Comparison ({scale_title})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(results, output_path):
    metrics = [
        "metric_depth_loss",
        "scale_aligned_depth_loss",
        "scale_aligned_abs_pose_loss",
        "scale_aligned_rel_pose_loss",
    ]
    titles = [
        "Metric Depth Loss",
        "Scale-Aligned Depth Loss",
        "Scale-Aligned Absolute Pose Loss",
        "Scale-Aligned Relative Pose Loss",
    ]
    names = list(results)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = ["#3478bf", "#d68c2f", "#3f995b"]
    for axis, metric, title in zip(axes.flat, metrics, titles):
        values = [results[name][metric] for name in names]
        bars = axis.bar(names, values, color=colors)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_comparison(predictions, gt_visuals, frame_ids, output_path):
    """Show scale-aligned predictions and absolute errors on the same scale."""
    num_rows = len(gt_visuals)
    column_names = [
        "Ground Truth",
        "Random Init",
        "Trained",
        "Random |Error|",
        "Trained |Error|",
    ]
    fig, axes = plt.subplots(
        num_rows,
        len(column_names),
        figsize=(21, 4.2 * num_rows),
        squeeze=False,
    )

    for row, gt in enumerate(gt_visuals):
        random_depth = predictions["Random Init"]["aligned"][row]
        trained_depth = predictions["Trained"]["aligned"][row]
        if random_depth.shape != gt.shape:
            random_depth = cv2.resize(random_depth, (gt.shape[1], gt.shape[0]))
        if trained_depth.shape != gt.shape:
            trained_depth = cv2.resize(trained_depth, (gt.shape[1], gt.shape[0]))

        valid = np.isfinite(gt) & (gt > 0)
        depth_values = gt[valid]
        depth_vmin = float(np.percentile(depth_values, 2)) if depth_values.size else 0.0
        depth_vmax = float(np.percentile(depth_values, 98)) if depth_values.size else 1.0
        random_error = np.abs(random_depth - gt)
        trained_error = np.abs(trained_depth - gt)
        random_error[~valid] = 0
        trained_error[~valid] = 0
        error_values = np.concatenate([random_error[valid], trained_error[valid]])
        error_vmax = (
            float(np.percentile(error_values, 95))
            if error_values.size else 1.0
        )

        panels = [gt, random_depth, trained_depth, random_error, trained_error]
        for col, (name, panel) in enumerate(zip(column_names, panels)):
            is_error = col >= 3
            image = axes[row, col].imshow(
                panel,
                cmap="gray",
                vmin=0 if is_error else depth_vmin,
                vmax=error_vmax if is_error else depth_vmax,
            )
            title = f"{name} | frame {frame_ids[row]}"
            if is_error and valid.any():
                title += f"\nmean={panel[valid].mean():.3f} m"
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
            fig.colorbar(image, ax=axes[row, col], fraction=0.046, pad=0.04)

    fig.suptitle("Random-Init vs Trained Aggregator: Scale-Aligned Error", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Validate GCT aggregator-only training")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--original_model", required=True)
    parser.add_argument("--trained_checkpoint", required=True)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(
        args.trained_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint.get("args", {})
    block_indices = checkpoint.get("block_indices")
    if not block_indices:
        raise RuntimeError("Checkpoint does not contain trained block indices")
    train_frame_blocks = checkpoint_args.get("train_frame_blocks", True)
    train_global_blocks = checkpoint_args.get("train_global_blocks", True)
    seed = checkpoint_args.get("seed", 42)

    output_dir = Path(
        args.output_dir
        or (Path(args.trained_checkpoint).parent / "vis_validation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=args.num_views,
        max_dim=518,
        shuffle_view_order=False,
        sampler_type="temporal_nearby",
        color_jitter_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
        seed=seed,
    )
    gt_visuals, frame_ids = collect_gt_visuals(dataset, args.num_vis)

    print("=" * 72)
    print("GCT Aggregator-Only Validation")
    print("=" * 72)
    print(f"  Checkpoint:    {args.trained_checkpoint}")
    print(f"  Blocks:        {block_indices}")
    print(f"  Frame/global:  {train_frame_blocks}/{train_global_blocks}")
    print(f"  Samples/views: {args.num_samples}/{args.num_views}")
    print(f"  Output:        {output_dir}")
    print("=" * 72)

    results = {}
    predictions = {}

    model = load_depth_gct_model(args.original_model, args.device, args.use_sdpa)
    results["Original"], predictions["Original"] = evaluate_model(
        model,
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "Original",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = load_depth_gct_model(args.original_model, args.device, args.use_sdpa)
    random_init_selected_blocks(
        model,
        block_indices,
        train_frame_blocks,
        train_global_blocks,
        seed,
    )
    results["Random Init"], predictions["Random Init"] = evaluate_model(
        model,
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "RandomInit",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = load_depth_gct_model(args.original_model, args.device, args.use_sdpa)
    load_trained_blocks(model, checkpoint)
    results["Trained"], predictions["Trained"] = evaluate_model(
        model,
        dataset,
        args.data_root,
        args.device,
        args.num_samples,
        args.num_vis,
        "Trained",
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    payload = {
        "metadata": {
            "trained_checkpoint": args.trained_checkpoint,
            "original_model": args.original_model,
            "iteration": checkpoint.get("iteration"),
            "block_indices": block_indices,
            "train_frame_blocks": train_frame_blocks,
            "train_global_blocks": train_global_blocks,
            "num_samples": args.num_samples,
            "num_views": args.num_views,
            "depth_alignment": "per-view GT median scale",
            "pose_alignment": (
                "first-frame translation anchor plus one positive least-squares "
                "translation scale per sample"
            ),
        },
        "results": results,
    }
    with open(output_dir / "aggregator_validation_results.json", "w") as handle:
        json.dump(payload, handle, indent=2)

    plot_depth_comparison(
        predictions,
        gt_visuals,
        frame_ids,
        output_dir / "depth_comparison_metric.png",
        "metric",
    )
    plot_depth_comparison(
        predictions,
        gt_visuals,
        frame_ids,
        output_dir / "depth_comparison_aligned.png",
        "aligned",
    )
    plot_metrics(results, output_dir / "metric_comparison.png")
    plot_error_comparison(
        predictions,
        gt_visuals,
        frame_ids,
        output_dir / "depth_error_comparison.png",
    )

    print("\n" + "-" * 72)
    print(
        f"{'Model':<18} {'Depth':>10} {'DepthA':>10} "
        f"{'AbsPoseA':>10} {'RelPoseA':>10}"
    )
    print("-" * 72)
    for name, metrics in results.items():
        print(
            f"{name:<18} "
            f"{metrics['metric_depth_loss']:>10.5f} "
            f"{metrics['scale_aligned_depth_loss']:>10.5f} "
            f"{metrics['scale_aligned_abs_pose_loss']:>10.5f} "
            f"{metrics['scale_aligned_rel_pose_loss']:>10.5f}"
        )
    print("-" * 72)
    initial = results["Random Init"]
    trained = results["Trained"]
    print(
        "Trained vs random-init depth improvement: "
        f"{(initial['metric_depth_loss'] - trained['metric_depth_loss']) / initial['metric_depth_loss'] * 100:+.1f}%"
    )
    print(f"Validation outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
