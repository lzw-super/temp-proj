"""
验证 GCTStream Backbone-Only 训练效果

对比三类模型在相同 Replica 数据上的表现：
1. Original GCTStream ViT-L 预训练模型（参考上限/原模型）
2. Initial Backbone-Only 模型（同 backbone-only 架构，DINOv2 初始化，未训练）
3. Trained Backbone-Only 模型（加载 train_backbone_only.py 的 checkpoint）

输出：
- depth loss / scale-aligned depth loss / pose loss / rel_pose loss 对比表
- Loss 对比柱状图
- 深度可视化对比

用法：
  python try_train/validate_backbone_only.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --original_model /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --trained_checkpoint try_train/checkpoints/backbone_vitb_2-24_5000iters/checkpoint_final.pt \
    --num_samples 20
"""

import sys
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.utils.load_fn import load_and_preprocess_images
from replica_dataset import ReplicaDataset
from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from train_replica_gct import load_gct_model, align_depth_to_gt
from train_backbone_only import BACKBONE_CONFIGS, create_gct_backbone_model


def _seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _autocast_context(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        cuda_device = torch.device(device)
        major = torch.cuda.get_device_capability(cuda_device)[0]
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return nullcontext()


def _load_checkpoint_args(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("args", {})


def _resolve_pretrained_path(dinov2_pretrained, require=True):
    if dinov2_pretrained and Path(dinov2_pretrained).exists():
        return dinov2_pretrained
    if require:
        raise FileNotFoundError(
            "DINOv2 pretrained weights are required for the initial backbone baseline. "
            f"Got: {dinov2_pretrained}"
        )
    print(f"[warn] DINOv2 pretrained path not found, constructing random init first: {dinov2_pretrained}")
    return ""


def _create_initial_backbone_model(backbone_type, dinov2_pretrained, downstream_checkpoint, device, use_sdpa, seed):
    """Create the untrained backbone-only baseline with the same seed as training."""
    _seed_everything(seed)
    pretrained_path = _resolve_pretrained_path(dinov2_pretrained, require=True)
    model = create_gct_backbone_model(
        backbone_type=backbone_type,
        dinov2_pretrained=pretrained_path,
        checkpoint_path=downstream_checkpoint,
        device=device,
        use_sdpa=use_sdpa,
    )
    model.eval()
    return model


def _load_trained_backbone_model(checkpoint_path, backbone_type, dinov2_pretrained, downstream_checkpoint, device, use_sdpa):
    """Load train_backbone_only.py checkpoint."""
    pretrained_path = _resolve_pretrained_path(dinov2_pretrained, require=False)
    model = create_gct_backbone_model(
        backbone_type=backbone_type,
        dinov2_pretrained=pretrained_path,
        checkpoint_path=downstream_checkpoint,
        device=device,
        use_sdpa=use_sdpa,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("full_model_state_dict", ckpt.get("model_state_dict", ckpt))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[_load_trained_backbone_model] Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[_load_trained_backbone_model] Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    return model, ckpt.get("iteration", -1), ckpt.get("loss_dict", {}), ckpt.get("args", {})


def _forward_gct_eval(model, images):
    """Forward pass through GCTStream in eval mode using Stage1 global attention."""
    B, S = images.shape[:2]
    with torch.no_grad():
        model.clean_kv_cache()
        aggregated_tokens_list, patch_start_idx = model.aggregator(
            images,
            selected_idx=[4, 11, 17, 23],
            num_frame_for_scale=S,
            sliding_window_size=-1,
            num_frame_per_block=S,
        )
        model.clean_kv_cache()

        if model.camera_head is not None:
            model.camera_head.clean_kv_cache()
        camera_output = model._predict_camera(
            aggregated_tokens_list,
            causal_inference=False,
            num_frame_per_block=S,
            num_frame_for_scale=S,
        )
        if model.camera_head is not None:
            model.camera_head.clean_kv_cache()

        depth_output = model._predict_depth(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )

    result = {}
    result.update(camera_output)
    result.update(depth_output)
    result["images"] = images
    return result


def _align_depth_scale_torch(pred, gt, mask):
    """
    Median-based scale alignment per (B, V) sample.
    pred, gt: [B, V, H, W]  mask: [B, V, H, W] (bool)
    """
    B, V = pred.shape[:2]
    aligned = pred.clone()
    for b in range(B):
        for v in range(V):
            m = mask[b, v]
            if m.sum() < 10:
                continue
            p_valid = pred[b, v][m]
            g_valid = gt[b, v][m]
            p_med = p_valid.median().clamp(min=1e-6)
            g_med = g_valid.median()
            aligned[b, v] = (pred[b, v] * (g_med / p_med)).clamp(min=0)
    return aligned


def compute_model_losses(model, dataset, data_root, device, num_samples, model_label):
    """Compute average losses for one model."""
    depth_loss_fn = DepthLoss(loss_type="masked_log_l1")
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()
    losses = {"depth": [], "depth_aligned": [], "abs_pose": [], "rel_pose": [], "total": []}

    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            depths_gt = sample["depths"].to(device)
            valid_masks = sample["valid_masks"].to(device)
            poses = sample["poses"].unsqueeze(0).to(device)
            frame_ids = sample["frame_ids"]

            image_paths = [
                str(Path(data_root) / "results" / f"frame{fid:06d}.jpg")
                for fid in frame_ids
            ]
            images_518 = load_and_preprocess_images(
                image_paths, mode="crop", image_size=518, patch_size=14
            )
            images_518 = images_518.unsqueeze(0).to(device)

            with _autocast_context(device):
                predictions = _forward_gct_eval(model, images_518)

            depth_pred, valid_masks_aligned = align_depth_to_gt(
                predictions["depth"], depths_gt.unsqueeze(0), valid_masks.unsqueeze(0)
            )

            depth_loss = depth_loss_fn(depth_pred, depths_gt.unsqueeze(0), valid_masks_aligned)

            depth_pred_f32 = depth_pred.float().clamp(min=1e-6)
            gt_f32 = depths_gt.unsqueeze(0).float()
            mask_bool = valid_masks_aligned.bool()
            depth_pred_aligned = _align_depth_scale_torch(depth_pred_f32, gt_f32, mask_bool)
            depth_loss_aligned = depth_loss_fn(
                depth_pred_aligned, depths_gt.unsqueeze(0), valid_masks_aligned
            )

            if "pose_enc" in predictions:
                abs_pose_loss = pose_loss_fn(predictions["pose_enc"], poses)
                rel_pose_loss = rel_pose_loss_fn(predictions["pose_enc"], poses)
            else:
                abs_pose_loss = torch.tensor(0.0, device=device)
                rel_pose_loss = torch.tensor(0.0, device=device)

            total = depth_loss.item() + 0.1 * abs_pose_loss.item() + 0.05 * rel_pose_loss.item()
            losses["depth"].append(depth_loss.item())
            losses["depth_aligned"].append(depth_loss_aligned.item())
            losses["abs_pose"].append(abs_pose_loss.item())
            losses["rel_pose"].append(rel_pose_loss.item())
            losses["total"].append(total)

            if (i + 1) % 10 == 0:
                print(
                    f"  [{model_label} {i+1}/{num_samples}] "
                    f"depth={depth_loss.item():.4f}, "
                    f"depth_aligned={depth_loss_aligned.item():.4f}, "
                    f"abs_pose={abs_pose_loss.item():.4f}, rel_pose={rel_pose_loss.item():.4f}"
                )

    return {k: float(np.mean(v)) for k, v in losses.items()}, losses


def _depth_to_numpy(predictions, view_idx=0):
    depth = predictions["depth"]
    if depth.dim() == 5:
        depth = depth[..., 0]
    return depth[0, view_idx].detach().cpu().float().numpy()


def _resize_to_gt(depth, gt_depth):
    if depth.shape != gt_depth.shape:
        depth = cv2.resize(depth, (gt_depth.shape[1], gt_depth.shape[0]))
    return depth


def _vrange(arr, lo=2, hi=98):
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def plot_loss_comparison(all_results, vis_dir):
    vis_dir = Path(vis_dir)
    metrics = ["depth", "depth_aligned", "abs_pose", "rel_pose", "total"]
    labels = [
        "Depth\n(unaligned)",
        "Depth\n(median-aligned)",
        "Abs Pose",
        "Rel Pose",
        "Total",
    ]
    colors = ["#2196F3", "#9C27B0", "#4CAF50", "#FF9800"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric, label in zip(axes, metrics, labels):
        names = list(all_results.keys())
        values = [all_results[name][metric] for name in names]
        bars = ax.bar(names, values, color=colors[:len(names)], alpha=0.85, edgecolor="black")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.set_ylabel("Loss")

    plt.suptitle("Backbone-Only Validation: Loss Comparison", fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / "backbone_loss_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] Loss comparison saved to {save_path}")


def visualize_depth_comparison(models, dataset, data_root, device, vis_dir, num_samples=5):
    """Generate first-view depth comparison for several samples."""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    num_samples = min(num_samples, len(dataset))
    num_cols = len(models) + 1

    fig, axes = plt.subplots(num_samples, num_cols, figsize=(5 * num_cols, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(num_samples):
        sample = dataset[i]
        depths_gt = sample["depths"].numpy()
        frame_ids = sample["frame_ids"]
        image_paths = [
            str(Path(data_root) / "results" / f"frame{fid:06d}.jpg")
            for fid in frame_ids
        ]
        images_518 = load_and_preprocess_images(
            image_paths, mode="crop", image_size=518, patch_size=14
        )
        images_518 = images_518.unsqueeze(0).to(device)

        v = 0
        for col, (name, model) in enumerate(models):
            with torch.no_grad():
                with _autocast_context(device):
                    pred = _forward_gct_eval(model, images_518)
            pred_depth = _resize_to_gt(_depth_to_numpy(pred, view_idx=v), depths_gt[v])
            vmin, vmax = _vrange(pred_depth)
            im = axes[i, col].imshow(pred_depth, cmap="gray", vmin=vmin, vmax=vmax)
            axes[i, col].set_title(
                f"{name}\nframe {frame_ids[v]}, range {vmin:.3f}-{vmax:.3f}",
                fontsize=9,
            )
            axes[i, col].axis("off")
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)

        gt_depth = depths_gt[v]
        vmin, vmax = _vrange(gt_depth)
        im = axes[i, -1].imshow(gt_depth, cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, -1].set_title(f"Ground Truth\nrange {vmin:.3f}-{vmax:.3f}", fontsize=9)
        axes[i, -1].axis("off")
        plt.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)

    plt.suptitle("Depth Comparison: Original vs Initial Backbone vs Trained Backbone", fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / "backbone_depth_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] Depth comparison saved to {save_path}")


def visualize_single_sample(models, dataset, data_root, device, vis_dir, sample_idx):
    """Generate a multi-view depth comparison for one sample."""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    sample = dataset[sample_idx]
    depths_gt = sample["depths"].numpy()
    frame_ids = sample["frame_ids"]
    num_views = min(len(frame_ids), 4)

    image_paths = [
        str(Path(data_root) / "results" / f"frame{fid:06d}.jpg")
        for fid in frame_ids
    ]
    images_518 = load_and_preprocess_images(
        image_paths, mode="crop", image_size=518, patch_size=14
    )
    images_518 = images_518.unsqueeze(0).to(device)

    predictions = []
    for name, model in models:
        with torch.no_grad():
            with _autocast_context(device):
                predictions.append((name, _forward_gct_eval(model, images_518)))

    num_rows = len(models) + 1
    fig, axes = plt.subplots(num_rows, num_views, figsize=(5 * num_views, 5 * num_rows))
    if num_views == 1:
        axes = axes[:, np.newaxis]

    for row, (name, pred) in enumerate(predictions):
        for col in range(num_views):
            pred_depth = _resize_to_gt(_depth_to_numpy(pred, view_idx=col), depths_gt[col])
            vmin, vmax = _vrange(pred_depth)
            im = axes[row, col].imshow(pred_depth, cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(
                f"{name} v{col} (frame {frame_ids[col]})\nrange {vmin:.3f}-{vmax:.3f}",
                fontsize=9,
            )
            axes[row, col].axis("off")
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    gt_row = len(models)
    for col in range(num_views):
        gt_depth = depths_gt[col]
        vmin, vmax = _vrange(gt_depth)
        im = axes[gt_row, col].imshow(gt_depth, cmap="gray", vmin=vmin, vmax=vmax)
        axes[gt_row, col].set_title(
            f"Ground Truth v{col}\nrange {vmin:.3f}-{vmax:.3f}",
            fontsize=9,
        )
        axes[gt_row, col].axis("off")
        plt.colorbar(im, ax=axes[gt_row, col], fraction=0.046, pad=0.04)

    plt.suptitle(f"Single Sample Backbone-Only Comparison (sample {sample_idx})", fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / f"single_sample_{sample_idx}_backbone_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[vis] Single sample comparison saved to {save_path}")


def _print_results_table(all_results):
    print(f"\n{'='*88}")
    print("Backbone-Only Validation Results")
    print(f"{'='*88}")
    print("  Note: DepthA = median-scale-aligned masked_log_l1 (structure-focused).")
    print(f"{'-'*88}")
    print(f"{'Model':<28} {'Depth':>10} {'DepthA':>10} {'AbsPose':>10} {'RelPose':>10} {'Total':>10}")
    print(f"{'-'*88}")
    for name, result in all_results.items():
        print(
            f"{name:<28} "
            f"{result['depth']:>10.4f} "
            f"{result['depth_aligned']:>10.4f} "
            f"{result['abs_pose']:>10.4f} "
            f"{result['rel_pose']:>10.4f} "
            f"{result['total']:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="验证 GCTStream Backbone-Only 训练效果")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--original_model", type=str,
                        default="/home/shared_files/model_weights/linbo_map/lingbot-map.pt")
    parser.add_argument("--trained_checkpoint", type=str, required=True)
    parser.add_argument("--backbone_type", type=str, default=None,
                        choices=list(BACKBONE_CONFIGS.keys()))
    parser.add_argument("--dinov2_pretrained", type=str, default=None)
    parser.add_argument("--num_views", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--num_vis", type=int, default=5)
    parser.add_argument("--single_sample", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use_sdpa", action="store_true", default=True)
    parser.add_argument("--use_flashinfer", dest="use_sdpa", action="store_false")
    args = parser.parse_args()

    ckpt_args = _load_checkpoint_args(args.trained_checkpoint)
    backbone_type = args.backbone_type or ckpt_args.get("backbone_type", "vitb")
    dinov2_pretrained = args.dinov2_pretrained or ckpt_args.get("dinov2_pretrained")
    downstream_checkpoint = ckpt_args.get("checkpoint") or args.original_model
    seed = args.seed if args.seed is not None else ckpt_args.get("seed", 42)

    if args.output_dir is None:
        args.output_dir = str(Path(args.trained_checkpoint).parent / "vis")
    vis_dir = Path(args.output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("GCTStream Backbone-Only Validation")
    print(f"{'='*70}")
    print(f"  Trained checkpoint: {args.trained_checkpoint}")
    print(f"  Original model:     {args.original_model}")
    print(f"  Downstream ckpt:    {downstream_checkpoint}")
    print(f"  Backbone:           {backbone_type}")
    print(f"  DINOv2 pretrained:  {dinov2_pretrained}")
    print(f"  Seed:               {seed}")
    print(f"  Num samples:        {args.num_samples}")
    print(f"  Num views:          {args.num_views}")
    print(f"  Output:             {vis_dir}")
    print(f"{'='*70}")

    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=args.num_views,
        max_dim=518,
        color_jitter_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
    )

    all_results = {}
    models_for_vis = []

    # 1. Original ViT-L GCT reference model.
    if args.original_model and Path(args.original_model).exists():
        print("\n[1/3] Loading ORIGINAL GCTStream ViT-L model...")
        original_model = load_gct_model(args.original_model, args.device, use_sdpa=args.use_sdpa)
        original_model.eval()
        original_result, _ = compute_model_losses(
            original_model, dataset, args.data_root, args.device, args.num_samples, "Original"
        )
        all_results["Original ViT-L"] = original_result
        models_for_vis.append(("Original ViT-L", original_model))
        print(
            f"  => Original: depth={original_result['depth']:.4f} "
            f"(aligned={original_result['depth_aligned']:.4f}), "
            f"abs_pose={original_result['abs_pose']:.4f}, rel_pose={original_result['rel_pose']:.4f}"
        )
    else:
        print(f"\n[1/3] Skip ORIGINAL model, path not found: {args.original_model}")

    # 2. Initial backbone-only baseline.
    print("\n[2/3] Creating INITIAL Backbone-Only model (untrained baseline)...")
    initial_model = _create_initial_backbone_model(
        backbone_type, dinov2_pretrained, downstream_checkpoint, args.device, args.use_sdpa, seed
    )
    initial_result, _ = compute_model_losses(
        initial_model, dataset, args.data_root, args.device, args.num_samples, "Initial"
    )
    all_results["Initial Backbone"] = initial_result
    models_for_vis.append(("Initial Backbone", initial_model))
    print(
        f"  => Initial: depth={initial_result['depth']:.4f} "
        f"(aligned={initial_result['depth_aligned']:.4f}), "
        f"abs_pose={initial_result['abs_pose']:.4f}, rel_pose={initial_result['rel_pose']:.4f}"
    )

    # 3. Trained backbone-only checkpoint.
    print("\n[3/3] Loading TRAINED Backbone-Only model...")
    trained_model, trained_iter, trained_loss, trained_args = _load_trained_backbone_model(
        args.trained_checkpoint, backbone_type, dinov2_pretrained, downstream_checkpoint, args.device, args.use_sdpa
    )
    print(f"  Checkpoint iteration: {trained_iter}")
    if trained_loss:
        print(f"  Training final loss: {trained_loss.get('total', 'N/A')}")
    trained_result, _ = compute_model_losses(
        trained_model, dataset, args.data_root, args.device, args.num_samples, "Trained"
    )
    all_results["Trained Backbone"] = trained_result
    models_for_vis.append(("Trained Backbone", trained_model))
    print(
        f"  => Trained: depth={trained_result['depth']:.4f} "
        f"(aligned={trained_result['depth_aligned']:.4f}), "
        f"abs_pose={trained_result['abs_pose']:.4f}, rel_pose={trained_result['rel_pose']:.4f}"
    )

    _print_results_table(all_results)

    if initial_result["depth"] > 0:
        print(f"\n{'='*70}")
        print("Improvement Analysis (Trained vs Initial Backbone)")
        print(f"{'='*70}")
        for metric in ["depth", "depth_aligned", "abs_pose", "rel_pose", "total"]:
            base = initial_result[metric]
            new = trained_result[metric]
            if base > 0:
                improvement = (base - new) / base * 100
                print(f"  {metric:<14}: {improvement:+.1f}%")

    json_payload = {
        "metadata": {
            "trained_checkpoint": args.trained_checkpoint,
            "original_model": args.original_model,
            "downstream_checkpoint": downstream_checkpoint,
            "backbone_type": backbone_type,
            "dinov2_pretrained": dinov2_pretrained,
            "seed": seed,
            "checkpoint_args": trained_args or ckpt_args,
        },
        "results": all_results,
    }
    with open(vis_dir / "backbone_validation_results.json", "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"[results] Saved to {vis_dir / 'backbone_validation_results.json'}")

    plot_loss_comparison(all_results, vis_dir)

    print("\n[vis] Generating depth comparison visualization...")
    visualize_depth_comparison(
        models_for_vis, dataset, args.data_root, args.device, vis_dir, args.num_vis
    )

    if args.single_sample is not None:
        sample_idx = min(args.single_sample, len(dataset) - 1)
        print(f"\n[vis] Generating single sample visualization (sample {sample_idx})...")
        visualize_single_sample(
            models_for_vis, dataset, args.data_root, args.device, vis_dir, sample_idx
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Validation complete. Results saved to {vis_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
