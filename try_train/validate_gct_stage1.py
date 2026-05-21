"""
验证 GCTStream Head-Only 训练效果

对比三种模型在相同数据上的表现：
1. 原始 GCTStream 模型（未训练，inference_streaming）
2. 训练后 GCTStream 模型（加载训练checkpoint）

输出：
- 各模型 depth loss / pose loss / rel_pose loss 对比表
- Loss 对比柱状图
- 深度可视化对比

用法：
  python try_train/validate_gct_stage1.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --original_model /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --trained_checkpoint try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt \
    --num_samples 50 \
    --output_dir try_train/checkpoints/gct_stage1_5k/vis
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from replica_dataset import ReplicaDataset
from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from train_replica_gct import load_gct_model, forward_gct_heads_only, align_depth_to_gt


def _load_gct_from_checkpoint(checkpoint_path, device, use_sdpa=True):
    """Load GCTStream from training checkpoint (contains full_model_state_dict)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=100, kv_cache_sliding_window=64,
        kv_cache_scale_frames=8, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=use_sdpa,
    )
    state_dict = ckpt.get('full_model_state_dict', ckpt.get('model_state_dict', ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    return model, ckpt.get('iteration', -1), ckpt.get('loss_dict', {})


def _forward_gct_eval(model, images):
    """Forward pass through GCTStream in eval mode (non-streaming)."""
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

        camera_output = model._predict_camera(
            aggregated_tokens_list,
            causal_inference=False,
            num_frame_per_block=S,
            num_frame_for_scale=S,
        )
        depth_output = model._predict_depth(
            aggregated_tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )

    result = {}
    result.update(camera_output)
    result.update(depth_output)
    result['images'] = images
    return result


def _align_depth_scale_torch(pred, gt, mask):
    """
    Median-based scale alignment per (B, V) sample.
    pred, gt: [B, V, H, W]  mask: [B, V, H, W] (bool)
    Returns aligned pred with same shape; clipped to >= 0.
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
            # avoid div-by-zero
            p_med = p_valid.median().clamp(min=1e-6)
            g_med = g_valid.median()
            scale = (g_med / p_med).item()
            aligned[b, v] = (pred[b, v] * scale).clamp(min=0)
    return aligned


def compute_gct_losses(model, dataset, data_root, device, num_samples):
    """Compute depth/pose/rel_pose losses for a GCTStream model.

    Reports BOTH:
      - depth: unaligned masked_log_l1 (sensitive to absolute metric scale)
      - depth_aligned: median-scale-aligned masked_log_l1 (fair comparison
        against original GCT model whose depth is in its own learned scale)
    """
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()

    losses = {'depth': [], 'depth_aligned': [], 'abs_pose': [], 'rel_pose': [], 'total': []}

    model.eval()
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            depths_gt = sample['depths'].to(device)
            valid_masks = sample['valid_masks'].to(device)
            poses = sample['poses'].unsqueeze(0).to(device)
            frame_ids = sample['frame_ids']

            # Load images at 518 resolution
            image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg") for fid in frame_ids]
            images_518 = load_and_preprocess_images(image_paths, mode="crop", image_size=518, patch_size=14)
            images_518 = images_518.unsqueeze(0).to(device)

            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = _forward_gct_eval(model, images_518)

            # Depth loss with resolution alignment
            depth_pred, valid_masks_aligned = align_depth_to_gt(
                predictions['depth'], depths_gt.unsqueeze(0), valid_masks.unsqueeze(0)
            )

            # Unaligned (absolute metric) depth loss
            depth_loss = depth_loss_fn(depth_pred, depths_gt.unsqueeze(0), valid_masks_aligned)

            # Scale-aligned depth loss (fair comparison: structure only)
            depth_pred_f32 = depth_pred.float().clamp(min=1e-6)
            gt_f32 = depths_gt.unsqueeze(0).float()
            mask_bool = valid_masks_aligned.bool()
            depth_pred_aligned = _align_depth_scale_torch(depth_pred_f32, gt_f32, mask_bool)
            depth_loss_aligned = depth_loss_fn(depth_pred_aligned, depths_gt.unsqueeze(0), valid_masks_aligned)

            # Pose loss
            if 'pose_enc' in predictions:
                abs_pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
                rel_pose_loss = rel_pose_loss_fn(predictions['pose_enc'], poses)
            else:
                abs_pose_loss = torch.tensor(0.0)
                rel_pose_loss = torch.tensor(0.0)

            total = depth_loss.item() + 0.1 * abs_pose_loss.item() + 0.05 * rel_pose_loss.item()
            losses['depth'].append(depth_loss.item())
            losses['depth_aligned'].append(depth_loss_aligned.item())
            losses['abs_pose'].append(abs_pose_loss.item())
            losses['rel_pose'].append(rel_pose_loss.item())
            losses['total'].append(total)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{num_samples}] depth={depth_loss.item():.4f}, "
                      f"depth_aligned={depth_loss_aligned.item():.4f}, "
                      f"abs_pose={abs_pose_loss.item():.4f}, rel_pose={rel_pose_loss.item():.4f}")

    result = {k: np.mean(v) for k, v in losses.items()}
    return result, losses


def visualize_depth_comparison(original_model, trained_model, dataset, data_root, device, vis_dir, num_samples=5):
    """Generate depth visualization comparing original vs trained model."""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        depths_gt = sample['depths'].numpy()
        frame_ids = sample['frame_ids']

        image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg") for fid in frame_ids]
        images_518 = load_and_preprocess_images(image_paths, mode="crop", image_size=518, patch_size=14)
        images_518 = images_518.unsqueeze(0).to(device)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                orig_pred = _forward_gct_eval(original_model, images_518)
                trained_pred = _forward_gct_eval(trained_model, images_518)

        # Use first view
        v = 0
        orig_depth = orig_pred['depth'][0, v, :, :, 0].cpu().float().numpy()
        trained_depth = trained_pred['depth'][0, v, :, :, 0].cpu().float().numpy()
        gt_depth = depths_gt[v]

        # Resize predictions to GT size if needed
        if orig_depth.shape != gt_depth.shape:
            orig_depth = cv2.resize(orig_depth, (gt_depth.shape[1], gt_depth.shape[0]))
        if trained_depth.shape != gt_depth.shape:
            trained_depth = cv2.resize(trained_depth, (gt_depth.shape[1], gt_depth.shape[0]))

        # Per-image vmin/vmax: 不同模型输出尺度可能不同（原始模型未经 metric 训练，
        # 输出值域与 GT 差异大；统一用 GT 尺度会让 original 显示为纯黑）
        def _vrange(arr, lo=2, hi=98):
            finite = arr[np.isfinite(arr) & (arr > 0)]
            if finite.size == 0:
                return 0.0, 1.0
            return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))

        v_orig = _vrange(orig_depth)
        v_train = _vrange(trained_depth)
        v_gt = _vrange(gt_depth)

        im0 = axes[i, 0].imshow(orig_depth, cmap='gray', vmin=v_orig[0], vmax=v_orig[1])
        axes[i, 0].set_title(f'Original Model\n(frame {frame_ids[v]}, range {v_orig[0]:.3f}-{v_orig[1]:.3f})')
        axes[i, 0].axis('off')

        im1 = axes[i, 1].imshow(trained_depth, cmap='gray', vmin=v_train[0], vmax=v_train[1])
        axes[i, 1].set_title(f'Trained Model (5K iters)\nrange {v_train[0]:.3f}-{v_train[1]:.3f}')
        axes[i, 1].axis('off')

        im2 = axes[i, 2].imshow(gt_depth, cmap='gray', vmin=v_gt[0], vmax=v_gt[1])
        axes[i, 2].set_title(f'Ground Truth\nrange {v_gt[0]:.3f}-{v_gt[1]:.3f}')
        axes[i, 2].axis('off')

        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.suptitle('Depth Comparison: Original vs Trained GCTStream', fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / 'depth_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Depth comparison saved to {save_path}")


def compare_single_sample(original_model, trained_model, dataset, data_root, device, vis_dir, sample_idx=0):
    """
    单样本详细对比：训练前后深度图、误差图、位姿差异、数值指标

    输出一张大图，包含：
    - Row 1: 原始深度 / 训练后深度 / GT深度
    - Row 2: 原始误差(绝对值) / 训练后误差(绝对值) / 误差差值(原始-训练，正=训练更好)
    - Row 3 (多view): 各view的深度对比
    - 文本: 位姿数值对比、各指标数值
    """
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[sample_idx]
    depths_gt = sample['depths'].numpy()
    valid_masks_np = sample['valid_masks'].numpy()
    poses_np = sample['poses'].numpy()
    frame_ids = sample['frame_ids']
    V = len(frame_ids)

    image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg") for fid in frame_ids]
    images_518 = load_and_preprocess_images(image_paths, mode="crop", image_size=518, patch_size=14)
    images_518 = images_518.unsqueeze(0).to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            orig_pred = _forward_gct_eval(original_model, images_518)
            trained_pred = _forward_gct_eval(trained_model, images_518)

    # ---- 数值指标 ----
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()

    print(f"\n{'='*70}")
    print(f"Single Sample Comparison (sample_idx={sample_idx}, frames={frame_ids})")
    print(f"{'='*70}")

    # Per-view depth metrics
    print(f"\n  {'View':>6} | {'Metric':<20} | {'Original':>12} | {'Trained':>12} | {'Improve':>10}")
    print(f"  {'-'*70}")

    for v in range(V):
        orig_depth_v = orig_pred['depth'][0, v, :, :, 0].cpu().float().numpy()
        trained_depth_v = trained_pred['depth'][0, v, :, :, 0].cpu().float().numpy()
        gt_v = depths_gt[v]
        mask_v = valid_masks_np[v] > 0

        if orig_depth_v.shape != gt_v.shape:
            orig_depth_v = cv2.resize(orig_depth_v, (gt_v.shape[1], gt_v.shape[0]))
        if trained_depth_v.shape != gt_v.shape:
            trained_depth_v = cv2.resize(trained_depth_v, (gt_v.shape[1], gt_v.shape[0]))

        # Scale-aligned abs rel
        def _abs_rel(pred, gt, mask):
            if mask.sum() < 10:
                return float('nan')
            p, g = pred[mask], gt[mask]
            scale = np.median(g) / (np.median(p) + 1e-8)
            p_aligned = p * scale
            return np.mean(np.abs(p_aligned - g) / (g + 1e-8))

        # RMSE
        def _rmse(pred, gt, mask):
            if mask.sum() < 10:
                return float('nan')
            p, g = pred[mask], gt[mask]
            scale = np.median(g) / (np.median(p) + 1e-8)
            p_aligned = p * scale
            return np.sqrt(np.mean((p_aligned - g) ** 2))

        # Log-L1
        def _log_l1(pred, gt, mask):
            if mask.sum() < 10:
                return float('nan')
            p, g = pred[mask], gt[mask]
            return np.mean(np.abs(np.log(p + 1e-8) - np.log(g + 1e-8)))

        orig_absrel = _abs_rel(orig_depth_v, gt_v, mask_v)
        trained_absrel = _abs_rel(trained_depth_v, gt_v, mask_v)
        orig_rmse = _rmse(orig_depth_v, gt_v, mask_v)
        trained_rmse = _rmse(trained_depth_v, gt_v, mask_v)
        orig_logl1 = _log_l1(orig_depth_v, gt_v, mask_v)
        trained_logl1 = _log_l1(trained_depth_v, gt_v, mask_v)

        for metric_name, o_val, t_val in [
            ('AbsRel', orig_absrel, trained_absrel),
            ('RMSE', orig_rmse, trained_rmse),
            ('Log-L1', orig_logl1, trained_logl1),
        ]:
            improve = ((o_val - t_val) / o_val * 100) if not np.isnan(o_val) and o_val > 0 else float('nan')
            print(f"  {f'v{v}':>6} | {metric_name:<20} | {o_val:>12.4f} | {t_val:>12.4f} | {improve:>+9.1f}%")

    # Pose comparison
    orig_pose_enc = orig_pred['pose_enc'][0].cpu().float().numpy()  # [V, 9]
    trained_pose_enc = trained_pred['pose_enc'][0].cpu().float().numpy()

    print(f"\n  Pose Encoding (per view):")
    print(f"  {'View':>6} | {'Model':<10} | {'Center (x,y,z)':<36} | {'Quat (w,x,y,z)':<44} | {'FovH,FovW':<16}")
    print(f"  {'-'*130}")
    for v in range(V):
        for name, enc in [('Original', orig_pose_enc), ('Trained', trained_pose_enc)]:
            c = enc[v, :3]
            q = enc[v, 3:7]
            f = enc[v, 7:9]
            print(f"  {f'v{v}':>6} | {name:<10} | ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})   "
                  f"| ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})   "
                  f"| ({f[0]:+.4f}, {f[1]:+.4f})")

    # ---- Visualization ----
    # Per-view: 3 rows (orig/trained/gt depth) + 2 rows (orig error / trained error)
    n_cols = min(V, 4)  # max 4 views shown
    n_rows = 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Per-image vmin/vmax helper（避免原始模型与 GT 尺度差异导致纯黑/纯白）
    def _vrange(arr, lo=2, hi=98):
        finite = arr[np.isfinite(arr) & (arr > 0)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))

    # 误差图使用 GT 尺度作为参考上限（误差量级与 GT 同单位）
    gt_vmax = max(np.percentile(depths_gt[v][valid_masks_np[v] > 0], 95)
                  for v in range(min(V, n_cols)) if (valid_masks_np[v] > 0).any())

    for col in range(n_cols):
        v = col
        orig_d = orig_pred['depth'][0, v, :, :, 0].cpu().float().numpy()
        trained_d = trained_pred['depth'][0, v, :, :, 0].cpu().float().numpy()
        gt_d = depths_gt[v]
        mask_d = valid_masks_np[v] > 0

        if orig_d.shape != gt_d.shape:
            orig_d = cv2.resize(orig_d, (gt_d.shape[1], gt_d.shape[0]))
        if trained_d.shape != gt_d.shape:
            trained_d = cv2.resize(trained_d, (gt_d.shape[1], gt_d.shape[0]))

        # Per-image grayscale ranges
        v_orig = _vrange(orig_d)
        v_train = _vrange(trained_d)
        v_gt = _vrange(gt_d)

        # Row 0-2: Depth maps (grayscale, each with own range)
        im0 = axes[0, col].imshow(orig_d, cmap='gray', vmin=v_orig[0], vmax=v_orig[1])
        axes[0, col].set_title(f'v{v} Original Depth (frame {frame_ids[v]})\nrange {v_orig[0]:.3f}-{v_orig[1]:.3f}')
        axes[0, col].axis('off')
        plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

        im1 = axes[1, col].imshow(trained_d, cmap='gray', vmin=v_train[0], vmax=v_train[1])
        axes[1, col].set_title(f'v{v} Trained Depth\nrange {v_train[0]:.3f}-{v_train[1]:.3f}')
        axes[1, col].axis('off')
        plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

        im2 = axes[2, col].imshow(gt_d, cmap='gray', vmin=v_gt[0], vmax=v_gt[1])
        axes[2, col].set_title(f'v{v} GT Depth\nrange {v_gt[0]:.3f}-{v_gt[1]:.3f}')
        axes[2, col].axis('off')
        plt.colorbar(im2, ax=axes[2, col], fraction=0.046, pad=0.04)

        # Row 3-4: Error maps (absolute error, masked)
        err_max = gt_vmax * 0.5
        orig_err = np.abs(orig_d - gt_d)
        trained_err = np.abs(trained_d - gt_d)
        orig_err[~mask_d] = 0
        trained_err[~mask_d] = 0

        im3 = axes[3, col].imshow(orig_err, cmap='hot', vmin=0, vmax=err_max)
        mean_orig_err = orig_err[mask_d].mean() if mask_d.any() else 0
        axes[3, col].set_title(f'v{v} Original Error\nmean={mean_orig_err:.3f}')
        axes[3, col].axis('off')
        plt.colorbar(im3, ax=axes[3, col], fraction=0.046, pad=0.04)

        im4 = axes[4, col].imshow(trained_err, cmap='hot', vmin=0, vmax=err_max)
        mean_trained_err = trained_err[mask_d].mean() if mask_d.any() else 0
        axes[4, col].set_title(f'v{v} Trained Error\nmean={mean_trained_err:.3f}')
        axes[4, col].axis('off')
        plt.colorbar(im4, ax=axes[4, col], fraction=0.046, pad=0.04)

    plt.suptitle(f'Single Sample Detail: Original vs Trained GCTStream (sample {sample_idx})', fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / f'single_sample_{sample_idx}_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[vis] Single sample comparison saved to {save_path}")

    # Also save pose comparison as JSON
    pose_compare = {
        'sample_idx': sample_idx,
        'frame_ids': [int(f) for f in frame_ids],
        'original_pose_enc': orig_pose_enc.tolist(),
        'trained_pose_enc': trained_pose_enc.tolist(),
        'pose_diff': (trained_pose_enc - orig_pose_enc).tolist(),
    }
    with open(vis_dir / f'single_sample_{sample_idx}_pose.json', 'w') as f:
        json.dump(pose_compare, f, indent=2)
    print(f"[vis] Pose comparison saved to {vis_dir / f'single_sample_{sample_idx}_pose.json'}")


def plot_loss_comparison(all_results, vis_dir):
    """Plot loss comparison bar chart."""
    vis_dir = Path(vis_dir)
    models = list(all_results.keys())
    metrics = ['depth', 'depth_aligned', 'abs_pose', 'rel_pose', 'total']
    labels = ['Depth Loss\n(unaligned, metric)', 'Depth Loss\n(median-aligned, structure)',
              'Abs Pose Loss\n(geodesic+huber)',
              'Rel Pose Loss\n(geodesic+huber)', 'Total Loss\n(weighted)']

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for ax, metric, label in zip(axes, metrics, labels):
        values, names = [], []
        for m in models:
            if metric in all_results[m] and all_results[m][metric] is not None:
                values.append(all_results[m][metric])
                names.append(m)
        if values:
            colors = ['#2196F3', '#4CAF50', '#FF9800'][:len(values)]
            bars = ax.bar(names, values, color=colors, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Loss')
        ax.tick_params(axis='x', rotation=20, labelsize=8)

    plt.suptitle('GCTStream Training Validation: Loss Comparison', fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / 'gct_loss_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] Loss comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="验证 GCTStream Head-Only 训练效果")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--original_model', type=str,
                        default='/home/shared_files/model_weights/linbo_map/lingbot-map.pt')
    parser.add_argument('--trained_checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--num_vis', type=int, default=5, help='Number of samples for depth visualization')
    parser.add_argument('--single_sample', type=int, default=None,
                        help='If set, run detailed single-sample comparison at this index')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: same as trained_checkpoint/vis)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.trained_checkpoint).parent / 'vis')
    vis_dir = Path(args.output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Create dataset (no augmentation for validation)
    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=2,
        max_dim=518,
        color_jitter_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
    )

    all_results = {}

    # 1. Original GCTStream model
    print(f"\n[1/2] Loading ORIGINAL GCTStream model...")
    original_model = load_gct_model(args.original_model, args.device, use_sdpa=True)
    original_model.eval()
    orig_result, _ = compute_gct_losses(
        original_model, dataset, args.data_root, args.device, args.num_samples)
    all_results['Original\nGCTStream'] = orig_result
    print(f"  => Original: depth={orig_result['depth']:.4f} (aligned={orig_result['depth_aligned']:.4f}), "
          f"abs_pose={orig_result['abs_pose']:.4f}, rel_pose={orig_result['rel_pose']:.4f}")

    # 2. Trained GCTStream model
    print(f"\n[2/2] Loading TRAINED GCTStream model...")
    trained_model, trained_iter, trained_loss = _load_gct_from_checkpoint(
        args.trained_checkpoint, args.device)
    trained_model.eval()
    print(f"  Checkpoint iteration: {trained_iter}")
    if trained_loss:
        print(f"  Training final loss: {trained_loss.get('total', 'N/A')}")

    trained_result, _ = compute_gct_losses(
        trained_model, dataset, args.data_root, args.device, args.num_samples)
    all_results['Trained\nGCTStream'] = trained_result
    print(f"  => Trained: depth={trained_result['depth']:.4f} (aligned={trained_result['depth_aligned']:.4f}), "
          f"abs_pose={trained_result['abs_pose']:.4f}, rel_pose={trained_result['rel_pose']:.4f}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"GCTStream Training Validation Results")
    print(f"{'='*80}")
    print(f"  Note: 'Depth' = unaligned masked_log_l1 (absolute metric scale).")
    print(f"        'DepthA' = median-scale-aligned masked_log_l1 (structure only,")
    print(f"                   fair vs original model which has its own learned scale).")
    print(f"{'-'*80}")
    print(f"{'Model':<25} {'Depth':>10} {'DepthA':>10} {'AbsPose':>10} {'RelPose':>10} {'Total':>10}")
    print(f"{'-'*75}")
    for name, result in all_results.items():
        d = f"{result['depth']:.4f}" if result.get('depth') is not None else "N/A"
        da = f"{result['depth_aligned']:.4f}" if result.get('depth_aligned') is not None else "N/A"
        p = f"{result['abs_pose']:.4f}" if result.get('abs_pose') is not None else "N/A"
        r = f"{result['rel_pose']:.4f}" if result.get('rel_pose') is not None else "N/A"
        t = f"{result['total']:.4f}" if result.get('total') is not None else "N/A"
        print(f"{name:<25} {d:>10} {da:>10} {p:>10} {r:>10} {t:>10}")

    # Improvement percentages
    if orig_result['depth'] > 0:
        depth_improve = (orig_result['depth'] - trained_result['depth']) / orig_result['depth'] * 100
        depth_aligned_improve = (orig_result['depth_aligned'] - trained_result['depth_aligned']) / orig_result['depth_aligned'] * 100
        pose_improve = (orig_result['abs_pose'] - trained_result['abs_pose']) / orig_result['abs_pose'] * 100
        print(f"\n  Depth improvement (unaligned, metric scale): {depth_improve:+.1f}%")
        print(f"  Depth improvement (aligned, structure only): {depth_aligned_improve:+.1f}%")
        print(f"  Pose improvement:                            {pose_improve:+.1f}%")

    # Save results
    json_results = {}
    for name, result in all_results.items():
        json_results[name.replace('\n', ' ')] = result
    with open(vis_dir / 'gct_validation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # Plot loss comparison
    plot_loss_comparison(all_results, vis_dir)

    # Depth visualization
    print(f"\n[vis] Generating depth comparison visualization...")
    visualize_depth_comparison(
        original_model, trained_model, dataset, args.data_root, args.device, vis_dir, args.num_vis)

    # Single sample detailed comparison
    if args.single_sample is not None:
        print(f"\n[vis] Generating single sample detailed comparison...")
        compare_single_sample(
            original_model, trained_model, dataset, args.data_root, args.device,
            vis_dir, sample_idx=args.single_sample)
    else:
        # Default: compare sample 0
        print(f"\n[vis] Generating single sample detailed comparison (default sample 0)...")
        compare_single_sample(
            original_model, trained_model, dataset, args.data_root, args.device,
            vis_dir, sample_idx=0)

    # Cleanup
    del original_model, trained_model
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"Validation complete. Results saved to {vis_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
