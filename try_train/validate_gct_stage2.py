"""
验证 GCTStream Stage2 训练效果（3-way 对比）

对比三种模型在相同 Replica 数据上的性能：
  1. Original GCTStream      - 未训练的 lingbot-map.pt
  2. Stage1 GCT (global attn) - train_replica_gct.py 训练产物
  3. Stage2 GCT (GCA streaming) - train_replica_gct_stage2.py 训练产物

关键设计：
  - Stage1/Original 用 global attention 前向评估 (forward_gct_heads_only)
  - Stage2 用 GCA streaming 前向评估 (forward_gct_heads_only_streaming)，
    与其训练时一致，公平体现 Stage2 的能力
  - 同时报告 unaligned + scale-aligned depth loss（公平对比 Original）
  - 灰度深度可视化用 per-image percentile vmin/vmax（避免尺度差导致纯黑）

用法：
  python try_train/validate_gct_stage2.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --original_model /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --stage1_checkpoint try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt \
    --stage2_checkpoint try_train/checkpoints/gct_stage2_5k/checkpoint_final.pt \
    --num_samples 20 \
    --num_vis 3 \
    --output_dir try_train/checkpoints/gct_stage2_5k/vis
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
from replica_dataset import ReplicaDataset
from head_only_model import DepthLoss, PoseLoss, RelativePoseLoss
from train_replica_gct import load_gct_model, align_depth_to_gt
from train_replica_gct_stage2 import (
    forward_gct_heads_only_streaming,
    load_gct_from_stage1,
)
from validate_gct_stage1 import (
    _forward_gct_eval,
    _load_gct_from_checkpoint,
    _align_depth_scale_torch,
)


# ---------------------------------------------------------------------------
# 3-way forward dispatcher
# ---------------------------------------------------------------------------

def forward_dispatch(model, images, mode, sliding_window_size=4, num_frame_for_scale=8):
    """根据 mode 选择 global-attention 或 GCA streaming forward。

    Args:
        mode: 'global'  -> forward_gct_heads_only (Stage1/Original)
              'gca'     -> forward_gct_heads_only_streaming (Stage2)
        sliding_window_size: GCA 模式下的局部窗口 k
        num_frame_for_scale: scale frames 数（GCA 模式下前 N 帧 bidirectional）
    """
    if mode == 'gca':
        return forward_gct_heads_only_streaming(
            model, images,
            sliding_window_size=sliding_window_size,
            num_frame_for_scale=num_frame_for_scale,
        )
    return _forward_gct_eval(model, images)


# ---------------------------------------------------------------------------
# Loss computation (single model, mode-aware)
# ---------------------------------------------------------------------------

def compute_model_losses(model, dataset, data_root, device, num_samples, mode,
                         sliding_window_size=4, num_frame_for_scale=8, model_label='Model'):
    """计算单个模型在 num_samples 个 sample 上的平均 loss（含 aligned 指标）。"""
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

            image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg")
                           for fid in frame_ids]
            images_518 = load_and_preprocess_images(image_paths, mode="crop",
                                                     image_size=518, patch_size=14)
            images_518 = images_518.unsqueeze(0).to(device)

            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = forward_dispatch(
                    model, images_518, mode,
                    sliding_window_size=sliding_window_size,
                    num_frame_for_scale=num_frame_for_scale,
                )

            # Depth (unaligned + aligned)
            depth_pred, valid_masks_aligned = align_depth_to_gt(
                predictions['depth'], depths_gt.unsqueeze(0), valid_masks.unsqueeze(0)
            )
            depth_loss = depth_loss_fn(depth_pred, depths_gt.unsqueeze(0), valid_masks_aligned)

            depth_pred_f32 = depth_pred.float().clamp(min=1e-6)
            gt_f32 = depths_gt.unsqueeze(0).float()
            mask_bool = valid_masks_aligned.bool()
            depth_pred_aligned = _align_depth_scale_torch(depth_pred_f32, gt_f32, mask_bool)
            depth_loss_aligned = depth_loss_fn(depth_pred_aligned,
                                                depths_gt.unsqueeze(0), valid_masks_aligned)

            # Pose
            if 'pose_enc' in predictions:
                abs_pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
                rel_pose_loss = rel_pose_loss_fn(predictions['pose_enc'], poses)
            else:
                abs_pose_loss = torch.tensor(0.0)
                rel_pose_loss = torch.tensor(0.0)

            total = (depth_loss.item()
                     + 0.1 * abs_pose_loss.item()
                     + 0.05 * rel_pose_loss.item())
            losses['depth'].append(depth_loss.item())
            losses['depth_aligned'].append(depth_loss_aligned.item())
            losses['abs_pose'].append(abs_pose_loss.item())
            losses['rel_pose'].append(rel_pose_loss.item())
            losses['total'].append(total)

            if (i + 1) % 10 == 0:
                print(f"  [{model_label} {i+1}/{num_samples}] "
                      f"depth={depth_loss.item():.4f} (aligned={depth_loss_aligned.item():.4f}), "
                      f"abs_pose={abs_pose_loss.item():.4f}, "
                      f"rel_pose={rel_pose_loss.item():.4f}")

    result = {k: float(np.mean(v)) for k, v in losses.items()}
    return result, losses


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _vrange(arr, lo=2, hi=98):
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


def visualize_3way_depth(original_model, stage1_model, stage2_model,
                          dataset, data_root, device, vis_dir,
                          num_samples=3, stage2_sliding_window=4,
                          stage2_num_frame_for_scale=8):
    """4 列对比图：Original / Stage1 / Stage2 / GT (灰度，per-image percentile)。"""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        depths_gt = sample['depths'].numpy()
        frame_ids = sample['frame_ids']

        image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg")
                       for fid in frame_ids]
        images_518 = load_and_preprocess_images(image_paths, mode="crop",
                                                 image_size=518, patch_size=14)
        images_518 = images_518.unsqueeze(0).to(device)

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=dtype):
                orig_pred = forward_dispatch(original_model, images_518, 'global')
                stage1_pred = forward_dispatch(stage1_model, images_518, 'global')
                stage2_pred = forward_dispatch(
                    stage2_model, images_518, 'gca',
                    sliding_window_size=stage2_sliding_window,
                    num_frame_for_scale=stage2_num_frame_for_scale,
                )

        v = 0
        gt_depth = depths_gt[v]
        depth_maps = {
            'Original': orig_pred['depth'][0, v, :, :, 0].cpu().float().numpy(),
            'Stage1 (global)': stage1_pred['depth'][0, v, :, :, 0].cpu().float().numpy(),
            'Stage2 (GCA)': stage2_pred['depth'][0, v, :, :, 0].cpu().float().numpy(),
            'GT': gt_depth,
        }

        for col, (label, dmap) in enumerate(depth_maps.items()):
            if dmap.shape != gt_depth.shape:
                dmap = cv2.resize(dmap, (gt_depth.shape[1], gt_depth.shape[0]))
            vmin, vmax = _vrange(dmap)
            im = axes[i, col].imshow(dmap, cmap='gray', vmin=vmin, vmax=vmax)
            title = f'{label}\n(frame {frame_ids[v]}, range {vmin:.3f}-{vmax:.3f})'
            axes[i, col].set_title(title, fontsize=10)
            axes[i, col].axis('off')
            plt.colorbar(im, ax=axes[i, col], fraction=0.046, pad=0.04)

    plt.suptitle('Depth Comparison: Original vs Stage1 (global) vs Stage2 (GCA) vs GT',
                 fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / '3way_depth_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] 3-way depth comparison saved to {save_path}")


def plot_3way_loss_comparison(all_results, vis_dir):
    """3-way 柱状图：每个 metric 一组柱（Original / Stage1 / Stage2）。"""
    vis_dir = Path(vis_dir)
    models = list(all_results.keys())
    metrics = ['depth', 'depth_aligned', 'abs_pose', 'rel_pose', 'total']
    labels = ['Depth Loss\n(unaligned, metric)',
              'Depth Loss\n(median-aligned, structure)',
              'Abs Pose Loss\n(geodesic+huber)',
              'Rel Pose Loss\n(geodesic+huber)',
              'Total Loss\n(weighted)']

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    palette = {'Original': '#2196F3', 'Stage1': '#4CAF50', 'Stage2': '#FF9800'}

    for ax, metric, label in zip(axes, metrics, labels):
        names, values, colors = [], [], []
        for m in models:
            v = all_results[m].get(metric)
            if v is None:
                continue
            short = m.split('\n')[0]
            names.append(short)
            values.append(v)
            colors.append(palette.get(short, '#999999'))
        if values:
            bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor='black')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f'{val:.4f}', ha='center', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Loss')
        ax.tick_params(axis='x', labelsize=9)

    plt.suptitle('3-way Validation: Original vs Stage1 vs Stage2', fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / '3way_loss_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[vis] 3-way loss comparison saved to {save_path}")


def compare_3way_single_sample(original_model, stage1_model, stage2_model,
                                dataset, data_root, device, vis_dir, sample_idx=0,
                                stage2_sliding_window=4, stage2_num_frame_for_scale=8):
    """单样本详细 3-way 对比：每 view 一列，4 行（3 模型 depth + GT），再 3 行误差图。"""
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample = dataset[sample_idx]
    depths_gt = sample['depths'].numpy()
    valid_masks_np = sample['valid_masks'].numpy()
    frame_ids = sample['frame_ids']
    V = len(frame_ids)

    image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg")
                   for fid in frame_ids]
    images_518 = load_and_preprocess_images(image_paths, mode="crop",
                                             image_size=518, patch_size=14)
    images_518 = images_518.unsqueeze(0).to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=dtype):
            orig_pred = forward_dispatch(original_model, images_518, 'global')
            stage1_pred = forward_dispatch(stage1_model, images_518, 'global')
            stage2_pred = forward_dispatch(
                stage2_model, images_518, 'gca',
                sliding_window_size=stage2_sliding_window,
                num_frame_for_scale=stage2_num_frame_for_scale,
            )

    # ---- 数值指标（per-view scale-aligned AbsRel/RMSE/Log-L1） ----
    print(f"\n{'='*80}")
    print(f"3-way Single Sample Comparison (sample_idx={sample_idx}, frames={frame_ids})")
    print(f"{'='*80}")

    def _absrel(p, g, m):
        if m.sum() < 10:
            return float('nan')
        pp, gg = p[m], g[m]
        scale = np.median(gg) / (np.median(pp) + 1e-8)
        return float(np.mean(np.abs(pp * scale - gg) / (gg + 1e-8)))

    def _rmse(p, g, m):
        if m.sum() < 10:
            return float('nan')
        pp, gg = p[m], g[m]
        scale = np.median(gg) / (np.median(pp) + 1e-8)
        return float(np.sqrt(np.mean((pp * scale - gg) ** 2)))

    print(f"\n  {'View':>6} | {'Model':<10} | {'AbsRel':>10} | {'RMSE':>10}")
    print(f"  {'-'*50}")
    for v in range(V):
        gt_v = depths_gt[v]
        mask_v = valid_masks_np[v] > 0
        for name, pred in [('Original', orig_pred), ('Stage1', stage1_pred),
                            ('Stage2', stage2_pred)]:
            d_v = pred['depth'][0, v, :, :, 0].cpu().float().numpy()
            if d_v.shape != gt_v.shape:
                d_v = cv2.resize(d_v, (gt_v.shape[1], gt_v.shape[0]))
            absrel = _absrel(d_v, gt_v, mask_v)
            rmse = _rmse(d_v, gt_v, mask_v)
            print(f"  {f'v{v}':>6} | {name:<10} | {absrel:>10.4f} | {rmse:>10.4f}")

    # ---- Visualization ----
    n_cols = min(V, 4)
    n_rows = 4  # row 0-3: orig / stage1 / stage2 / gt
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_cols):
        v = col
        gt_d = depths_gt[v]
        rows = [
            ('Original', orig_pred['depth'][0, v, :, :, 0].cpu().float().numpy()),
            ('Stage1 (global)', stage1_pred['depth'][0, v, :, :, 0].cpu().float().numpy()),
            ('Stage2 (GCA)', stage2_pred['depth'][0, v, :, :, 0].cpu().float().numpy()),
            ('GT', gt_d),
        ]
        for r, (label, dmap) in enumerate(rows):
            if dmap.shape != gt_d.shape:
                dmap = cv2.resize(dmap, (gt_d.shape[1], gt_d.shape[0]))
            vmin, vmax = _vrange(dmap)
            im = axes[r, col].imshow(dmap, cmap='gray', vmin=vmin, vmax=vmax)
            axes[r, col].set_title(f'v{v} {label}\nrange {vmin:.3f}-{vmax:.3f}', fontsize=10)
            axes[r, col].axis('off')
            plt.colorbar(im, ax=axes[r, col], fraction=0.046, pad=0.04)

    plt.suptitle(f'3-way Single Sample Detail (sample {sample_idx})', fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / f'3way_single_sample_{sample_idx}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[vis] 3-way single sample comparison saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-way validation: Original vs Stage1 (global) vs Stage2 (GCA)")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--original_model', type=str,
                        default='/home/shared_files/model_weights/linbo_map/lingbot-map.pt')
    parser.add_argument('--stage1_checkpoint', type=str, required=True)
    parser.add_argument('--stage2_checkpoint', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Samples for numerical loss comparison')
    parser.add_argument('--num_vis', type=int, default=3,
                        help='Samples shown in 4-col depth comparison figure')
    parser.add_argument('--single_sample', type=int, default=0,
                        help='Sample index for detailed single-sample comparison')
    parser.add_argument('--num_views', type=int, default=4,
                        help='Views per sample (Stage2 needs >=2 to exercise GCA)')
    parser.add_argument('--stage2_sliding_window', type=int, default=4,
                        help='GCA local window k for Stage2 forward')
    parser.add_argument('--stage2_num_frame_for_scale', type=int, default=8,
                        help='Stage2 scale frames (clamped to S at runtime)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output dir (default: <stage2_ckpt_dir>/vis)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = str(Path(args.stage2_checkpoint).parent / 'vis')
    vis_dir = Path(args.output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Dataset (no augmentation)
    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=args.num_views,
        max_dim=518,
        color_jitter_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
    )

    all_results = {}

    # ---- 1. Original model ----
    print(f"\n[1/3] Loading ORIGINAL GCTStream model...")
    original_model = load_gct_model(args.original_model, args.device, use_sdpa=True)
    original_model.eval()
    orig_result, _ = compute_model_losses(
        original_model, dataset, args.data_root, args.device,
        args.num_samples, mode='global', model_label='Original')
    all_results['Original\n(no train)'] = orig_result
    print(f"  => Original: depth={orig_result['depth']:.4f} "
          f"(aligned={orig_result['depth_aligned']:.4f}), "
          f"abs_pose={orig_result['abs_pose']:.4f}")

    # ---- 2. Stage1 model ----
    print(f"\n[2/3] Loading STAGE1 GCT model...")
    stage1_model, s1_iter, s1_loss = _load_gct_from_checkpoint(
        args.stage1_checkpoint, args.device)
    stage1_model.eval()
    print(f"  Stage1 iter={s1_iter}, training loss_dict={s1_loss}")
    stage1_result, _ = compute_model_losses(
        stage1_model, dataset, args.data_root, args.device,
        args.num_samples, mode='global', model_label='Stage1')
    all_results['Stage1\n(global attn)'] = stage1_result
    print(f"  => Stage1: depth={stage1_result['depth']:.4f} "
          f"(aligned={stage1_result['depth_aligned']:.4f}), "
          f"abs_pose={stage1_result['abs_pose']:.4f}")

    # ---- 3. Stage2 model (evaluated under GCA streaming forward) ----
    print(f"\n[3/3] Loading STAGE2 GCT model...")
    stage2_model = load_gct_from_stage1(args.stage2_checkpoint, args.device, use_sdpa=True)
    stage2_model.eval()
    stage2_result, _ = compute_model_losses(
        stage2_model, dataset, args.data_root, args.device,
        args.num_samples, mode='gca',
        sliding_window_size=args.stage2_sliding_window,
        num_frame_for_scale=args.stage2_num_frame_for_scale,
        model_label='Stage2')
    all_results['Stage2\n(GCA streaming)'] = stage2_result
    print(f"  => Stage2: depth={stage2_result['depth']:.4f} "
          f"(aligned={stage2_result['depth_aligned']:.4f}), "
          f"abs_pose={stage2_result['abs_pose']:.4f}")

    # ---- Comparison table ----
    print(f"\n{'='*88}")
    print(f"3-way Validation Results")
    print(f"{'='*88}")
    print(f"  Note: 'Depth'  = unaligned masked_log_l1 (absolute metric scale)")
    print(f"        'DepthA' = median-scale-aligned (fair structure-only comparison)")
    print(f"{'-'*88}")
    print(f"{'Model':<25} {'Depth':>10} {'DepthA':>10} {'AbsPose':>10} {'RelPose':>10} {'Total':>10}")
    print(f"{'-'*78}")
    for name, result in all_results.items():
        short = name.replace('\n', ' ')
        d = f"{result['depth']:.4f}"
        da = f"{result['depth_aligned']:.4f}"
        p = f"{result['abs_pose']:.4f}"
        r = f"{result['rel_pose']:.4f}"
        t = f"{result['total']:.4f}"
        print(f"{short:<25} {d:>10} {da:>10} {p:>10} {r:>10} {t:>10}")

    # Improvement deltas (% relative to Original)
    print(f"\n  Improvements vs Original (negative = better):")
    for model_key in ['Stage1\n(global attn)', 'Stage2\n(GCA streaming)']:
        res = all_results[model_key]
        d_imp = (orig_result['depth'] - res['depth']) / max(orig_result['depth'], 1e-8) * 100
        da_imp = (orig_result['depth_aligned'] - res['depth_aligned']) / max(orig_result['depth_aligned'], 1e-8) * 100
        p_imp = (orig_result['abs_pose'] - res['abs_pose']) / max(orig_result['abs_pose'], 1e-8) * 100
        print(f"    {model_key.replace(chr(10), ' '):<22} "
              f"depth: {d_imp:+.1f}%, depthA: {da_imp:+.1f}%, pose: {p_imp:+.1f}%")

    # ---- Save results JSON ----
    json_results = {name.replace('\n', ' '): result for name, result in all_results.items()}
    json_results['_meta'] = {
        'num_samples': args.num_samples,
        'num_views': args.num_views,
        'stage2_sliding_window': args.stage2_sliding_window,
        'stage2_num_frame_for_scale': args.stage2_num_frame_for_scale,
    }
    with open(vis_dir / 'gct_stage2_validation_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n[results] Saved to {vis_dir / 'gct_stage2_validation_results.json'}")

    # ---- Visualization ----
    plot_3way_loss_comparison(all_results, vis_dir)

    print(f"\n[vis] Generating 4-column depth comparison...")
    visualize_3way_depth(
        original_model, stage1_model, stage2_model,
        dataset, args.data_root, args.device, vis_dir,
        num_samples=args.num_vis,
        stage2_sliding_window=args.stage2_sliding_window,
        stage2_num_frame_for_scale=args.stage2_num_frame_for_scale,
    )

    print(f"\n[vis] Generating single-sample detail comparison...")
    compare_3way_single_sample(
        original_model, stage1_model, stage2_model,
        dataset, args.data_root, args.device, vis_dir,
        sample_idx=args.single_sample,
        stage2_sliding_window=args.stage2_sliding_window,
        stage2_num_frame_for_scale=args.stage2_num_frame_for_scale,
    )

    # Cleanup
    del original_model, stage1_model, stage2_model
    torch.cuda.empty_cache()

    print(f"\n{'='*88}")
    print(f"3-way validation complete. Results saved to {vis_dir}")
    print(f"{'='*88}")


if __name__ == '__main__':
    main()
