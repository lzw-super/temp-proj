"""
计算未训练模型/原始模型的训练loss，用于与训练后模型对比

功能：
1. 加载 HeadOnlyModel（未训练），计算 depth/pose/rel_pose loss
2. 加载原始 GCTStream 模型，计算 depth loss（原始模型输出518分辨率，需resize到GT尺寸）
3. 输出各模型loss对比表，保存到vis文件夹

用法：
  python try_train/compute_baseline_losses.py \
    --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
    --original_model /home/shared_files/model_weights/linbo_map/lingbot-map.pt \
    --num_samples 50 \
    --output_dir try_train/checkpoints/v2_enhanced_10k/vis

作者：Claude Code
日期：2026-05-21
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
from head_only_model import create_head_only_model, DepthLoss, PoseLoss, RelativePoseLoss


def compute_untrained_head_losses(dataset, device, num_samples, img_size=224):
    """计算未训练HeadOnlyModel的loss"""
    model = create_head_only_model(
        backbone_name='dinov2_vits14',
        freeze_backbone=True,
        img_size=img_size,
        train_depth_head=True,
        train_pose_head=True,
        num_views=2,
    )
    model = model.to(device).eval()

    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()

    losses = {'depth': [], 'abs_pose': [], 'rel_pose': [], 'total': []}

    print(f"\n[Untrained HeadOnlyModel] Computing losses on {num_samples} samples...")

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            images = sample['images'].unsqueeze(0).to(device)   # [1, V, 3, H, W]
            depths = sample['depths'].unsqueeze(0).to(device)   # [1, V, H, W]
            valid_masks = sample['valid_masks'].unsqueeze(0).to(device)  # [1, V, H, W]
            poses = sample['poses'].unsqueeze(0).to(device)     # [1, V, 4, 4]

            with torch.amp.autocast('cuda', enabled=True):
                predictions = model(images)

                depth_loss = depth_loss_fn(predictions['depth'], depths, valid_masks)
                abs_pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
                rel_pose_loss = rel_pose_loss_fn(predictions['pose_enc'], poses)

            total = depth_loss.item() + 0.1 * abs_pose_loss.item() + 0.05 * rel_pose_loss.item()

            losses['depth'].append(depth_loss.item())
            losses['abs_pose'].append(abs_pose_loss.item())
            losses['rel_pose'].append(rel_pose_loss.item())
            losses['total'].append(total)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{num_samples}] depth={depth_loss.item():.4f}, "
                      f"abs_pose={abs_pose_loss.item():.4f}, rel_pose={rel_pose_loss.item():.4f}")

    result = {k: np.mean(v) for k, v in losses.items()}
    print(f"  => Mean: depth={result['depth']:.4f}, abs_pose={result['abs_pose']:.4f}, "
          f"rel_pose={result['rel_pose']:.4f}, total={result['total']:.4f}")
    return result, losses


def compute_original_depth_loss(original_model, dataset, data_root, device, num_samples):
    """计算原始GCTStream模型的depth loss"""
    losses = {'depth': []}

    print(f"\n[Original GCTStream] Computing depth loss on {num_samples} samples...")

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            depths_gt = sample['depths'].to(device)
            valid_masks = sample['valid_masks'].to(device)
            frame_ids = sample['frame_ids']

            image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg") for fid in frame_ids]
            images_518 = load_and_preprocess_images(image_paths, mode="crop", image_size=518, patch_size=14)
            images_518 = images_518.unsqueeze(0).to(device)

            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
            with torch.amp.autocast("cuda", dtype=dtype):
                predictions = original_model.inference_streaming(images_518, num_scale_frames=8)

            depth_pred = predictions['depth']
            if depth_pred.dim() == 5:
                depth_pred = depth_pred.squeeze(-1)

            depth_pred_np = depth_pred[0].cpu().float().numpy()
            depths_gt_np = depths_gt.cpu().numpy()
            valid_masks_np = valid_masks.cpu().numpy()

            for v in range(len(frame_ids)):
                pred_v = depth_pred_np[v]
                gt_v = depths_gt_np[v]
                mask_v = valid_masks_np[v]

                if pred_v.shape != gt_v.shape:
                    pred_v = cv2.resize(pred_v, (gt_v.shape[1], gt_v.shape[0]),
                                       interpolation=cv2.INTER_LINEAR)

                pred_v = _align_depth_scale(pred_v, gt_v, mask_v)

                pred_t = torch.from_numpy(pred_v).float().to(device)
                gt_t = torch.from_numpy(gt_v).float().to(device)
                mask_t = torch.from_numpy(mask_v).bool().to(device)

                valid_pred = pred_t[mask_t]
                valid_gt = gt_t[mask_t]
                if len(valid_pred) > 0:
                    log_l1 = torch.abs(torch.log(valid_pred + 1e-8) - torch.log(valid_gt + 1e-8))
                    loss_val = log_l1.mean().item()
                    losses['depth'].append(loss_val)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{num_samples}] current mean depth_loss={np.mean(losses['depth']):.4f}")

    result = {k: np.mean(v) for k, v in losses.items()}
    print(f"  => Mean depth_loss={result['depth']:.4f}")
    return result, losses


def compute_trained_head_losses(checkpoint_path, dataset, device, num_samples):
    """计算已训练HeadOnlyModel的loss"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt['args']

    model = create_head_only_model(
        backbone_name=model_args['backbone'],
        freeze_backbone=model_args['freeze_backbone'],
        img_size=model_args['img_size'],
        train_depth_head=True,
        train_pose_head=model_args.get('train_pose', True),
        num_views=2,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()

    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = RelativePoseLoss()

    losses = {'depth': [], 'abs_pose': [], 'rel_pose': [], 'total': []}

    print(f"\n[Trained HeadOnlyModel] Computing losses on {num_samples} samples...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Iteration: {ckpt['iteration']}")

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            images = sample['images'].unsqueeze(0).to(device)   # [1, V, 3, H, W]
            depths = sample['depths'].unsqueeze(0).to(device)   # [1, V, H, W]
            valid_masks = sample['valid_masks'].unsqueeze(0).to(device)  # [1, V, H, W]
            poses = sample['poses'].unsqueeze(0).to(device)     # [1, V, 4, 4]

            with torch.amp.autocast('cuda', enabled=True):
                predictions = model(images)

                depth_loss = depth_loss_fn(predictions['depth'], depths, valid_masks)
                abs_pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
                rel_pose_loss = rel_pose_loss_fn(predictions['pose_enc'], poses)

            total = depth_loss.item() + 0.1 * abs_pose_loss.item() + 0.05 * rel_pose_loss.item()

            losses['depth'].append(depth_loss.item())
            losses['abs_pose'].append(abs_pose_loss.item())
            losses['rel_pose'].append(rel_pose_loss.item())
            losses['total'].append(total)

            if (i + 1) % 10 == 0:
                print(f"  [{i+1}/{num_samples}] depth={depth_loss.item():.4f}, "
                      f"abs_pose={abs_pose_loss.item():.4f}, rel_pose={rel_pose_loss.item():.4f}")

    result = {k: np.mean(v) for k, v in losses.items()}
    print(f"  => Mean: depth={result['depth']:.4f}, abs_pose={result['abs_pose']:.4f}, "
          f"rel_pose={result['rel_pose']:.4f}, total={result['total']:.4f}")
    return result, losses


def _align_depth_scale(pred, gt, mask):
    """尺度对齐"""
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    if len(pred_valid) < 10:
        return pred
    A = np.column_stack([pred_valid, np.ones_like(pred_valid)])
    result = np.linalg.lstsq(A, gt_valid, rcond=None)
    scale, shift = result[0][0], result[0][1]
    aligned = scale * pred + shift
    return np.maximum(aligned, 0)


def plot_loss_comparison(all_results, vis_dir):
    """绘制不同模型的loss对比图"""
    models = list(all_results.keys())
    metrics = ['depth', 'abs_pose', 'rel_pose', 'total']
    metric_labels = ['Depth Loss\n(masked_log_l1)', 'Abs Pose Loss\n(geodesic+huber)',
                     'Rel Pose Loss\n(geodesic+huber)', 'Total Loss\n(weighted sum)']

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, metric, label in zip(axes, metrics, metric_labels):
        values = []
        labels = []
        for model_name in models:
            if metric in all_results[model_name] and all_results[model_name][metric] is not None:
                values.append(all_results[model_name][metric])
                labels.append(model_name)

        if values:
            colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336'][:len(values)]
            bars = ax.bar(labels, values, color=colors, alpha=0.8, edgecolor='black')
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                       f'{val:.4f}', ha='center', fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel('Loss')
        ax.tick_params(axis='x', rotation=30, labelsize=8)

    plt.suptitle('Training Loss Comparison (same data, same loss functions)', fontsize=14)
    plt.tight_layout()
    save_path = vis_dir / 'loss_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="计算基线模型的训练loss用于对比")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--original_model', type=str,
                        default='/home/shared_files/model_weights/linbo_map/lingbot-map.pt')
    parser.add_argument('--trained_checkpoint', type=str, default=None,
                        help="已训练模型的checkpoint路径")
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='try_train/checkpoints/baseline_losses')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    vis_dir = Path(args.output_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Baseline Loss Computation")
    print(f"{'='*60}")
    print(f"  Data: {args.data_root}")
    print(f"  Num samples: {args.num_samples}")

    # 创建验证数据集（无增强）
    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=2,
        max_dim=args.img_size,
        color_jitter_prob=0.0,
        spatial_rescale_range=None,
        aspect_ratio_range=None,
    )

    all_results = {}

    # 1. 未训练HeadOnlyModel
    untrained_result, _ = compute_untrained_head_losses(dataset, args.device, args.num_samples, args.img_size)
    all_results['Untrained\nHeadOnly'] = untrained_result

    # 2. 已训练HeadOnlyModel（如果提供）
    if args.trained_checkpoint:
        trained_result, _ = compute_trained_head_losses(
            args.trained_checkpoint, dataset, args.device, args.num_samples)
        all_results['Trained\nHeadOnly'] = trained_result

    # 3. 原始GCTStream模型（仅depth loss，架构不同无法直接比pose loss）
    if Path(args.original_model).exists():
        print(f"\n[Original GCTStream] Loading model...")
        original_model = GCTStream(
            img_size=518, patch_size=14, enable_3d_rope=True,
            max_frame_num=100, kv_cache_sliding_window=64,
            kv_cache_scale_frames=8, kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True, use_sdpa=True,
        )
        ckpt = torch.load(args.original_model, map_location=args.device, weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        original_model.load_state_dict(state_dict, strict=False)
        original_model = original_model.to(args.device).eval()
        print(f"  Loaded from {args.original_model}")

        orig_result, _ = compute_original_depth_loss(
            original_model, dataset, args.data_root, args.device, args.num_samples)
        # 原始模型架构不同，无法直接用相同PoseLoss，设为None
        orig_result['abs_pose'] = None
        orig_result['rel_pose'] = None
        orig_result['total'] = None
        all_results['Original\nGCTStream'] = orig_result

        del original_model
        torch.cuda.empty_cache()

    # 打印对比表
    print(f"\n{'='*60}")
    print(f"Loss Comparison Summary")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Depth':>10} {'AbsPose':>10} {'RelPose':>10} {'Total':>10}")
    print(f"{'-'*65}")
    for name, result in all_results.items():
        depth_str = f"{result['depth']:.4f}" if result.get('depth') is not None else "N/A"
        pose_str = f"{result['abs_pose']:.4f}" if result.get('abs_pose') is not None else "N/A"
        rel_str = f"{result['rel_pose']:.4f}" if result.get('rel_pose') is not None else "N/A"
        total_str = f"{result['total']:.4f}" if result.get('total') is not None else "N/A"
        print(f"{name:<25} {depth_str:>10} {pose_str:>10} {rel_str:>10} {total_str:>10}")

    # 保存结果JSON
    json_results = {}
    for name, result in all_results.items():
        json_results[name.replace('\n', ' ')] = {
            k: (v if v is not None else "N/A") for k, v in result.items()
        }
    with open(vis_dir / 'baseline_loss_comparison.json', 'w') as f:
        json.dump(json_results, f, indent=2)

    # 绘制对比图
    plot_loss_comparison(all_results, vis_dir)

    print(f"\n{'='*60}")
    print(f"Done. Results saved to {vis_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
