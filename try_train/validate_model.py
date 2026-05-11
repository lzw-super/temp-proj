"""
训练模型验证脚本

验证训练后的模型是否能够正确预测深度：
1. 加载 checkpoint
2. 对 Replica 数据进行推理
3. 检查 depth 预测范围是否合理
4. 可视化 depth 预测结果

作者：Claude Code
日期：2026-05-11
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from replica_dataset import ReplicaDataset
from head_only_model import create_head_only_model


def load_checkpoint(checkpoint_path, device='cuda'):
    """加载 checkpoint"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"[load_checkpoint] 加载完成: {checkpoint_path}")
    print(f"  - Epoch: {ckpt['epoch']}")
    print(f"  - Loss: {ckpt['loss']:.4f}")
    return ckpt


def validate_model(model, dataset, device='cuda', num_samples=5):
    """验证模型推理"""
    model.eval()

    results = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            images = sample['images'].unsqueeze(0).to(device)  # [1, V, 3, H, W]
            depths_gt = sample['depths'].to(device)  # [V, H, W]
            valid_masks = sample['valid_masks'].to(device)

            # 推理
            predictions = model(images)

            # 检查预测
            depth_pred = predictions['depth']  # [1, V, H, W, 1] or [1, V, H, W]
            if depth_pred.dim() == 5:
                depth_pred = depth_pred.squeeze(-1)

            # 统计
            depth_pred_np = depth_pred[0].cpu().numpy()  # [V, H, W]
            depths_gt_np = depths_gt.cpu().numpy()

            for v in range(depth_pred_np.shape[0]):
                pred_v = depth_pred_np[v]
                gt_v = depths_gt_np[v]
                valid_mask = valid_masks[v].cpu().numpy()

                # 有效区域的统计
                pred_valid = pred_v[valid_mask]
                gt_valid = gt_v[valid_mask]

                if len(pred_valid) > 0 and len(gt_valid) > 0:
                    # 深度范围
                    pred_min, pred_max = pred_valid.min(), pred_valid.max()
                    pred_mean = pred_valid.mean()
                    gt_min, gt_max = gt_valid.min(), gt_valid.max()
                    gt_mean = gt_valid.mean()

                    # 相对误差
                    rel_error = np.abs(pred_valid - gt_valid) / (gt_valid + 1e-3)
                    mean_rel_error = rel_error.mean()

                    results.append({
                        'sample_idx': i,
                        'view_idx': v,
                        'pred_min': pred_min,
                        'pred_max': pred_max,
                        'pred_mean': pred_mean,
                        'gt_min': gt_min,
                        'gt_max': gt_max,
                        'gt_mean': gt_mean,
                        'mean_rel_error': mean_rel_error,
                        'depth_pred': pred_v,
                        'depth_gt': gt_v,
                        'valid_mask': valid_mask,
                    })

                    print(f"[Sample {i}, View {v}]")
                    print(f"  Pred: min={pred_min:.3f}, max={pred_max:.3f}, mean={pred_mean:.3f}")
                    print(f"  GT:   min={gt_min:.3f}, max={gt_max:.3f}, mean={gt_mean:.3f}")
                    print(f"  Rel Error: {mean_rel_error:.3f}")

    return results


def visualize_results(results, output_dir):
    """可视化深度预测结果"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, result in enumerate(results[:5]):  # 只可视化前5个
        depth_pred = result['depth_pred']
        depth_gt = result['depth_gt']
        valid_mask = result['valid_mask']

        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 预测深度
        pred_vis = np.ma.masked_where(~valid_mask, depth_pred)
        axes[0].imshow(pred_vis, cmap='viridis', vmin=0, vmax=10)
        axes[0].set_title(f'Predicted Depth\n(range: {result["pred_min"]:.2f}-{result["pred_max"]:.2f})')
        axes[0].axis('off')

        # GT深度
        gt_vis = np.ma.masked_where(~valid_mask, depth_gt)
        axes[1].imshow(gt_vis, cmap='viridis', vmin=0, vmax=10)
        axes[1].set_title(f'GT Depth\n(range: {result["gt_min"]:.2f}-{result["gt_max"]:.2f})')
        axes[1].axis('off')

        # 误差图
        error = np.abs(depth_pred - depth_gt)
        error_vis = np.ma.masked_where(~valid_mask, error)
        axes[2].imshow(error_vis, cmap='hot', vmin=0, vmax=2)
        axes[2].set_title(f'Absolute Error\n(mean: {result["mean_rel_error"]:.3f})')
        axes[2].axis('off')

        plt.tight_layout()
        save_path = output_dir / f'depth_comparison_{idx}.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[visualize] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="验证训练后的模型")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--output_dir', type=str, default='try_train/vis_results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"模型验证")
    print(f"{'='*60}")

    # 1. 加载 checkpoint
    ckpt = load_checkpoint(args.checkpoint, args.device)

    # 2. 创建模型并加载权重
    model_args = ckpt['args']
    model = create_head_only_model(
        backbone_name=model_args['backbone'],
        freeze_backbone=model_args['freeze_backbone'],
        img_size=model_args['img_size'],
        train_depth_head=True,
        train_pose_head=model_args['train_pose'],
        num_views=model_args['num_views'],
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(args.device)
    print(f"[model] 加载权重完成")

    # 3. 创建数据集
    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=model_args['num_views'],
        max_dim=model_args['img_size'],
    )

    # 4. 验证推理
    print(f"\n{'='*60}")
    print(f"验证推理")
    print(f"{'='*60}")
    results = validate_model(model, dataset, args.device, args.num_samples)

    # 5. 统计汇总
    if len(results) > 0:
        all_rel_errors = [r['mean_rel_error'] for r in results]
        mean_error = np.mean(all_rel_errors)
        print(f"\n{'='*60}")
        print(f"验证结果汇总")
        print(f"{'='*60}")
        print(f"  - 平均相对误差: {mean_error:.4f}")
        print(f"  - 最小相对误差: {min(all_rel_errors):.4f}")
        print(f"  - 最大相对误差: {max(all_rel_errors):.4f}")

        # 6. 可视化
        visualize_results(results, args.output_dir)

    print(f"\n{'='*60}")
    print(f"验证完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()