"""
论文4.1对齐模型验证脚本

验证训练后的模型是否能够正确预测深度和位姿：
1. 加载 checkpoint（iteration-based）
2. 对 Replica 数据进行推理
3. 检查 depth 和 pose 预测
4. 计算数值指标：depth相对误差、pose rotation/translation误差
5. 可视化 depth 预测和 pose 对比

作者：Claude Code
日期：2026-05-12
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


def rotation_matrix_to_quaternion_numpy(R):
    """将 rotation matrix 转换为 quaternion (numpy版本)"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    qw = np.sqrt(max(trace + 1.0, 1e-8)) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-8)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-8)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-8)
    quat = np.array([qw, qx, qy, qz])
    return quat / (np.linalg.norm(quat) + 1e-8)


def quaternion_geodesic_distance(q1, q2):
    """计算两个 quaternion 之间的 geodesic 距离（度数）"""
    dot = np.abs(np.sum(q1 * q2))
    dot = np.clip(dot, 0.0, 1.0)
    angle = 2.0 * np.arccos(dot)
    return np.degrees(angle)


def load_checkpoint_v2(checkpoint_path, device='cuda'):
    """加载 checkpoint（iteration-based版本）"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"[load_checkpoint] 加载完成: {checkpoint_path}")
    print(f"  - Iteration: {ckpt['iteration']}")
    print(f"  - Loss dict: {ckpt['loss_dict']}")
    return ckpt


def validate_model_v2(model, dataset, device='cuda', num_samples=5):
    """验证模型推理（论文4.1对齐版，包含depth和pose）"""
    model.eval()

    results = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            images = sample['images'].unsqueeze(0).to(device)  # [1, V, 3, H, W]
            depths_gt = sample['depths'].to(device)  # [V, H, W]
            poses_gt = sample['poses'].to(device)  # [V, 4, 4]
            valid_masks = sample['valid_masks'].to(device)

            # 推理
            predictions = model(images)

            # 检查深度预测
            depth_pred = predictions['depth']  # [1, V, H, W, 1] or [1, V, H, W]
            if depth_pred.dim() == 5:
                depth_pred = depth_pred.squeeze(-1)

            # 检查位姿预测
            pose_enc_pred = predictions.get('pose_enc', None)  # [1, V, 9]

            # 统计
            depth_pred_np = depth_pred[0].cpu().numpy()  # [V, H, W]
            depths_gt_np = depths_gt.cpu().numpy()
            poses_gt_np = poses_gt.cpu().numpy()

            for v in range(depth_pred_np.shape[0]):
                pred_v = depth_pred_np[v]
                gt_v = depths_gt_np[v]
                pose_gt_v = poses_gt_np[v]
                valid_mask = valid_masks[v].cpu().numpy()

                # 有效区域的深度统计
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

                    # 位姿误差（如果有pose预测）
                    rot_error_deg = None
                    trans_error_m = None

                    if pose_enc_pred is not None:
                        pose_enc_np = pose_enc_pred[0].cpu().numpy()  # [V, 9]

                        # 提取预测的 center 和 quaternion
                        center_pred = pose_enc_np[v, :3]
                        quat_pred = pose_enc_np[v, 3:7]
                        quat_pred = quat_pred / (np.linalg.norm(quat_pred) + 1e-8)

                        # 提取 GT
                        center_gt = pose_gt_v[:3, 3]
                        rot_gt = pose_gt_v[:3, :3]
                        quat_gt = rotation_matrix_to_quaternion_numpy(rot_gt)

                        # Rotation error (geodesic distance in degrees)
                        # 排除第一帧（第一帧GT是identity，预测也应接近identity）
                        if v > 0:
                            rot_error_deg = quaternion_geodesic_distance(quat_pred, quat_gt)
                            trans_error_m = np.linalg.norm(center_pred - center_gt)

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
                        'rot_error_deg': rot_error_deg,
                        'trans_error_m': trans_error_m,
                        'depth_pred': pred_v,
                        'depth_gt': gt_v,
                        'valid_mask': valid_mask,
                        'pose_gt': pose_gt_v,
                        'pose_pred_enc': pose_enc_np[v] if pose_enc_pred is not None else None,
                    })

                    print(f"[Sample {i}, View {v}]")
                    print(f"  Depth: pred=[{pred_min:.3f}, {pred_max:.3f}], gt=[{gt_min:.3f}, {gt_max:.3f}]")
                    print(f"  Depth Rel Error: {mean_rel_error:.3f}")
                    if rot_error_deg is not None:
                        print(f"  Pose: rot_error={rot_error_deg:.2f}deg, trans_error={trans_error_m:.3f}m")

    return results


def visualize_results_v2(results, output_dir):
    """可视化深度和位姿预测结果（使用灰度colormap）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算GT深度的最大值用于统一可视化范围
    gt_max = max([r['gt_max'] for r in results[:5]])
    vmax = max(gt_max, 10)  # 至少10m的范围

    for idx, result in enumerate(results[:5]):  # 只可视化前5个
        depth_pred = result['depth_pred']
        depth_gt = result['depth_gt']
        valid_mask = result['valid_mask']

        # 创建对比图（使用灰度colormap）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 预测深度（灰度）
        pred_vis = np.ma.masked_where(~valid_mask, depth_pred)
        im_pred = axes[0].imshow(pred_vis, cmap='gray', vmin=0, vmax=vmax)
        axes[0].set_title(f'Predicted Depth\n(range: {result["pred_min"]:.2f}-{result["pred_max"]:.2f}m)')
        axes[0].axis('off')
        plt.colorbar(im_pred, ax=axes[0], fraction=0.046, pad=0.04)

        # GT深度（灰度）
        gt_vis = np.ma.masked_where(~valid_mask, depth_gt)
        im_gt = axes[1].imshow(gt_vis, cmap='gray', vmin=0, vmax=vmax)
        axes[1].set_title(f'GT Depth\n(range: {result["gt_min"]:.2f}-{result["gt_max"]:.2f}m)')
        axes[1].axis('off')
        plt.colorbar(im_gt, ax=axes[1], fraction=0.046, pad=0.04)

        # 误差图（保持hot colormap，因为是误差图）
        error = np.abs(depth_pred - depth_gt)
        error_vis = np.ma.masked_where(~valid_mask, error)
        im_err = axes[2].imshow(error_vis, cmap='hot', vmin=0, vmax=2)
        axes[2].set_title(f'Absolute Error\n(rel_error: {result["mean_rel_error"]:.3f})')
        axes[2].axis('off')
        plt.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = output_dir / f'depth_comparison_v2_{idx}.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[visualize] Saved: {save_path}")

    # 创建位姿误差统计图
    pose_results = [r for r in results if r['rot_error_deg'] is not None]
    if len(pose_results) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Rotation error
        rot_errors = [r['rot_error_deg'] for r in pose_results]
        axes[0].hist(rot_errors, bins=20, edgecolor='black')
        axes[0].set_xlabel('Rotation Error (degrees)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Rotation Error Distribution\n(mean: {np.mean(rot_errors):.2f}deg)')

        # Translation error
        trans_errors = [r['trans_error_m'] for r in pose_results]
        axes[1].hist(trans_errors, bins=20, edgecolor='black')
        axes[1].set_xlabel('Translation Error (meters)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Translation Error Distribution\n(mean: {np.mean(trans_errors):.3f}m)')

        plt.tight_layout()
        save_path = output_dir / 'pose_error_distribution.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[visualize] Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="验证论文4.1对齐训练后的模型")
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint文件路径')
    parser.add_argument('--data_root', type=str, required=True, help='Replica数据根目录')
    parser.add_argument('--num_samples', type=int, default=10, help='验证样本数')
    parser.add_argument('--output_dir', type=str, default='try_train/vis_results_v2', help='可视化输出目录')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"论文4.1对齐模型验证")
    print(f"{'='*60}")

    # 1. 加载 checkpoint
    ckpt = load_checkpoint_v2(args.checkpoint, args.device)

    # 2. 创建模型并加载权重
    model_args = ckpt['args']
    model = create_head_only_model(
        backbone_name=model_args['backbone'],
        freeze_backbone=model_args['freeze_backbone'],
        img_size=model_args['img_size'],
        train_depth_head=True,
        train_pose_head=model_args.get('train_pose', True),
        num_views=2,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(args.device)
    print(f"[model] 加载权重完成")

    # 3. 创建数据集
    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=model_args.get('batch_size', 1) * 2,  # 使用默认2 views
        max_dim=model_args['img_size'],
        color_jitter_prob=0.0,  # 验证时不使用增强
    )

    # 4. 验证推理
    print(f"\n{'='*60}")
    print(f"验证推理")
    print(f"{'='*60}")
    results = validate_model_v2(model, dataset, args.device, args.num_samples)

    # 5. 统计汇总
    if len(results) > 0:
        all_rel_errors = [r['mean_rel_error'] for r in results]
        mean_depth_error = np.mean(all_rel_errors)

        pose_results = [r for r in results if r['rot_error_deg'] is not None]

        print(f"\n{'='*60}")
        print(f"验证结果汇总")
        print(f"{'='*60}")
        print(f"  Depth 相对误差:")
        print(f"    - 平均: {mean_depth_error:.4f}")
        print(f"    - 最小: {min(all_rel_errors):.4f}")
        print(f"    - 最大: {max(all_rel_errors):.4f}")

        if len(pose_results) > 0:
            rot_errors = [r['rot_error_deg'] for r in pose_results]
            trans_errors = [r['trans_error_m'] for r in pose_results]
            print(f"  Pose 误差:")
            print(f"    - Rotation: 平均 {np.mean(rot_errors):.2f}deg, 范围 [{np.min(rot_errors):.2f}, {np.max(rot_errors):.2f}]")
            print(f"    - Translation: 平均 {np.mean(trans_errors):.3f}m, 范围 [{np.min(trans_errors):.3f}, {np.max(trans_errors):.3f}]")

        # 6. 可视化
        visualize_results_v2(results, args.output_dir)

    print(f"\n{'='*60}")
    print(f"验证完成")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()