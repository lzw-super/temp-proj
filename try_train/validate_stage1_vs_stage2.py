"""
Stage1 vs Stage2 模型性能对比验证脚本

对比验证两个阶段的训练效果：
1. 加载 Stage1 和 Stage2 checkpoint
2. 对 Replica 数据进行推理
3. 计算数值指标对比：depth相对误差、pose rotation/translation误差
4. 生成对比可视化

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


def load_and_validate_model(checkpoint_path, dataset, device='cuda', num_samples=10):
    """加载checkpoint并验证模型"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    print(f"\n[load_checkpoint] {checkpoint_path}")
    print(f"  - Iteration: {ckpt['iteration']}")
    print(f"  - Loss dict: {ckpt['loss_dict']}")

    # 创建模型
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
    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            images = sample['images'].unsqueeze(0).to(device)
            depths_gt = sample['depths'].to(device)
            poses_gt = sample['poses'].to(device)
            valid_masks = sample['valid_masks'].to(device)

            predictions = model(images)

            depth_pred = predictions['depth']
            if depth_pred.dim() == 5:
                depth_pred = depth_pred.squeeze(-1)

            pose_enc_pred = predictions.get('pose_enc', None)

            depth_pred_np = depth_pred[0].cpu().numpy()
            depths_gt_np = depths_gt.cpu().numpy()
            poses_gt_np = poses_gt.cpu().numpy()

            for v in range(depth_pred_np.shape[0]):
                pred_v = depth_pred_np[v]
                gt_v = depths_gt_np[v]
                pose_gt_v = poses_gt_np[v]
                valid_mask = valid_masks[v].cpu().numpy()

                pred_valid = pred_v[valid_mask]
                gt_valid = gt_v[valid_mask]

                if len(pred_valid) > 0 and len(gt_valid) > 0:
                    rel_error = np.abs(pred_valid - gt_valid) / (gt_valid + 1e-3)
                    mean_rel_error = rel_error.mean()

                    rot_error_deg = None
                    trans_error_m = None

                    if pose_enc_pred is not None and v > 0:
                        pose_enc_np = pose_enc_pred[0].cpu().numpy()
                        center_pred = pose_enc_np[v, :3]
                        quat_pred = pose_enc_np[v, 3:7]
                        quat_pred = quat_pred / (np.linalg.norm(quat_pred) + 1e-8)

                        center_gt = pose_gt_v[:3, 3]
                        rot_gt = pose_gt_v[:3, :3]
                        quat_gt = rotation_matrix_to_quaternion_numpy(rot_gt)

                        rot_error_deg = quaternion_geodesic_distance(quat_pred, quat_gt)
                        trans_error_m = np.linalg.norm(center_pred - center_gt)

                    results.append({
                        'sample_idx': i,
                        'view_idx': v,
                        'depth_rel_error': mean_rel_error,
                        'rot_error_deg': rot_error_deg,
                        'trans_error_m': trans_error_m,
                        'depth_pred': pred_v,
                        'depth_gt': gt_v,
                        'valid_mask': valid_mask,
                    })

    return results, ckpt['iteration']


def compute_statistics(results):
    """计算统计指标"""
    depth_errors = [r['depth_rel_error'] for r in results]

    pose_results = [r for r in results if r['rot_error_deg'] is not None]
    rot_errors = [r['rot_error_deg'] for r in pose_results] if pose_results else []
    trans_errors = [r['trans_error_m'] for r in pose_results] if pose_results else []

    stats = {
        'depth_rel_error_mean': np.mean(depth_errors),
        'depth_rel_error_min': np.min(depth_errors),
        'depth_rel_error_max': np.max(depth_errors),
        'depth_rel_error_std': np.std(depth_errors),
        'rot_error_mean': np.mean(rot_errors) if rot_errors else None,
        'rot_error_std': np.std(rot_errors) if rot_errors else None,
        'trans_error_mean': np.mean(trans_errors) if trans_errors else None,
        'trans_error_std': np.std(trans_errors) if trans_errors else None,
        'num_pose_samples': len(pose_results),
    }

    return stats


def visualize_comparison(results_stage1, results_stage2, stats_stage1, stats_stage2, output_dir):
    """生成对比可视化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 深度误差对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 深度误差分布对比
    depth_errors_s1 = [r['depth_rel_error'] for r in results_stage1]
    depth_errors_s2 = [r['depth_rel_error'] for r in results_stage2]

    axes[0, 0].hist(depth_errors_s1, bins=30, alpha=0.7, label='Stage1', color='blue', edgecolor='black')
    axes[0, 0].hist(depth_errors_s2, bins=30, alpha=0.7, label='Stage2', color='green', edgecolor='black')
    axes[0, 0].set_xlabel('Depth Relative Error')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Depth Error Distribution')
    axes[0, 0].legend()

    # 深度误差对比条形图
    labels = ['Mean', 'Min', 'Max', 'Std']
    s1_vals = [stats_stage1['depth_rel_error_mean'], stats_stage1['depth_rel_error_min'],
               stats_stage1['depth_rel_error_max'], stats_stage1['depth_rel_error_std']]
    s2_vals = [stats_stage2['depth_rel_error_mean'], stats_stage2['depth_rel_error_min'],
               stats_stage2['depth_rel_error_max'], stats_stage2['depth_rel_error_std']]

    x = np.arange(len(labels))
    width = 0.35
    axes[0, 1].bar(x - width/2, s1_vals, width, label='Stage1', color='blue')
    axes[0, 1].bar(x + width/2, s2_vals, width, label='Stage2', color='green')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(labels)
    axes[0, 1].set_ylabel('Depth Relative Error')
    axes[0, 1].set_title('Depth Error Statistics Comparison')
    axes[0, 1].legend()

    # 位姿误差对比（如果有）
    pose_s1 = [r for r in results_stage1 if r['rot_error_deg'] is not None]
    pose_s2 = [r for r in results_stage2 if r['rot_error_deg'] is not None]

    if pose_s1 and pose_s2:
        # Rotation error
        rot_s1 = [r['rot_error_deg'] for r in pose_s1]
        rot_s2 = [r['rot_error_deg'] for r in pose_s2]

        axes[1, 0].hist(rot_s1, bins=30, alpha=0.7, label='Stage1', color='blue', edgecolor='black')
        axes[1, 0].hist(rot_s2, bins=30, alpha=0.7, label='Stage2', color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Rotation Error (degrees)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Rotation Error Distribution')
        axes[1, 0].legend()

        # Translation error
        trans_s1 = [r['trans_error_m'] for r in pose_s1]
        trans_s2 = [r['trans_error_m'] for r in pose_s2]

        axes[1, 1].hist(trans_s1, bins=30, alpha=0.7, label='Stage1', color='blue', edgecolor='black')
        axes[1, 1].hist(trans_s2, bins=30, alpha=0.7, label='Stage2', color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Translation Error (meters)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Translation Error Distribution')
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No pose data', ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, 'No pose data', ha='center', va='center')

    plt.tight_layout()
    save_path = output_dir / 'stage1_vs_stage2_comparison.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")

    # 2. 深度预测可视化对比（选取一个样本）
    if len(results_stage1) > 0 and len(results_stage2) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Stage1
        result_s1 = results_stage1[0]
        depth_pred_s1 = result_s1['depth_pred']
        depth_gt = result_s1['depth_gt']
        valid_mask = result_s1['valid_mask']

        pred_vis = np.ma.masked_where(~valid_mask, depth_pred_s1)
        axes[0, 0].imshow(pred_vis, cmap='viridis', vmin=0, vmax=10)
        axes[0, 0].set_title(f'Stage1 Predicted\n(rel_error: {result_s1["depth_rel_error"]:.3f})')
        axes[0, 0].axis('off')

        gt_vis = np.ma.masked_where(~valid_mask, depth_gt)
        axes[0, 1].imshow(gt_vis, cmap='viridis', vmin=0, vmax=10)
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')

        error_s1 = np.abs(depth_pred_s1 - depth_gt)
        error_vis_s1 = np.ma.masked_where(~valid_mask, error_s1)
        axes[0, 2].imshow(error_vis_s1, cmap='hot', vmin=0, vmax=2)
        axes[0, 2].set_title('Stage1 Error')
        axes[0, 2].axis('off')

        # Stage2
        result_s2 = results_stage2[0]
        depth_pred_s2 = result_s2['depth_pred']

        pred_vis_s2 = np.ma.masked_where(~valid_mask, depth_pred_s2)
        axes[1, 0].imshow(pred_vis_s2, cmap='viridis', vmin=0, vmax=10)
        axes[1, 0].set_title(f'Stage2 Predicted\n(rel_error: {result_s2["depth_rel_error"]:.3f})')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(gt_vis, cmap='viridis', vmin=0, vmax=10)
        axes[1, 1].set_title('Ground Truth')
        axes[1, 1].axis('off')

        error_s2 = np.abs(depth_pred_s2 - depth_gt)
        error_vis_s2 = np.ma.masked_where(~valid_mask, error_s2)
        axes[1, 2].imshow(error_vis_s2, cmap='hot', vmin=0, vmax=2)
        axes[1, 2].set_title('Stage2 Error')
        axes[1, 2].axis('off')

        plt.tight_layout()
        save_path = output_dir / 'depth_visual_comparison.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[visualize] Saved: {save_path}")


def print_comparison_table(stats_stage1, stats_stage2, iter_stage1, iter_stage2):
    """打印对比表格"""
    print(f"\n{'='*70}")
    print(f"Stage1 vs Stage2 性能对比")
    print(f"{'='*70}")
    print(f"{'指标':<30} {'Stage1':<20} {'Stage2':<20} {'改进':<15}")
    print(f"{'-'*70}")

    # Depth
    print(f"{'训练 Iterations':<30} {iter_stage1:<20} {iter_stage2:<20}")
    print(f"{'-'*70}")

    depth_improvement = ((stats_stage1['depth_rel_error_mean'] - stats_stage2['depth_rel_error_mean']) / stats_stage1['depth_rel_error_mean'] * 100)
    print(f"{'Depth相对误差 (Mean)':<30} "
          f"{stats_stage1['depth_rel_error_mean']:.4f}                "
          f"{stats_stage2['depth_rel_error_mean']:.4f}                "
          f"{depth_improvement:.1f}%")

    print(f"{'Depth相对误差 (Std)':<30} "
          f"{stats_stage1['depth_rel_error_std']:.4f}                "
          f"{stats_stage2['depth_rel_error_std']:.4f}")

    print(f"{'Depth相对误差 (Min)':<30} "
          f"{stats_stage1['depth_rel_error_min']:.4f}                "
          f"{stats_stage2['depth_rel_error_min']:.4f}")

    print(f"{'Depth相对误差 (Max)':<30} "
          f"{stats_stage1['depth_rel_error_max']:.4f}                "
          f"{stats_stage2['depth_rel_error_max']:.4f}")

    print(f"{'-'*70}")

    # Pose
    if stats_stage1['rot_error_mean'] is not None and stats_stage2['rot_error_mean'] is not None:
        print(f"{'Pose样本数':<30} "
              f"{stats_stage1['num_pose_samples']:<20} "
              f"{stats_stage2['num_pose_samples']:<20}")

        rot_improvement = ((stats_stage1['rot_error_mean'] - stats_stage2['rot_error_mean']) / stats_stage1['rot_error_mean'] * 100)
        print(f"{'Rotation误差 (Mean)°':<30} "
              f"{stats_stage1['rot_error_mean']:.2f}                 "
              f"{stats_stage2['rot_error_mean']:.2f}                 "
              f"{rot_improvement:.1f}%")

        print(f"{'Rotation误差 (Std)°':<30} "
              f"{stats_stage1['rot_error_std']:.2f}                 "
              f"{stats_stage2['rot_error_std']:.2f}")

        trans_improvement = ((stats_stage1['trans_error_mean'] - stats_stage2['trans_error_mean']) / stats_stage1['trans_error_mean'] * 100)
        print(f"{'Translation误差 (Mean)m':<30} "
              f"{stats_stage1['trans_error_mean']:.3f}                "
              f"{stats_stage2['trans_error_mean']:.3f}                "
              f"{trans_improvement:.1f}%")

        print(f"{'Translation误差 (Std)m':<30} "
              f"{stats_stage1['trans_error_std']:.3f}                "
              f"{stats_stage2['trans_error_std']:.3f}")

    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Stage1 vs Stage2 性能对比验证")
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='try_train/checkpoints/v2_smoke_test/checkpoint_final.pt',
                        help='Stage1 checkpoint路径')
    parser.add_argument('--stage2_checkpoint', type=str,
                        default='try_train/checkpoints/stage2_smoke_test/checkpoint_stage2_final.pt',
                        help='Stage2 checkpoint路径')
    parser.add_argument('--data_root', type=str,
                        default='/home/shared_files/datasets/dovsg/Replica/room0',
                        help='Replica数据根目录')
    parser.add_argument('--num_samples', type=int, default=20, help='验证样本数')
    parser.add_argument('--output_dir', type=str, default='try_train/vis_comparison',
                        help='可视化输出目录')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Stage1 vs Stage2 性能对比验证")
    print(f"{'='*70}")
    print(f"Stage1 checkpoint: {args.stage1_checkpoint}")
    print(f"Stage2 checkpoint: {args.stage2_checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Num samples: {args.num_samples}")

    # 创建验证数据集（不使用增强）
    dataset = ReplicaDataset(
        data_root=args.data_root,
        num_views=2,
        max_dim=224,
        color_jitter_prob=0.0,
        seed=42,
    )

    # 验证 Stage1
    print(f"\n{'='*70}")
    print(f"验证 Stage1 模型")
    print(f"{'='*70}")
    results_stage1, iter_stage1 = load_and_validate_model(
        args.stage1_checkpoint, dataset, args.device, args.num_samples
    )
    stats_stage1 = compute_statistics(results_stage1)

    # 验证 Stage2
    print(f"\n{'='*70}")
    print(f"验证 Stage2 模型")
    print(f"{'='*70}")
    results_stage2, iter_stage2 = load_and_validate_model(
        args.stage2_checkpoint, dataset, args.device, args.num_samples
    )
    stats_stage2 = compute_statistics(results_stage2)

    # 打印对比表格
    print_comparison_table(stats_stage1, stats_stage2, iter_stage1, iter_stage2)

    # 生成可视化
    visualize_comparison(results_stage1, results_stage2, stats_stage1, stats_stage2, args.output_dir)

    print(f"\n{'='*70}")
    print(f"验证完成")
    print(f"{'='*70}")
    print(f"可视化结果保存至: {args.output_dir}")


if __name__ == '__main__':
    main()