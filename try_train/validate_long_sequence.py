"""
长序列验证脚本：Stage1 vs Stage2 vs GT 对比

使用长序列（8-16 views）验证Stage2训练效果：
1. 加载 Stage1 和 Stage2 checkpoint
2. 使用长序列采样（foldback sampler）
3. 计算数值指标对比
4. 生成正确的灰度深度可视化

作者：Claude Code
日期：2026-05-13
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images

from foldback_video_sampler import FoldbackVideoSampler
from head_only_model import create_head_only_model
from replica_dataset import REPLICA_DEPTH_SCALE


# 创建专业的深度colormap（灰度）
def create_depth_colormap():
    """创建深度专用colormap：近处暗色，远处亮色"""
    colors = [
        (0.0, 0.0, 0.2),   # 近处：深蓝
        (0.1, 0.1, 0.3),
        (0.2, 0.2, 0.4),
        (0.3, 0.3, 0.5),
        (0.5, 0.5, 0.6),   # 中间：灰
        (0.7, 0.7, 0.7),
        (0.8, 0.8, 0.8),
        (0.9, 0.9, 0.9),
        (1.0, 1.0, 1.0),   # 远处：白
    ]
    return LinearSegmentedColormap.from_list('depth_gray', colors)


def load_original_model(model_path, device='cuda'):
    """加载原始LingBot-Map模型"""
    model = GCTStream(
        img_size=518, patch_size=14, enable_3d_rope=True,
        max_frame_num=100, kv_cache_sliding_window=64,
        kv_cache_scale_frames=8, kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True, use_sdpa=True,
    )
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    print(f"[Original Model] Loaded from {model_path}")
    return model


def infer_original_model(model, image_paths, device='cuda'):
    """用原始模型推理（518分辨率，streaming模式）"""
    images = load_and_preprocess_images(image_paths, mode="crop", image_size=518, patch_size=14)
    images = images.unsqueeze(0).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(images, num_scale_frames=8)
    depth_pred = predictions['depth']
    if depth_pred.dim() == 5:
        depth_pred = depth_pred.squeeze(-1)
    return depth_pred[0].cpu().float().numpy()  # [S, H, W]


def resize_depth_to_gt(depth_pred, gt_shape):
    """将预测深度resize到GT分辨率"""
    if depth_pred.shape == gt_shape:
        return depth_pred
    return cv2.resize(depth_pred, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)


def align_depth_scale(pred, gt, mask):
    """使用最小二乘法将预测深度对齐到GT的尺度 (scale * pred + shift = gt)"""
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    if len(pred_valid) < 10:
        return pred
    A = np.column_stack([pred_valid, np.ones_like(pred_valid)])
    result = np.linalg.lstsq(A, gt_valid, rcond=None)
    scale, shift = result[0][0], result[0][1]
    aligned = scale * pred + shift
    aligned = np.maximum(aligned, 0)
    return aligned


def load_poses(data_root):
    """加载 traj.txt 文件中的 poses"""
    traj_file = Path(data_root) / "traj.txt"
    poses = {}
    with open(traj_file, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            values = [float(x) for x in line.strip().split()]
            if len(values) != 16:
                continue
            T_c2w = np.array(values, dtype=np.float32).reshape(4, 4)
            poses[idx] = T_c2w
    return poses


def load_rgb_depth(data_root, frame_id):
    """加载RGB和深度图"""
    results_dir = Path(data_root) / "results"

    # RGB
    rgb_path = results_dir / f"frame{frame_id:06d}.jpg"
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Depth (Replica uint16 / 6553.5 -> m; NOT standard mm)
    depth_path = results_dir / f"depth{frame_id:06d}.png"
    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth_m = depth_png.astype(np.float32) / REPLICA_DEPTH_SCALE

    return rgb, depth_m


def align_to_train_size(rgb, depth, max_dim=224):
    """对齐到训练尺寸"""
    H, W = rgb.shape[:2]
    scale = max_dim / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)

    rgb_resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

    # Pad to 14 multiples
    patch_size = 14
    pad_H = (patch_size - new_H % patch_size) % patch_size
    pad_W = (patch_size - new_W % patch_size) % patch_size

    if pad_H > 0 or pad_W > 0:
        rgb_padded = np.pad(rgb_resized, ((0, pad_H), (0, pad_W), (0, 0)), mode='constant', constant_values=0)
        depth_padded = np.pad(depth_resized, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
    else:
        rgb_padded, depth_padded = rgb_resized, depth_resized

    valid_mask = depth_padded > 0

    return rgb_padded, depth_padded, valid_mask


def rotation_matrix_to_quaternion_numpy(R):
    """将 rotation matrix 转换为 quaternion"""
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


def normalize_poses(poses_list):
    """pose 归一化到第一帧坐标系"""
    T_ref = poses_list[0]
    T_ref_inv = np.linalg.inv(T_ref)
    poses_normalized = []
    for T_c2w in poses_list:
        poses_normalized.append(T_ref_inv @ T_c2w)
    return poses_normalized


def load_model_and_infer(checkpoint_path, images, device='cuda'):
    """加载模型并进行推理"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt['args']

    model = create_head_only_model(
        backbone_name=model_args['backbone'],
        freeze_backbone=model_args['freeze_backbone'],
        img_size=model_args['img_size'],
        train_depth_head=True,
        train_pose_head=model_args.get('train_pose', True),
        num_views=images.shape[1],
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(images)

    return predictions, ckpt['iteration']


def validate_long_sequence(
    stage1_checkpoint, stage2_checkpoint, original_model, data_root,
    num_views=8, num_samples=5, max_dim=224, device='cuda'
):
    """长序列验证（含原始模型对比）"""

    print(f"\n{'='*70}")
    print(f"长序列验证 (num_views={num_views}, 含原始模型对比)")
    print(f"{'='*70}")

    # 加载poses
    poses_dict = load_poses(data_root)
    total_frames = len(poses_dict)

    # 创建foldback sampler
    sampler = FoldbackVideoSampler(
        total_frames=total_frames,
        stride_range=(1, 3),
        redraw_stride_after_reverse=True,
        seed=42,
    )

    results_stage1 = []
    results_stage2 = []
    results_orig = []
    results_gt = []

    for sample_idx in range(num_samples):
        # 采样长序列
        start_frame = random.randint(0, total_frames - 1)
        frame_ids = sampler.sample_sequence(num_views, start_frame=start_frame)

        print(f"\n[Sample {sample_idx}] 长序列帧ID: {frame_ids[:5]}...{frame_ids[-1]} ({len(frame_ids)} views)")

        # 加载所有帧的数据
        rgb_list, depth_list, pose_list, mask_list = [], [], [], []

        for frame_id in frame_ids:
            rgb, depth_m = load_rgb_depth(data_root, frame_id)
            rgb_a, depth_a, valid_mask = align_to_train_size(rgb, depth_m, max_dim)

            rgb_list.append(rgb_a)
            depth_list.append(depth_a)
            pose_list.append(poses_dict[frame_id])
            mask_list.append(valid_mask)

        # 归一化poses
        poses_normalized = normalize_poses(pose_list)

        # 转换为tensor
        images = torch.from_numpy(np.stack([
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            for rgb in rgb_list
        ], axis=0)).unsqueeze(0).to(device)  # [1, V, 3, H, W]

        depths_gt = np.stack(depth_list, axis=0)  # [V, H, W]
        poses_gt = np.stack(poses_normalized, axis=0)  # [V, 4, 4]
        masks = np.stack(mask_list, axis=0)  # [V, H, W]

        # Stage1 推理
        pred_s1, iter_s1 = load_model_and_infer(stage1_checkpoint, images, device)
        depth_pred_s1 = pred_s1['depth']
        if depth_pred_s1.dim() == 5:
            depth_pred_s1 = depth_pred_s1.squeeze(-1)
        depth_pred_s1_np = depth_pred_s1[0].cpu().numpy()  # [V, H, W]

        # Stage2 推理
        pred_s2, iter_s2 = load_model_and_infer(stage2_checkpoint, images, device)
        depth_pred_s2 = pred_s2['depth']
        if depth_pred_s2.dim() == 5:
            depth_pred_s2 = depth_pred_s2.squeeze(-1)
        depth_pred_s2_np = depth_pred_s2[0].cpu().numpy()  # [V, H, W]

        # 原始模型推理（518分辨率）
        image_paths = [str(Path(data_root) / "results" / f"frame{fid:06d}.jpg") for fid in frame_ids]
        print("  原始模型推理中...")
        depth_pred_orig = infer_original_model(original_model, image_paths, device)

        # 计算每帧的误差
        for v in range(num_views):
            gt_v = depths_gt[v]
            pred_s1_v = depth_pred_s1_np[v]
            pred_s2_v = depth_pred_s2_np[v]
            mask_v = masks[v]

            # 有效区域的统计
            gt_valid = gt_v[mask_v]
            pred_s1_valid = pred_s1_v[mask_v]
            pred_s2_valid = pred_s2_v[mask_v]

            if len(gt_valid) > 0:
                # 深度误差
                rel_error_s1 = np.abs(pred_s1_valid - gt_valid) / (gt_valid + 1e-3)
                rel_error_s2 = np.abs(pred_s2_valid - gt_valid) / (gt_valid + 1e-3)

                # 原始模型深度：resize到GT分辨率并做尺度对齐
                orig_v = resize_depth_to_gt(depth_pred_orig[v], gt_v.shape)
                orig_v = align_depth_scale(orig_v, gt_v, mask_v)
                orig_valid = orig_v[mask_v]
                rel_error_orig = np.abs(orig_valid - gt_valid) / (gt_valid + 1e-3)

                # 位姿误差
                pose_gt_v = poses_gt[v]
                rot_error_s1 = None
                rot_error_s2 = None
                trans_error_s1 = None
                trans_error_s2 = None

                if v > 0:  # 排除第一帧
                    pose_gt_v = poses_gt[v]

                    # Stage1 pose
                    pose_enc_s1 = pred_s1['pose_enc'][0].cpu().numpy() if 'pose_enc' in pred_s1 else None
                    if pose_enc_s1 is not None:
                        center_s1 = pose_enc_s1[v, :3]
                        quat_s1 = pose_enc_s1[v, 3:7]
                        quat_s1 = quat_s1 / (np.linalg.norm(quat_s1) + 1e-8)

                        center_gt = pose_gt_v[:3, 3]
                        rot_gt = pose_gt_v[:3, :3]
                        quat_gt = rotation_matrix_to_quaternion_numpy(rot_gt)

                        rot_error_s1 = quaternion_geodesic_distance(quat_s1, quat_gt)
                        trans_error_s1 = np.linalg.norm(center_s1 - center_gt)

                    # Stage2 pose
                    pose_enc_s2 = pred_s2['pose_enc'][0].cpu().numpy() if 'pose_enc' in pred_s2 else None
                    if pose_enc_s2 is not None:
                        center_s2 = pose_enc_s2[v, :3]
                        quat_s2 = pose_enc_s2[v, 3:7]
                        quat_s2 = quat_s2 / (np.linalg.norm(quat_s2) + 1e-8)

                        rot_error_s2 = quaternion_geodesic_distance(quat_s2, quat_gt)
                        trans_error_s2 = np.linalg.norm(center_s2 - center_gt)

                results_stage1.append({
                    'sample_idx': sample_idx,
                    'view_idx': v,
                    'frame_id': frame_ids[v],
                    'depth_rel_error': rel_error_s1.mean(),
                    'rot_error_deg': rot_error_s1,
                    'trans_error_m': trans_error_s1,
                    'depth_pred': pred_s1_v,
                    'depth_gt': gt_v,
                    'valid_mask': mask_v,
                })

                results_stage2.append({
                    'sample_idx': sample_idx,
                    'view_idx': v,
                    'frame_id': frame_ids[v],
                    'depth_rel_error': rel_error_s2.mean(),
                    'rot_error_deg': rot_error_s2,
                    'trans_error_m': trans_error_s2,
                    'depth_pred': pred_s2_v,
                    'depth_gt': gt_v,
                    'valid_mask': mask_v,
                })

                results_orig.append({
                    'sample_idx': sample_idx,
                    'view_idx': v,
                    'frame_id': frame_ids[v],
                    'depth_rel_error': rel_error_orig.mean(),
                    'depth_pred': orig_v,
                    'depth_gt': gt_v,
                    'valid_mask': mask_v,
                })

                results_gt.append({
                    'sample_idx': sample_idx,
                    'view_idx': v,
                    'frame_id': frame_ids[v],
                    'depth_gt': gt_v,
                    'valid_mask': mask_v,
                })

                print(f"  View {v}: Depth误差 Orig={rel_error_orig.mean():.3f}, S1={rel_error_s1.mean():.3f}, S2={rel_error_s2.mean():.3f}")
                if rot_error_s1 is not None:
                    print(f"    Pose: Rot S1={rot_error_s1:.2f}°, S2={rot_error_s2:.2f}° | Trans S1={trans_error_s1:.3f}m, S2={trans_error_s2:.3f}m")

    return results_stage1, results_stage2, results_orig, results_gt, iter_s1, iter_s2


def visualize_long_sequence_comparison(
    results_stage1, results_stage2, results_orig, results_gt, output_dir, num_views
):
    """生成长序列可视化对比（4行：GT / Original / Stage1 / Stage2）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 选取一个完整的长序列样本进行可视化
    sample_idx = results_stage1[0]['sample_idx']
    views_in_sample = [r for r in results_stage1 if r['sample_idx'] == sample_idx]

    num_display = min(8, len(views_in_sample))  # 最多显示8帧

    # 创建大图：每帧4行（GT、Original、Stage1、Stage2）
    fig, axes = plt.subplots(4, num_display, figsize=(4 * num_display, 16))

    if num_display == 1:
        axes = axes.reshape(4, 1)

    row_labels = ['GT Depth', 'Original LingBot-Map', 'Stage1 (Head-Only)', 'Stage2 (Streaming)']

    for v_idx in range(num_display):
        result_s1 = views_in_sample[v_idx]
        result_s2 = [r for r in results_stage2 if r['sample_idx'] == sample_idx and r['view_idx'] == result_s1['view_idx']][0]
        result_orig = [r for r in results_orig if r['sample_idx'] == sample_idx and r['view_idx'] == result_s1['view_idx']][0]
        result_gt = [r for r in results_gt if r['sample_idx'] == sample_idx and r['view_idx'] == result_s1['view_idx']][0]

        depth_gt = result_gt['depth_gt']
        depth_pred_orig = result_orig['depth_pred']
        depth_pred_s1 = result_s1['depth_pred']
        depth_pred_s2 = result_s2['depth_pred']
        valid_mask = result_gt['valid_mask']

        gt_max = np.max(depth_gt[valid_mask]) if np.any(valid_mask) else 10
        vmax = max(gt_max, 5)

        # GT深度
        gt_vis = np.ma.masked_where(~valid_mask, depth_gt)
        im_gt = axes[0, v_idx].imshow(gt_vis, cmap='gray', vmin=0, vmax=vmax)
        axes[0, v_idx].set_title(f'GT Frame {result_gt["frame_id"]}')
        axes[0, v_idx].axis('off')
        if v_idx == num_display - 1:
            plt.colorbar(im_gt, ax=axes[0, v_idx], fraction=0.046, pad=0.04, label='m')

        # Original LingBot-Map
        orig_vis = np.ma.masked_where(~valid_mask, depth_pred_orig)
        im_orig = axes[1, v_idx].imshow(orig_vis, cmap='gray', vmin=0, vmax=vmax)
        axes[1, v_idx].set_title(f'Original\n(rel_err: {result_orig["depth_rel_error"]:.3f})')
        axes[1, v_idx].axis('off')
        if v_idx == num_display - 1:
            plt.colorbar(im_orig, ax=axes[1, v_idx], fraction=0.046, pad=0.04, label='m')

        # Stage1预测
        pred_s1_vis = np.ma.masked_where(~valid_mask, depth_pred_s1)
        im_s1 = axes[2, v_idx].imshow(pred_s1_vis, cmap='gray', vmin=0, vmax=vmax)
        axes[2, v_idx].set_title(f'Stage1\n(rel_err: {result_s1["depth_rel_error"]:.3f})')
        axes[2, v_idx].axis('off')
        if v_idx == num_display - 1:
            plt.colorbar(im_s1, ax=axes[2, v_idx], fraction=0.046, pad=0.04, label='m')

        # Stage2预测
        pred_s2_vis = np.ma.masked_where(~valid_mask, depth_pred_s2)
        im_s2 = axes[3, v_idx].imshow(pred_s2_vis, cmap='gray', vmin=0, vmax=vmax)
        axes[3, v_idx].set_title(f'Stage2\n(rel_err: {result_s2["depth_rel_error"]:.3f})')
        axes[3, v_idx].axis('off')
        if v_idx == num_display - 1:
            plt.colorbar(im_s2, ax=axes[3, v_idx], fraction=0.046, pad=0.04, label='m')

    # 添加行标签
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=10, rotation=0, labelpad=80, ha='right', va='center')

    plt.suptitle(f'长序列深度对比 ({num_views} views)', fontsize=14)
    plt.tight_layout()
    save_path = output_dir / f'long_sequence_depth_comparison_{num_views}views.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")

    # 创建误差分布对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    depth_errors_s1 = [r['depth_rel_error'] for r in results_stage1]
    depth_errors_s2 = [r['depth_rel_error'] for r in results_stage2]
    depth_errors_orig = [r['depth_rel_error'] for r in results_orig]

    # 深度误差分布
    axes[0, 0].hist(depth_errors_orig, bins=30, alpha=0.5, label=f'Original (mean={np.mean(depth_errors_orig):.3f})', color='red', edgecolor='black')
    axes[0, 0].hist(depth_errors_s1, bins=30, alpha=0.5, label=f'Stage1 (mean={np.mean(depth_errors_s1):.3f})', color='blue', edgecolor='black')
    axes[0, 0].hist(depth_errors_s2, bins=30, alpha=0.5, label=f'Stage2 (mean={np.mean(depth_errors_s2):.3f})', color='green', edgecolor='black')
    axes[0, 0].set_xlabel('Depth Relative Error')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Depth Error Distribution')
    axes[0, 0].legend(fontsize=8)

    # 深度误差随帧变化
    view_indices = [r['view_idx'] for r in results_stage1]
    axes[0, 1].scatter(view_indices, depth_errors_orig, alpha=0.4, label='Original', color='red')
    axes[0, 1].scatter(view_indices, depth_errors_s1, alpha=0.5, label='Stage1', color='blue')
    axes[0, 1].scatter(view_indices, depth_errors_s2, alpha=0.5, label='Stage2', color='green')
    axes[0, 1].set_xlabel('View Index in Sequence')
    axes[0, 1].set_ylabel('Depth Relative Error')
    axes[0, 1].set_title('Depth Error vs Sequence Position')
    axes[0, 1].legend(fontsize=8)

    # 位姿误差
    pose_s1 = [r for r in results_stage1 if r['rot_error_deg'] is not None]
    pose_s2 = [r for r in results_stage2 if r['rot_error_deg'] is not None]

    if pose_s1 and pose_s2:
        rot_s1 = [r['rot_error_deg'] for r in pose_s1]
        rot_s2 = [r['rot_error_deg'] for r in pose_s2]
        trans_s1 = [r['trans_error_m'] for r in pose_s1]
        trans_s2 = [r['trans_error_m'] for r in pose_s2]

        # Rotation误差
        axes[1, 0].hist(rot_s1, bins=30, alpha=0.7, label='Stage1', color='blue', edgecolor='black')
        axes[1, 0].hist(rot_s2, bins=30, alpha=0.7, label='Stage2', color='green', edgecolor='black')
        axes[1, 0].set_xlabel('Rotation Error (degrees)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Rotation Error (Mean: S1={np.mean(rot_s1):.2f}°, S2={np.mean(rot_s2):.2f}°)')
        axes[1, 0].legend()

        # Translation误差
        axes[1, 1].hist(trans_s1, bins=30, alpha=0.7, label='Stage1', color='blue', edgecolor='black')
        axes[1, 1].hist(trans_s2, bins=30, alpha=0.7, label='Stage2', color='green', edgecolor='black')
        axes[1, 1].set_xlabel('Translation Error (meters)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Translation Error (Mean: S1={np.mean(trans_s1):.3f}m, S2={np.mean(trans_s2):.3f}m)')
        axes[1, 1].legend()

    plt.tight_layout()
    save_path = output_dir / f'long_sequence_error_distribution_{num_views}views.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")

    # 创建单帧详细对比（选取中间帧，4列：GT / Original / Stage1 / Stage2）
    mid_frame_idx = num_views // 2
    result_s1 = [r for r in results_stage1 if r['view_idx'] == mid_frame_idx][0]
    result_s2 = [r for r in results_stage2 if r['view_idx'] == mid_frame_idx][0]
    result_orig = [r for r in results_orig if r['view_idx'] == mid_frame_idx][0]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    depth_gt = result_s1['depth_gt']
    depth_pred_orig = result_orig['depth_pred']
    depth_pred_s1 = result_s1['depth_pred']
    depth_pred_s2 = result_s2['depth_pred']
    valid_mask = result_s1['valid_mask']

    gt_max = np.max(depth_gt[valid_mask]) if np.any(valid_mask) else 10

    # 第一行：4个深度图（灰度）
    im0 = axes[0, 0].imshow(np.ma.masked_where(~valid_mask, depth_gt), cmap='gray', vmin=0, vmax=gt_max)
    axes[0, 0].set_title('GT Depth')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(np.ma.masked_where(~valid_mask, depth_pred_orig), cmap='gray', vmin=0, vmax=gt_max)
    axes[0, 1].set_title(f'Original LingBot-Map\nrel_err={result_orig["depth_rel_error"]:.3f}')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(np.ma.masked_where(~valid_mask, depth_pred_s1), cmap='gray', vmin=0, vmax=gt_max)
    axes[0, 2].set_title(f'Stage1 Depth\nrel_err={result_s1["depth_rel_error"]:.3f}')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[0, 3].imshow(np.ma.masked_where(~valid_mask, depth_pred_s2), cmap='gray', vmin=0, vmax=gt_max)
    axes[0, 3].set_title(f'Stage2 Depth\nrel_err={result_s2["depth_rel_error"]:.3f}')
    plt.colorbar(im3, ax=axes[0, 3], fraction=0.046)

    # 第二行：3个误差图 + 误差对比柱状图
    error_orig = np.abs(depth_pred_orig - depth_gt)
    error_s1 = np.abs(depth_pred_s1 - depth_gt)
    error_s2 = np.abs(depth_pred_s2 - depth_gt)

    axes[1, 0].imshow(np.ma.masked_where(~valid_mask, error_orig), cmap='hot', vmin=0, vmax=2)
    axes[1, 0].set_title('Original Abs Error')

    axes[1, 1].imshow(np.ma.masked_where(~valid_mask, error_s1), cmap='hot', vmin=0, vmax=2)
    axes[1, 1].set_title('Stage1 Abs Error')

    axes[1, 2].imshow(np.ma.masked_where(~valid_mask, error_s2), cmap='hot', vmin=0, vmax=2)
    axes[1, 2].set_title('Stage2 Abs Error')

    # 误差对比柱状图
    models = ['Original', 'Stage1', 'Stage2']
    means = [result_orig['depth_rel_error'], result_s1['depth_rel_error'], result_s2['depth_rel_error']]
    colors = ['red', 'blue', 'green']
    axes[1, 3].bar(models, means, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 3].set_ylabel('Depth Relative Error')
    for j, m in enumerate(means):
        axes[1, 3].text(j, m + 0.01, f'{m:.3f}', ha='center', fontsize=10)

    plt.suptitle(f'Single Frame Detail (view {mid_frame_idx})', fontsize=12)
    plt.tight_layout()
    save_path = output_dir / f'single_frame_detailed_comparison_view{mid_frame_idx}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")


def print_comparison_summary(results_stage1, results_stage2, results_orig, iter_s1, iter_s2, num_views):
    """打印对比总结"""
    depth_errors_s1 = [r['depth_rel_error'] for r in results_stage1]
    depth_errors_s2 = [r['depth_rel_error'] for r in results_stage2]
    depth_errors_orig = [r['depth_rel_error'] for r in results_orig]

    pose_s1 = [r for r in results_stage1 if r['rot_error_deg'] is not None]
    pose_s2 = [r for r in results_stage2 if r['rot_error_deg'] is not None]

    print(f"\n{'='*80}")
    print(f"长序列验证结果汇总 ({num_views} views, 含原始模型对比)")
    print(f"{'='*80}")
    print(f"{'指标':<35} {'Original':<15} {'Stage1':<15} {'Stage2':<15}")
    print(f"{'-'*80}")
    print(f"{'训练 Iterations':<35} {'N/A':<15} {iter_s1:<15} {iter_s2:<15}")
    print(f"{'验证样本数':<35} {len(depth_errors_orig):<15} {len(depth_errors_s1):<15} {len(depth_errors_s2):<15}")
    print(f"{'-'*80}")

    print(f"{'Depth相对误差 (Mean)':<35} {np.mean(depth_errors_orig):.4f}         {np.mean(depth_errors_s1):.4f}         {np.mean(depth_errors_s2):.4f}")
    print(f"{'Depth相对误差 (Std)':<35} {np.std(depth_errors_orig):.4f}         {np.std(depth_errors_s1):.4f}         {np.std(depth_errors_s2):.4f}")
    print(f"{'Depth相对误差 (Min)':<35} {np.min(depth_errors_orig):.4f}         {np.min(depth_errors_s1):.4f}         {np.min(depth_errors_s2):.4f}")
    print(f"{'Depth相对误差 (Max)':<35} {np.max(depth_errors_orig):.4f}         {np.max(depth_errors_s1):.4f}         {np.max(depth_errors_s2):.4f}")
    print(f"{'='*80}")

    if pose_s1 and pose_s2:
        rot_s1 = [r['rot_error_deg'] for r in pose_s1]
        rot_s2 = [r['rot_error_deg'] for r in pose_s2]
        trans_s1 = [r['trans_error_m'] for r in pose_s1]
        trans_s2 = [r['trans_error_m'] for r in pose_s2]

        print(f"{'-'*70}")
        print(f"{'Pose样本数':<35} {len(pose_s1):<20} {len(pose_s2):<20}")
        print(f"{'Rotation误差 (Mean) °':<35} {np.mean(rot_s1):.2f}               {np.mean(rot_s2):.2f}")
        print(f"{'Rotation误差 °':<35} {np.std(rot_s1):.2f}               {np.std(rot_s2):.2f}")
        print(f"{'Translation误差 (Mean) m':<35} {np.mean(trans_s1):.3f}              {np.mean(trans_s2):.3f}")
        print(f"{'Translation误差 m':<35} {np.std(trans_s1):.3f}              {np.std(trans_s2):.3f}")

    print(f"{'='*70}")

    # 改进分析
    s1_vs_orig = (np.mean(depth_errors_orig) - np.mean(depth_errors_s1)) / np.mean(depth_errors_orig) * 100
    s2_vs_orig = (np.mean(depth_errors_orig) - np.mean(depth_errors_s2)) / np.mean(depth_errors_orig) * 100
    s2_vs_s1 = (np.mean(depth_errors_s1) - np.mean(depth_errors_s2)) / np.mean(depth_errors_s1) * 100
    print(f"\n改进分析 (正值=训练后更好):")
    print(f"  Stage1 vs Original: {s1_vs_orig:+.1f}%")
    print(f"  Stage2 vs Original: {s2_vs_orig:+.1f}%")
    print(f"  Stage2 vs Stage1:   {s2_vs_s1:+.1f}%")

    if pose_s1 and pose_s2:
        rot_improvement = ((np.mean(rot_s1) - np.mean(rot_s2)) / np.mean(rot_s1) * 100)
        trans_improvement = ((np.mean(trans_s1) - np.mean(trans_s2)) / np.mean(trans_s1) * 100)
        print(f"  Rotation误差变化: {rot_improvement:.1f}%")
        print(f"  Translation误差变化: {trans_improvement:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="长序列验证：Original vs Stage1 vs Stage2 vs GT")
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='try_train/checkpoints/v2_smoke_test/checkpoint_final.pt')
    parser.add_argument('--stage2_checkpoint', type=str,
                        default='try_train/checkpoints/stage2_smoke_test/checkpoint_stage2_final.pt')
    parser.add_argument('--original_model', type=str,
                        default='/home/shared_files/model_weights/linbo_map/lingbot-map.pt',
                        help='原始LingBot-Map模型权重路径')
    parser.add_argument('--data_root', type=str,
                        default='/home/shared_files/datasets/dovsg/Replica/room0')
    parser.add_argument('--num_views', type=int, default=8, help='长序列views数')
    parser.add_argument('--num_samples', type=int, default=5, help='验证样本数')
    parser.add_argument('--max_dim', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='try_train/vis_long_sequence')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"长序列验证：Original vs Stage1 vs Stage2 vs GT")
    print(f"{'='*70}")
    print(f"Original: {args.original_model}")
    print(f"Stage1: {args.stage1_checkpoint}")
    print(f"Stage2: {args.stage2_checkpoint}")
    print(f"Data: {args.data_root}")
    print(f"Num views: {args.num_views}")
    print(f"Num samples: {args.num_samples}")

    # 加载原始模型
    original_model = load_original_model(args.original_model, args.device)

    # 验证
    results_stage1, results_stage2, results_orig, results_gt, iter_s1, iter_s2 = validate_long_sequence(
        args.stage1_checkpoint, args.stage2_checkpoint, original_model, args.data_root,
        args.num_views, args.num_samples, args.max_dim, args.device
    )

    # 打印总结
    print_comparison_summary(results_stage1, results_stage2, results_orig, iter_s1, iter_s2, args.num_views)

    # 可视化
    visualize_long_sequence_comparison(
        results_stage1, results_stage2, results_orig, results_gt,
        args.output_dir, args.num_views
    )

    print(f"\n{'='*70}")
    print(f"验证完成")
    print(f"{'='*70}")
    print(f"可视化保存至: {args.output_dir}")


if __name__ == '__main__':
    main()