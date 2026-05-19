"""
原始LingBot-Map模型 vs Stage1 vs Stage2 vs GT 深度对比验证

对比训练前后的深度预测质量：
1. 加载原始LingBot-Map预训练模型 (GCTStream, 518分辨率)
2. 加载Stage1/Stage2训练模型 (HeadOnlyModel, 224分辨率)
3. 对相同的Replica数据进行推理
4. 生成4行对比可视化 (GT / Original / Stage1 / Stage2)

作者：Claude Code
日期：2026-05-15
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.models.gct_stream import GCTStream
from lingbot_map.utils.load_fn import load_and_preprocess_images

sys.path.insert(0, str(Path(__file__).parent))

from foldback_video_sampler import FoldbackVideoSampler
from head_only_model import create_head_only_model


def load_poses(data_root):
    """加载 traj.txt 中的 poses"""
    traj_file = Path(data_root) / "traj.txt"
    poses = {}
    with open(traj_file, 'r') as f:
        for idx, line in enumerate(f):
            values = [float(x) for x in line.strip().split()]
            if len(values) != 16:
                continue
            T_c2w = np.array(values, dtype=np.float32).reshape(4, 4)
            poses[idx] = T_c2w
    return poses


def load_rgb_depth(data_root, frame_id):
    """加载RGB和深度图"""
    results_dir = Path(data_root) / "results"
    rgb_path = results_dir / f"frame{frame_id:06d}.jpg"
    rgb = cv2.imread(str(rgb_path))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth_path = results_dir / f"depth{frame_id:06d}.png"
    depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    depth_m = depth_png.astype(np.float32) / 1000.0
    return rgb, depth_m


def align_to_train_size(rgb, depth, max_dim=224):
    """对齐到训练尺寸"""
    H, W = rgb.shape[:2]
    scale = max_dim / max(H, W)
    new_H, new_W = int(H * scale), int(W * scale)
    rgb_resized = cv2.resize(rgb, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    patch_size = 14
    pad_H = (patch_size - new_H % patch_size) % patch_size
    pad_W = (patch_size - new_W % patch_size) % patch_size
    if pad_H > 0 or pad_W > 0:
        rgb_padded = np.pad(rgb_resized, ((0, pad_H), (0, pad_W), (0, 0)), mode='constant')
        depth_padded = np.pad(depth_resized, ((0, pad_H), (0, pad_W)), mode='constant')
    else:
        rgb_padded, depth_padded = rgb_resized, depth_resized
    valid_mask = depth_padded > 0
    return rgb_padded, depth_padded, valid_mask


def load_original_model(model_path, device='cuda'):
    """加载原始LingBot-Map模型"""
    model = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=100,
        kv_cache_sliding_window=64,
        kv_cache_scale_frames=8,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
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
    images = images.unsqueeze(0).to(device)  # [1, S, 3, H, W]

    # 必须使用autocast混合精度推理，与demo.py一致
    # 原始模型在bfloat16下训练，float32推理会导致深度预测接近0（纯黑）
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        predictions = model.inference_streaming(images, num_scale_frames=8)

    depth_pred = predictions['depth']  # [1, S, H, W, 1]
    if depth_pred.dim() == 5:
        depth_pred = depth_pred.squeeze(-1)  # [1, S, H, W]

    return depth_pred[0].cpu().float().numpy()  # [S, H, W]


def load_trained_model(checkpoint_path, num_views, device='cuda'):
    """加载Stage1/Stage2训练模型"""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt['args']
    model = create_head_only_model(
        backbone_name=model_args['backbone'],
        freeze_backbone=model_args['freeze_backbone'],
        img_size=model_args['img_size'],
        train_depth_head=True,
        train_pose_head=model_args.get('train_pose', True),
        num_views=num_views,
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device).eval()
    return model, ckpt['iteration']


def infer_trained_model(model, images_tensor, device='cuda'):
    """用训练模型推理（224分辨率）"""
    with torch.no_grad():
        predictions = model(images_tensor)
    depth_pred = predictions['depth']
    if depth_pred.dim() == 5:
        depth_pred = depth_pred.squeeze(-1)
    return depth_pred[0].cpu().numpy()  # [V, H, W]


def resize_depth_to_gt(depth_pred, gt_shape):
    """将预测深度resize到GT分辨率"""
    if depth_pred.shape == gt_shape:
        return depth_pred
    return cv2.resize(depth_pred, (gt_shape[1], gt_shape[0]), interpolation=cv2.INTER_LINEAR)


def align_depth_scale(pred, gt, mask):
    """使用最小二乘法将预测深度对齐到GT的尺度 (scale * pred + shift = gt)

    原始LingBot-Map模型输出归一化深度（~0.38-1.58），而非真实米制深度，
    因此需要做线性对齐才能和GT在同一尺度下比较。
    """
    pred_valid = pred[mask]
    gt_valid = gt[mask]
    if len(pred_valid) < 10:
        return pred
    # 最小二乘: gt = scale * pred + shift
    A = np.column_stack([pred_valid, np.ones_like(pred_valid)])
    result = np.linalg.lstsq(A, gt_valid, rcond=None)
    scale, shift = result[0][0], result[0][1]
    aligned = scale * pred + shift
    # 深度不能为负
    aligned = np.maximum(aligned, 0)
    return aligned


def normalize_poses(poses_list):
    """pose归一化到第一帧坐标系"""
    T_ref = poses_list[0]
    T_ref_inv = np.linalg.inv(T_ref)
    return [T_ref_inv @ T_c2w for T_c2w in poses_list]


def rotation_matrix_to_quaternion_numpy(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    qw = np.sqrt(max(trace + 1.0, 1e-8)) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw + 1e-8)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw + 1e-8)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw + 1e-8)
    quat = np.array([qw, qx, qy, qz])
    return quat / (np.linalg.norm(quat) + 1e-8)


def quaternion_geodesic_distance(q1, q2):
    dot = np.abs(np.sum(q1 * q2))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def validate_with_original(
    original_model_path, stage1_checkpoint, stage2_checkpoint,
    data_root, num_views=8, num_samples=5, max_dim=224, device='cuda'
):
    """4模型对比验证"""
    print(f"\n{'='*70}")
    print(f"4模型深度对比验证 (num_views={num_views})")
    print(f"{'='*70}")

    # 加载原始模型
    original_model = load_original_model(original_model_path, device)

    # 加载Stage1和Stage2模型
    model_s1, iter_s1 = load_trained_model(stage1_checkpoint, num_views, device)
    model_s2, iter_s2 = load_trained_model(stage2_checkpoint, num_views, device)

    # 加载poses
    poses_dict = load_poses(data_root)
    total_frames = len(poses_dict)

    sampler = FoldbackVideoSampler(
        total_frames=total_frames, stride_range=(1, 3),
        redraw_stride_after_reverse=True, seed=42,
    )

    results = []

    for sample_idx in range(num_samples):
        start_frame = random.randint(0, total_frames - 1)
        frame_ids = sampler.sample_sequence(num_views, start_frame=start_frame)
        print(f"\n[Sample {sample_idx}] 帧ID: {frame_ids[:5]}...{frame_ids[-1]} ({len(frame_ids)} views)")

        # 加载数据
        rgb_list, depth_list, pose_list, mask_list = [], [], [], []
        image_paths = []

        for frame_id in frame_ids:
            rgb, depth_m = load_rgb_depth(data_root, frame_id)
            rgb_a, depth_a, valid_mask = align_to_train_size(rgb, depth_m, max_dim)
            rgb_list.append(rgb_a)
            depth_list.append(depth_a)
            pose_list.append(poses_dict[frame_id])
            mask_list.append(valid_mask)

            # 保存临时图片用于原始模型推理
            results_dir = Path(data_root) / "results"
            image_paths.append(str(results_dir / f"frame{frame_id:06d}.jpg"))

        depths_gt = np.stack(depth_list, axis=0)
        masks = np.stack(mask_list, axis=0)

        # 训练模型输入 (224分辨率)
        images_tensor = torch.from_numpy(np.stack([
            rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            for rgb in rgb_list
        ], axis=0)).unsqueeze(0).to(device)

        # 原始模型推理 (518分辨率)
        print("  原始模型推理中...")
        depth_pred_orig = infer_original_model(original_model, image_paths, device)

        # Stage1推理
        print("  Stage1推理中...")
        depth_pred_s1 = infer_trained_model(model_s1, images_tensor, device)

        # Stage2推理
        print("  Stage2推理中...")
        depth_pred_s2 = infer_trained_model(model_s2, images_tensor, device)

        # 逐帧比较
        for v in range(num_views):
            gt_v = depths_gt[v]
            mask_v = masks[v]
            gt_valid = gt_v[mask_v]

            if len(gt_valid) == 0:
                continue

            # 将原始模型预测resize到GT分辨率，并进行尺度对齐
            # 原始LingBot-Map输出归一化深度，需要线性对齐到GT的米制尺度
            orig_v = resize_depth_to_gt(depth_pred_orig[v], gt_v.shape)
            orig_v = align_depth_scale(orig_v, gt_v, mask_v)
            s1_v = resize_depth_to_gt(depth_pred_s1[v], gt_v.shape)
            s2_v = resize_depth_to_gt(depth_pred_s2[v], gt_v.shape)

            orig_valid = orig_v[mask_v]
            s1_valid = s1_v[mask_v]
            s2_valid = s2_v[mask_v]

            rel_err_orig = np.abs(orig_valid - gt_valid) / (gt_valid + 1e-3)
            rel_err_s1 = np.abs(s1_valid - gt_valid) / (gt_valid + 1e-3)
            rel_err_s2 = np.abs(s2_valid - gt_valid) / (gt_valid + 1e-3)

            results.append({
                'sample_idx': sample_idx,
                'view_idx': v,
                'frame_id': frame_ids[v],
                'depth_orig': orig_v,
                'depth_s1': s1_v,
                'depth_s2': s2_v,
                'depth_gt': gt_v,
                'valid_mask': mask_v,
                'rel_err_orig': rel_err_orig.mean(),
                'rel_err_s1': rel_err_s1.mean(),
                'rel_err_s2': rel_err_s2.mean(),
                'abs_err_orig': np.abs(orig_valid - gt_valid).mean(),
                'abs_err_s1': np.abs(s1_valid - gt_valid).mean(),
                'abs_err_s2': np.abs(s2_valid - gt_valid).mean(),
            })

            print(f"  View {v}: RelErr Orig={rel_err_orig.mean():.3f}, S1={rel_err_s1.mean():.3f}, S2={rel_err_s2.mean():.3f}")

    return results, iter_s1, iter_s2


def visualize_comparison(results, output_dir, num_views):
    """生成4行对比可视化"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 长序列4行对比 (GT / Original / Stage1 / Stage2)
    sample_idx = results[0]['sample_idx']
    views = [r for r in results if r['sample_idx'] == sample_idx]
    num_display = min(8, len(views))

    fig, axes = plt.subplots(4, num_display, figsize=(4 * num_display, 16))
    if num_display == 1:
        axes = axes.reshape(4, 1)

    gt_max = max(np.max(r['depth_gt'][r['valid_mask']]) for r in views[:num_display] if np.any(r['valid_mask']))
    vmax = max(gt_max, 5)

    row_labels = ['GT Depth', 'Original LingBot-Map', 'Stage1 (Head-Only)', 'Stage2 (Streaming)']
    depth_keys = ['depth_gt', 'depth_orig', 'depth_s1', 'depth_s2']
    err_keys = [None, 'rel_err_orig', 'rel_err_s1', 'rel_err_s2']

    for row, (label, dkey, ekey) in enumerate(zip(row_labels, depth_keys, err_keys)):
        for col in range(num_display):
            r = views[col]
            depth = r[dkey]
            mask = r['valid_mask']
            vis = np.ma.masked_where(~mask, depth)
            im = axes[row, col].imshow(vis, cmap='gray', vmin=0, vmax=vmax)
            axes[row, col].axis('off')

            if ekey is not None:
                axes[row, col].set_title(f'{label}\nrel_err={r[ekey]:.3f}', fontsize=8)
            else:
                axes[row, col].set_title(f'{label}\nFrame {r["frame_id"]}', fontsize=8)

            if col == num_display - 1:
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04, label='m')

    plt.suptitle(f'Depth Comparison: GT vs Original vs Stage1 vs Stage2 ({num_views} views)', fontsize=14)
    plt.tight_layout()
    save_path = output_dir / f'4way_depth_comparison_{num_views}views.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")

    # 2. 单帧详细对比 (2x4 grid)
    mid_idx = num_views // 2
    mid_result = [r for r in results if r['view_idx'] == mid_idx and r['sample_idx'] == sample_idx][0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    mask = mid_result['valid_mask']
    gt = mid_result['depth_gt']
    vmax_single = max(np.max(gt[mask]), 5)

    # 第一行：4个深度图
    for col, (label, key) in enumerate(zip(
        ['GT', 'Original LingBot-Map', 'Stage1 (Head-Only)', 'Stage2 (Streaming)'],
        ['depth_gt', 'depth_orig', 'depth_s1', 'depth_s2']
    )):
        vis = np.ma.masked_where(~mask, mid_result[key])
        im = axes[0, col].imshow(vis, cmap='gray', vmin=0, vmax=vmax_single)
        err_key = [None, 'rel_err_orig', 'rel_err_s1', 'rel_err_s2'][col]
        title = f'{label}\nrel_err={mid_result[err_key]:.3f}' if err_key else f'{label}'
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].axis('off')
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)

    # 第二行：3个误差图 + 误差对比
    for col, (label, key) in enumerate(zip(
        ['Original Abs Error', 'Stage1 Abs Error', 'Stage2 Abs Error'],
        ['depth_orig', 'depth_s1', 'depth_s2']
    )):
        error = np.abs(mid_result[key] - gt)
        vis = np.ma.masked_where(~mask, error)
        axes[1, col].imshow(vis, cmap='hot', vmin=0, vmax=2)
        axes[1, col].set_title(label, fontsize=9)
        axes[1, col].axis('off')

    # 训练前后改进对比
    err_orig = np.abs(mid_result['depth_orig'] - gt)
    err_s1 = np.abs(mid_result['depth_s1'] - gt)
    improvement = err_orig - err_s1  # 正值=训练后更好
    vis = np.ma.masked_where(~mask, improvement)
    axes[1, 3].imshow(vis, cmap='RdBu', vmin=-1, vmax=1)
    axes[1, 3].set_title('Orig - Stage1 Error\n(Blue: trained better)', fontsize=9)
    axes[1, 3].axis('off')

    plt.tight_layout()
    save_path = output_dir / f'single_frame_4way_detail_view{mid_idx}.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")

    # 3. 误差分布柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    rel_orig = [r['rel_err_orig'] for r in results]
    rel_s1 = [r['rel_err_s1'] for r in results]
    rel_s2 = [r['rel_err_s2'] for r in results]

    # 深度相对误差分布
    axes[0].hist(rel_orig, bins=25, alpha=0.6, label=f'Original (mean={np.mean(rel_orig):.3f})', color='red', edgecolor='black')
    axes[0].hist(rel_s1, bins=25, alpha=0.6, label=f'Stage1 (mean={np.mean(rel_s1):.3f})', color='blue', edgecolor='black')
    axes[0].hist(rel_s2, bins=25, alpha=0.6, label=f'Stage2 (mean={np.mean(rel_s2):.3f})', color='green', edgecolor='black')
    axes[0].set_xlabel('Depth Relative Error')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Depth Error Distribution')
    axes[0].legend(fontsize=8)

    # 模型间平均误差柱状图
    models = ['Original', 'Stage1', 'Stage2']
    means = [np.mean(rel_orig), np.mean(rel_s1), np.mean(rel_s2)]
    stds = [np.std(rel_orig), np.std(rel_s1), np.std(rel_s2)]
    colors = ['red', 'blue', 'green']
    bars = axes[1].bar(models, means, yerr=stds, color=colors, alpha=0.7, edgecolor='black', capsize=5)
    axes[1].set_ylabel('Mean Depth Relative Error')
    axes[1].set_title('Mean Depth Error Comparison')
    for bar, mean in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

    # 绝对误差分布
    abs_orig = [r['abs_err_orig'] for r in results]
    abs_s1 = [r['abs_err_s1'] for r in results]
    abs_s2 = [r['abs_err_s2'] for r in results]
    axes[2].hist(abs_orig, bins=25, alpha=0.6, label=f'Original (mean={np.mean(abs_orig):.3f})', color='red', edgecolor='black')
    axes[2].hist(abs_s1, bins=25, alpha=0.6, label=f'Stage1 (mean={np.mean(abs_s1):.3f})', color='blue', edgecolor='black')
    axes[2].hist(abs_s2, bins=25, alpha=0.6, label=f'Stage2 (mean={np.mean(abs_s2):.3f})', color='green', edgecolor='black')
    axes[2].set_xlabel('Depth Absolute Error (m)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Depth Absolute Error Distribution')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    save_path = output_dir / f'4way_error_distribution_{num_views}views.png'
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[visualize] Saved: {save_path}")


def print_summary(results, iter_s1, iter_s2, num_views):
    """打印汇总"""
    rel_orig = [r['rel_err_orig'] for r in results]
    rel_s1 = [r['rel_err_s1'] for r in results]
    rel_s2 = [r['rel_err_s2'] for r in results]
    abs_orig = [r['abs_err_orig'] for r in results]
    abs_s1 = [r['abs_err_s1'] for r in results]
    abs_s2 = [r['abs_err_s2'] for r in results]

    print(f"\n{'='*70}")
    print(f"4模型深度对比汇总 ({num_views} views)")
    print(f"{'='*70}")
    print(f"{'指标':<30} {'Original':<15} {'Stage1':<15} {'Stage2':<15}")
    print(f"{'-'*70}")
    print(f"{'Rel Error Mean':<30} {np.mean(rel_orig):.4f}         {np.mean(rel_s1):.4f}         {np.mean(rel_s2):.4f}")
    print(f"{'Rel Error Std':<30} {np.std(rel_orig):.4f}         {np.std(rel_s1):.4f}         {np.std(rel_s2):.4f}")
    print(f"{'Rel Error Min':<30} {np.min(rel_orig):.4f}         {np.min(rel_s1):.4f}         {np.min(rel_s2):.4f}")
    print(f"{'Rel Error Max':<30} {np.max(rel_orig):.4f}         {np.max(rel_s1):.4f}         {np.max(rel_s2):.4f}")
    print(f"{'-'*70}")
    print(f"{'Abs Error Mean (m)':<30} {np.mean(abs_orig):.4f}         {np.mean(abs_s1):.4f}         {np.mean(abs_s2):.4f}")
    print(f"{'Abs Error Std (m)':<30} {np.std(abs_orig):.4f}         {np.std(abs_s1):.4f}         {np.std(abs_s2):.4f}")
    print(f"{'='*70}")

    # 改进分析
    s1_vs_orig = (np.mean(rel_orig) - np.mean(rel_s1)) / np.mean(rel_orig) * 100
    s2_vs_orig = (np.mean(rel_orig) - np.mean(rel_s2)) / np.mean(rel_orig) * 100
    s2_vs_s1 = (np.mean(rel_s1) - np.mean(rel_s2)) / np.mean(rel_s1) * 100
    print(f"\n改进分析 (正值=训练后更好):")
    print(f"  Stage1 vs Original: {s1_vs_orig:+.1f}%")
    print(f"  Stage2 vs Original: {s2_vs_orig:+.1f}%")
    print(f"  Stage2 vs Stage1:   {s2_vs_s1:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="4模型深度对比：Original vs Stage1 vs Stage2 vs GT")
    parser.add_argument('--original_model', type=str,
                        default='/home/shared_files/model_weights/linbo_map/lingbot-map.pt')
    parser.add_argument('--stage1_checkpoint', type=str,
                        default='try_train/checkpoints/v2_smoke_test/checkpoint_final.pt')
    parser.add_argument('--stage2_checkpoint', type=str,
                        default='try_train/checkpoints/stage2_smoke_test/checkpoint_stage2_final.pt')
    parser.add_argument('--data_root', type=str,
                        default='/home/shared_files/datasets/dovsg/Replica/room0')
    parser.add_argument('--num_views', type=int, default=8)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--max_dim', type=int, default=224)
    parser.add_argument('--output_dir', type=str, default='try_train/vis_with_original')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"4模型深度对比验证")
    print(f"{'='*70}")
    print(f"Original Model: {args.original_model}")
    print(f"Stage1: {args.stage1_checkpoint}")
    print(f"Stage2: {args.stage2_checkpoint}")
    print(f"Data: {args.data_root}")

    results, iter_s1, iter_s2 = validate_with_original(
        args.original_model, args.stage1_checkpoint, args.stage2_checkpoint,
        args.data_root, args.num_views, args.num_samples, args.max_dim, args.device
    )

    print_summary(results, iter_s1, iter_s2, args.num_views)
    visualize_comparison(results, args.output_dir, args.num_views)

    print(f"\n可视化保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
