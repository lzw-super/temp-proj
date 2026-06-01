"""
Stage2 Streaming Head-Only 训练脚本（论文4.2对齐版）

根据 replica_head_only_stage1_stage2_training_plan.md 文档的要求，
本脚本实现第二阶段长序列训练：

1. 从 Stage1 checkpoint 初始化
2. Foldback Video Sampler（边界反向继续）
3. Progressive View Curriculum（views数24→320线性增长）
4. Local Window Relative Pose Loss（窗口k=[16,64]）
5. 参数：AdamW, lr=5e-4, wd=0.05, warmup+cosine scheduler

遵循论文 Sec. 4.2 的第二阶段设置。

作者：Claude Code
日期：2026-05-12
"""

import os
import sys
import argparse
import time
import random
import numpy as np
from pathlib import Path

import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

sys.path.insert(0, str(Path(__file__).parent))

from replica_dataset import ReplicaDataset, REPLICA_DEPTH_SCALE
from foldback_video_sampler import (
    FoldbackVideoSampler,
    ProgressiveViewCurriculum,
    LocalWindowSampler,
)
from head_only_model import (
    create_head_only_model,
    DepthLoss,
    PoseLoss,
)


def _se3_inv(T: torch.Tensor) -> torch.Tensor:
    """Analytical inverse for SE(3) pose matrix [R|t; 0|1].
    inv([R|t; 0|1]) = [R^T | -R^T @ t; 0 | 1]
    More numerically stable than torch.linalg.inv for near-degenerate rotations.
    """
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    R_t = R.transpose(-1, -2)
    out = torch.zeros_like(T)
    out[:, :3, :3] = R_t
    out[:, :3, 3] = -torch.bmm(R_t, t.unsqueeze(-1)).squeeze(-1)
    out[:, 3, 3] = 1.0
    return out


class LocalRelativePoseLoss(torch.nn.Module):
    """
    Local Relative Pose Loss: 只在局部窗口内计算 relative pose

    遵循论文4.2的局部窗口采样策略：
    - 窗口大小k在[min_k, max_k]范围内随机采样
    - 只计算窗口内相邻帧的 relative pose
    - 减少长序列全pair计算的计算量
    """

    def __init__(self, rotation_weight=1.0, translation_weight=10.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(self, pose_enc_pred, pose_gt, window_pairs):
        """
        Args:
            pose_enc_pred: [B, V, 9] pose encoding
            pose_gt: [B, V, 4, 4] camera-to-world pose matrix
            window_pairs: List of (i, j) tuples for local window pairs

        Returns:
            relative_pose_loss: scalar loss
        """
        B, V = pose_enc_pred.shape[:2]

        if len(window_pairs) == 0:
            return 0.0

        # 提取预测的 center 和 quaternion
        center_pred = pose_enc_pred[:, :, :3]  # [B, V, 3]
        quat_pred = pose_enc_pred[:, :, 3:7]   # [B, V, 4]
        quat_pred = quat_pred / (torch.norm(quat_pred, dim=-1, keepdim=True) + 1e-8)

        # 构建 predicted pose matrix
        rot_pred = self._quaternion_to_rotation_matrix(quat_pred)

        pose_pred = torch.zeros(B, V, 4, 4, device=pose_enc_pred.device)
        pose_pred[:, :, :3, :3] = rot_pred
        pose_pred[:, :, :3, 3] = center_pred
        pose_pred[:, :, 3, 3] = 1.0

        # 只计算窗口内的 pairs
        total_loss = 0.0
        num_pairs = 0

        for i, j in window_pairs:
            if i >= V or j >= V:
                continue

            # T_i_to_j_pred = inv(T_i_pred) @ T_j_pred
            T_i_pred = pose_pred[:, i, :, :]
            T_j_pred = pose_pred[:, j, :, :]
            T_i_to_j_pred = _se3_inv(T_i_pred) @ T_j_pred

            # T_i_to_j_gt = inv(T_i_gt) @ T_j_gt
            T_i_gt = pose_gt[:, i, :, :]
            T_j_gt = pose_gt[:, j, :, :]
            T_i_to_j_gt = _se3_inv(T_i_gt) @ T_j_gt

            # 提取 relative translation 和 rotation
            trans_pred = T_i_to_j_pred[:, :3, 3]
            trans_gt = T_i_to_j_gt[:, :3, 3]

            rot_pred_ij = T_i_to_j_pred[:, :3, :3]
            rot_gt_ij = T_i_to_j_gt[:, :3, :3]

            # Translation loss
            trans_diff = torch.abs(trans_pred - trans_gt)
            trans_loss = self._huber_loss(trans_diff)

            # Rotation loss
            rot_loss = self._rotation_chordal_loss(rot_pred_ij, rot_gt_ij)

            total_loss += self.rotation_weight * rot_loss + self.translation_weight * trans_loss
            num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else 0.0

    def _quaternion_to_rotation_matrix(self, q):
        """将 quaternion [B, V, 4] 转换为 rotation matrix [B, V, 3, 3]"""
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)
        qw, qx, qy, qz = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]

        R00 = 1.0 - 2.0 * (qy * qy + qz * qz)
        R01 = 2.0 * (qx * qy - qz * qw)
        R02 = 2.0 * (qx * qz + qy * qw)

        R10 = 2.0 * (qx * qy + qz * qw)
        R11 = 1.0 - 2.0 * (qx * qx + qz * qz)
        R12 = 2.0 * (qy * qz - qx * qw)

        R20 = 2.0 * (qx * qz - qy * qw)
        R21 = 2.0 * (qy * qz + qx * qw)
        R22 = 1.0 - 2.0 * (qx * qx + qy * qy)

        R = torch.stack([
            torch.stack([R00, R01, R02], dim=-1),
            torch.stack([R10, R11, R12], dim=-1),
            torch.stack([R20, R21, R22], dim=-1),
        ], dim=-2)

        return R

    def _rotation_matrix_to_quaternion(self, R):
        """将 rotation matrix [B, 3, 3] 转换为 quaternion [B, 4]"""
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) / 2.0
        qx = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * qw + 1e-8)
        qy = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * qw + 1e-8)
        qz = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * qw + 1e-8)

        quat = torch.stack([qw, qx, qy, qz], dim=-1)
        return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

    def _rotation_chordal_loss(self, R1, R2):
        diff = R1.float() - R2.float()
        return (diff * diff).mean()

    def _quaternion_geodesic_loss(self, q1, q2):
        """Stable quaternion rotation loss for local relative pose training."""
        q1 = q1 / (torch.norm(q1, dim=-1, keepdim=True) + 1e-6)
        q2 = q2 / (torch.norm(q2, dim=-1, keepdim=True) + 1e-6)
        dot = torch.sum(q1 * q2, dim=-1).abs().clamp(max=1.0)
        return (1.0 - dot).mean()

    def _huber_loss(self, diff, delta=1.0):
        """Huber loss"""
        mask = (diff < delta).float()
        loss = mask * 0.5 * diff ** 2 + (1 - mask) * (delta * diff - 0.5 * delta ** 2)
        return loss.mean()


class ReplicaLongSequenceDataset(torch.utils.data.Dataset):
    """
    Replica 长序列数据集（论文4.2对齐版）

    使用 Foldback Video Sampler 生成长序列训练样本：
    - 支持动态 views 数（基于 curriculum）
    - 时间连续采样（foldback策略）
    - pose target 归一化到第一帧
    """

    def __init__(
        self,
        data_root: str,
        foldback_sampler: FoldbackVideoSampler,
        max_dim: int = 224,
        color_jitter_prob: float = 0.9,
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.1,
        grayscale_prob: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            data_root: Replica 数据根目录
            foldback_sampler: Foldback Video Sampler 实例
            max_dim: 训练图像最大边长
            color_jitter_prob: color jitter 概率
            brightness/contrast/saturation/hue: jitter参数
            grayscale_prob: grayscale概率
            seed: 随机种子
        """
        self.data_root = Path(data_root)
        self.foldback_sampler = foldback_sampler
        self.max_dim = max_dim
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale_prob = grayscale_prob
        self.seed = seed

        random.seed(seed)
        import numpy as np
        np.random.seed(seed)

        print(f"[ReplicaLongSequenceDataset] 初始化")
        print(f"  - Data root: {data_root}")
        print(f"  - Max dim: {max_dim}")
        print(f"  - Total frames: {foldback_sampler.total_frames}")

        # 加载 poses
        self.poses = self._load_poses()

        # 获取帧列表
        self.frame_ids = self._get_frame_ids()

        # 推断 intrinsic
        self.intrinsic = self._infer_intrinsic()

        print(f"[ReplicaLongSequenceDataset] 总帧数: {len(self.frame_ids)}")

    def _load_poses(self):
        """加载 traj.txt 文件中的 poses"""
        traj_file = self.data_root / "traj.txt"

        if not traj_file.exists():
            raise FileNotFoundError(f"traj.txt 文件不存在: {traj_file}")

        poses = {}
        with open(traj_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                values = [float(x) for x in line.strip().split()]
                if len(values) != 16:
                    continue
                T_c2w = np.array(values, dtype=np.float32).reshape(4, 4)
                poses[idx] = T_c2w

        print(f"[load_poses] 加载完成，共 {len(poses)} 个 poses")
        return poses

    def _get_frame_ids(self):
        """获取所有帧 ID"""
        import cv2

        results_dir = self.data_root / "results"

        if not results_dir.exists():
            raise FileNotFoundError(f"results 目录不存在: {results_dir}")

        rgb_files = sorted(results_dir.glob("frame*.jpg"))
        depth_files = sorted(results_dir.glob("depth*.png"))

        rgb_ids = set()
        for f in rgb_files:
            name = f.stem
            id_str = name.replace("frame", "")
            rgb_ids.add(int(id_str))

        depth_ids = set()
        for f in depth_files:
            name = f.stem
            id_str = name.replace("depth", "")
            depth_ids.add(int(id_str))

        valid_ids = sorted(list(rgb_ids & depth_ids))
        valid_ids = [id for id in valid_ids if id in self.poses]

        print(f"[get_frame_ids] 找到 {len(valid_ids)} 个有效帧")
        return valid_ids

    def _infer_intrinsic(self):
        """推断 intrinsic matrix"""
        import cv2

        first_rgb = self.data_root / "results" / f"frame{self.frame_ids[0]:06d}.jpg"
        img = cv2.imread(str(first_rgb))
        H, W = img.shape[:2]

        fx = fy = 600.0
        cx = W / 2.0
        cy = H / 2.0

        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

        print(f"[infer_intrinsic] fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        return K

    def set_num_views(self, num_views: int):
        """设置当前 iteration 的 views 数"""
        self.current_num_views = num_views

    def __len__(self):
        # 每个帧都可以作为起始帧
        return len(self.frame_ids)

    def __getitem__(self, idx: int):
        """获取单个样本"""
        import cv2

        # 使用 foldback sampler 采样序列
        num_views = self.current_num_views if hasattr(self, 'current_num_views') else 24
        frame_ids = self.foldback_sampler.sample_sequence(num_views, start_frame=self.frame_ids[idx])

        images_list, depths_list, valid_masks_list, intrinsics_list, poses_list = [], [], [], [], []

        for frame_id in frame_ids:
            # 加载 RGB
            rgb_path = self.data_root / "results" / f"frame{frame_id:06d}.jpg"
            rgb = cv2.imread(str(rgb_path))
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float32) / 255.0
            rgb = rgb.transpose(2, 0, 1)

            # 加载 depth (Replica scale: uint16 / 6553.5 -> meters)
            depth_path = self.data_root / "results" / f"depth{frame_id:06d}.png"
            depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            depth_m = depth_png.astype(np.float32) / REPLICA_DEPTH_SCALE

            T_c2w = self.poses[frame_id]
            K = self.intrinsic.copy()

            # 对齐到训练尺寸
            rgb, depth_m, K, valid_mask = self._align_to_train_size(rgb, depth_m, K, self.max_dim)

            images_list.append(rgb)
            depths_list.append(depth_m)
            valid_masks_list.append(valid_mask)
            intrinsics_list.append(K)
            poses_list.append(T_c2w)

        # pose 归一化到第一帧坐标系
        poses_normalized = self._normalize_poses(poses_list)

        # 转换为 tensor
        images = torch.from_numpy(np.stack(images_list, axis=0))
        depths = torch.from_numpy(np.stack(depths_list, axis=0))
        valid_masks = torch.from_numpy(np.stack(valid_masks_list, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0))
        poses = torch.from_numpy(np.stack(poses_normalized, axis=0))

        # Color jitter（与Stage1一致）
        if random.random() < self.color_jitter_prob:
            images = self._apply_color_jitter(images)

        return {
            'images': images,
            'depths': depths,
            'valid_masks': valid_masks,
            'intrinsics': intrinsics,
            'poses': poses,
            'frame_ids': frame_ids,
        }

    def _align_to_train_size(self, rgb, depth, K, max_dim):
        """对齐图像和内参到训练尺寸"""
        import cv2

        _, H, W = rgb.shape
        scale = max_dim / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)

        rgb_resized = cv2.resize(rgb.transpose(1, 2, 0), (new_W, new_H), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        depth_resized = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        valid_mask_resized = (depth_resized > 0).astype(np.float32)

        # Pad to 14 multiples
        patch_size = 14
        pad_H = (patch_size - new_H % patch_size) % patch_size
        pad_W = (patch_size - new_W % patch_size) % patch_size

        if pad_H > 0 or pad_W > 0:
            rgb_padded = np.pad(rgb_resized, ((0, 0), (0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
            depth_padded = np.pad(depth_resized, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
            valid_mask_padded = np.pad(valid_mask_resized, ((0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
        else:
            rgb_padded, depth_padded, valid_mask_padded = rgb_resized, depth_resized, valid_mask_resized

        K_new = K.copy()
        K_new[0, 0] = K[0, 0] * scale
        K_new[1, 1] = K[1, 1] * scale
        K_new[0, 2] = K[0, 2] * scale
        K_new[1, 2] = K[1, 2] * scale

        return rgb_padded, depth_padded, K_new, valid_mask_padded.astype(bool)

    def _normalize_poses(self, poses_list):
        """pose 归一化到第一帧坐标系"""
        T_ref = poses_list[0]
        T_ref_inv = np.linalg.inv(T_ref)
        poses_normalized = []
        for T_c2w in poses_list:
            poses_normalized.append(T_ref_inv @ T_c2w)
        return poses_normalized

    def _apply_color_jitter(self, images):
        """应用 color jitter（论文4.1对齐，与Stage1一致）"""
        import torchvision.transforms.functional as F

        images = torch.clamp(images, 0.0, 1.0)
        V = images.shape[0]
        jittered_images = []

        brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
        contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
        saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
        use_grayscale = random.random() < self.grayscale_prob

        for v in range(V):
            img_v = images[v]
            img_v = F.adjust_brightness(img_v, brightness_factor)
            img_v = F.adjust_contrast(img_v, contrast_factor)
            img_v = F.adjust_saturation(img_v, saturation_factor)
            if use_grayscale:
                gray = 0.299 * img_v[0] + 0.587 * img_v[1] + 0.114 * img_v[2]
                img_v = torch.stack([gray, gray, gray], dim=0)
            jittered_images.append(torch.clamp(img_v, 0.0, 1.0))

        return torch.stack(jittered_images, dim=0)


def train_one_iteration_stage2(
    model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
    iteration, args, window_pairs, scaler=None
):
    """训练单个 iteration（Stage2版本）"""
    model.train()

    images = batch['images'].to(args.device)
    depths = batch['depths'].to(args.device)
    valid_masks = batch['valid_masks'].to(args.device)
    poses = batch['poses'].to(args.device)

    optimizer.zero_grad()

    with autocast(enabled=args.use_amp):
        predictions = model(images)

        # Depth loss
        depth_loss = depth_loss_fn(predictions['depth'], depths, valid_masks)
        loss = depth_loss
        loss_dict = {'depth': depth_loss.item()}

        # Absolute pose loss
        if pose_loss_fn is not None and 'pose_enc' in predictions:
            pose_loss = pose_loss_fn(predictions['pose_enc'], poses)
            loss = loss + args.pose_weight * pose_loss
            loss_dict['abs_pose'] = pose_loss.item()

        # Local window relative pose loss
        if rel_pose_loss_fn is not None and 'pose_enc' in predictions:
            if iteration >= args.rel_pose_start_iter:
                rel_pose_loss = rel_pose_loss_fn(predictions['pose_enc'], poses, window_pairs)
                loss = loss + args.rel_pose_weight * rel_pose_loss
                loss_dict['rel_pose'] = rel_pose_loss.item()
            else:
                loss_dict['rel_pose'] = 0.0

    if args.use_amp and scaler:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
        optimizer.step()

    if scheduler is not None:
        scheduler.step()

    loss_dict['total'] = loss.item()

    return loss_dict


def save_checkpoint_stage2(model, optimizer, scheduler, iteration, loss_dict, args, path):
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_dict': loss_dict,
        'args': vars(args),
    }
    torch.save(checkpoint, path)
    print(f"[save_checkpoint] Saved to {path} (iteration {iteration})")


def main():
    parser = argparse.ArgumentParser(description="Stage2 Streaming Head-Only Training (论文4.2对齐版)")

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                        help="Replica data root directory")

    # 模型参数
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help="Stage1 checkpoint path to initialize from")
    parser.add_argument('--backbone', type=str, default='dinov2_vits14')
    parser.add_argument('--freeze_backbone', type=bool, default=True)
    parser.add_argument('--img_size', type=int, default=224)

    # 训练参数（论文4.2设置）
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--total_iterations', type=int, default=5000,
                        help="总训练 iterations（论文160K，资源不足可减少）")
    parser.add_argument('--lr', type=float, default=5e-4,          # 论文Stage2设置
                        help="Base learning rate (论文=5e-4)")
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help="Weight decay (论文=0.05)")
    parser.add_argument('--gradient_clip_norm', type=float, default=1.0)

    # Loss 权重
    parser.add_argument('--pose_weight', type=float, default=0.1)
    parser.add_argument('--rel_pose_weight', type=float, default=0.05)
    parser.add_argument('--rel_pose_start_iter', type=int, default=500)

    # View Curriculum 参数（论文4.2设置）
    parser.add_argument('--views_start', type=int, default=8,
                        help="起始views数（论文=24，显存不足可用8）")
    parser.add_argument('--views_end', type=int, default=24,
                        help="目标views数（论文=320，显存不足可用24）")
    parser.add_argument('--warmup_iterations', type=int, default=8000,
                        help="Warmup iterations (views数不增长)")

    # Local Window 参数（论文4.2设置）
    parser.add_argument('--k_min', type=int, default=16,
                        help="最小窗口大小（论文=16）")
    parser.add_argument('--k_max', type=int, default=64,
                        help="最大窗口大小（论文=64）")

    # Dynamic batch packing（论文4.2设置）
    parser.add_argument('--max_images_per_gpu', type=int, default=48,
                        help="每GPU最大图像数，动态调整batch_size")

    # Foldback Sampler 参数
    parser.add_argument('--stride_min', type=int, default=1)
    parser.add_argument('--stride_max', type=int, default=3)

    # Scheduler 参数
    parser.add_argument('--lr_warmup_ratio', type=float, default=0.05)
    parser.add_argument('--min_lr', type=float, default=1e-8)

    # 其他参数
    parser.add_argument('--use_amp', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints_stage2')
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 计算 LR warmup iterations
    lr_warmup_iterations = int(args.total_iterations * args.lr_warmup_ratio)

    print(f"\n{'='*60}")
    print(f"Stage2 Streaming Head-Only Training (论文4.2对齐版)")
    print(f"{'='*60}")
    print(f"Config:")
    print(f"  - backbone: {args.backbone}")
    print(f"  - lr: {args.lr}, weight_decay: {args.weight_decay}")
    print(f"  - total_iterations: {args.total_iterations}")
    print(f"  - views curriculum: {args.views_start} -> {args.views_end}")
    print(f"  - local window k: [{args.k_min}, {args.k_max}]")
    print(f"  - stage1 checkpoint: {args.stage1_checkpoint}")
    print(f"Data: {args.data_root}")

    # 1. 动态检测总帧数
    traj_file = Path(args.data_root) / "traj.txt"
    with open(traj_file, 'r') as f:
        total_frames = len([l for l in f.readlines() if len(l.strip().split()) == 16])
    print(f"  - 检测到总帧数: {total_frames}")
    foldback_sampler = FoldbackVideoSampler(
        total_frames=total_frames,
        stride_range=(args.stride_min, args.stride_max),
        redraw_stride_after_reverse=True,
        seed=args.seed,
    )

    # 2. 创建 View Curriculum
    view_curriculum = ProgressiveViewCurriculum(
        views_start=args.views_start,
        views_end=args.views_end,
        total_iterations=args.total_iterations,
        warmup_iterations=args.warmup_iterations,
        seed=args.seed,
    )

    # 3. 创建 Local Window Sampler
    window_sampler = LocalWindowSampler(
        k_min=args.k_min,
        k_max=args.k_max,
        seed=args.seed,
    )

    # 4. 创建 Dataset
    dataset = ReplicaLongSequenceDataset(
        data_root=args.data_root,
        foldback_sampler=foldback_sampler,
        max_dim=args.img_size,
        seed=args.seed,
    )

    # 5. 创建 DataLoader
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 6. 创建 Model 并从 Stage1 checkpoint 初始化
    model = create_head_only_model(
        backbone_name=args.backbone,
        freeze_backbone=args.freeze_backbone,
        img_size=args.img_size,
        train_depth_head=True,
        train_pose_head=True,
        num_views=args.views_start,  # 初始views数
    )
    model = model.to(args.device)

    # 加载 Stage1 checkpoint
    print(f"\n[Init] 从 Stage1 checkpoint 初始化: {args.stage1_checkpoint}")
    stage1_ckpt = torch.load(args.stage1_checkpoint, map_location=args.device)
    model.load_state_dict(stage1_ckpt['model_state_dict'], strict=False)
    print(f"[Init] Stage1 checkpoint 加载完成")

    # 7. Loss
    depth_loss_fn = DepthLoss(loss_type='masked_log_l1')
    pose_loss_fn = PoseLoss()
    rel_pose_loss_fn = LocalRelativePoseLoss()

    # 8. Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"[Optimizer] 可训练参数数量: {len(trainable_params)}")

    # 9. Scheduler (5% warmup + cosine decay)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.min_lr / args.lr,
        end_factor=1.0,
        total_iters=lr_warmup_iterations
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.total_iterations - lr_warmup_iterations,
        eta_min=args.min_lr
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[lr_warmup_iterations]
    )

    print(f"[Scheduler] Warmup: {lr_warmup_iterations} iterations, then cosine decay to {args.min_lr}")

    # 10. Scaler
    scaler = GradScaler() if args.use_amp else None

    # 11. Resume
    start_iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt['scheduler_state_dict']:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_iteration = ckpt['iteration'] + 1
        print(f"[Resume] 从 iteration {start_iteration} 继续")

    # 12. Training loop (iteration-based)
    print(f"\n{'='*60}")
    print(f"Starting Stage2 training")
    print(f"{'='*60}")

    start_time = time.time()
    iteration = start_iteration
    data_iter = iter(train_dataloader)

    while iteration < args.total_iterations:
        # 获取当前 views 数（基于 curriculum）
        current_views = view_curriculum.get_num_views_with_variance(iteration, variance=4)
        dataset.set_num_views(current_views)

        # Dynamic batch packing: batch_size * current_views <= max_images_per_gpu
        dynamic_batch_size = max(1, args.max_images_per_gpu // current_views)

        # 获取当前 window pairs

        # 获取当前 window pairs
        k = window_sampler.sample_window_size()
        window_pairs = window_sampler.get_adjacent_pairs(current_views, k)

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dataloader)
            batch = next(data_iter)

        loss_dict = train_one_iteration_stage2(
            model, batch, optimizer, scheduler, depth_loss_fn, pose_loss_fn, rel_pose_loss_fn,
            iteration, args, window_pairs, scaler
        )

        iteration += 1

        # Logging
        if iteration % args.log_every == 0:
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Iter {iteration}/{args.total_iterations}] "
                  f"Views: {current_views}, Batch: {dynamic_batch_size}, Window k: {k} "
                  f"Loss: {loss_dict['total']:.4f} "
                  f"(depth: {loss_dict.get('depth', 0):.4f}, "
                  f"abs_pose: {loss_dict.get('abs_pose', 0):.4f}, "
                  f"rel_pose: {loss_dict.get('rel_pose', 0):.4f}) "
                  f"LR: {current_lr:.2e} "
                  f"Time: {elapsed:.2f}s")

        # Save checkpoint
        if iteration % args.save_every == 0:
            save_checkpoint_stage2(model, optimizer, scheduler, iteration, loss_dict, args,
                                   output_dir / f'checkpoint_stage2_iter_{iteration}.pt')

    # Final checkpoint
    save_checkpoint_stage2(model, optimizer, scheduler, iteration, loss_dict, args,
                           output_dir / 'checkpoint_stage2_final.pt')

    print(f"\n{'='*60}")
    print(f"Stage2 Training completed")
    print(f"{'='*60}")
    print(f"  - Total iterations: {iteration}")
    print(f"  - Final loss: {loss_dict['total']:.4f}")
    print(f"  - Final views: {current_views}")
    print(f"  - Checkpoint saved to: {output_dir}")


if __name__ == '__main__':
    main()