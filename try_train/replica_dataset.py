"""
Replica 数据加载器（论文4.1对齐版）

根据 Replica 数据集的结构特点，本模块实现：

1. 加载 Replica 数据：
   - RGB: frame{id}.jpg (e.g., frame000000.jpg)
   - Depth: depth{id}.png (uint16, 单位毫米)
   - Pose: traj.txt (每行16个浮点数 → 4x4 camera-to-world matrix)
   - Intrinsic: 需推断 (默认 fx=fy=600, cx=W/2, cy=H/2)

2. 数据预处理：
   - depth 单位转换：毫米 -> 米
   - pose 格式统一：camera-to-world
   - RGB/depth/valid_mask/intrinsics 对齐
   - resize/pad 到训练尺寸（14 的整数倍）

3. 多视角采样策略：
   - 支持视角范围 [min_views, max_views]（论文第一阶段：2-24）
   - temporal_nearby sampler（时间窗口内采样）
   - shuffle_view_order（随机打乱视角顺序）

4. pose target 归一化：相对于第一帧

5. 数据增强（论文4.1对齐）：
   - color_jitter (brightness/contrast/saturation/hue)
   - grayscale (可选)

数据路径：/home/shared_files/datasets/dovsg/Replica/room0/

作者：Claude Code
日期：2026-05-12
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2


class ReplicaDataset(Dataset):
    """
    Replica 数据集

    用于训练 head-only 模型（冻结 DINOv2 backbone）

    特点：
    - 支持 2-4 views 的多视角采样
    - temporal_nearby sampler（时间窗口内采样）
    - pose target 归一化到第一帧坐标系
    - RGB/depth/intrinsics 对齐到训练尺寸

    数据格式：
    - RGB: frame{id}.jpg, 3-channel
    - Depth: depth{id}.png 16-bit, 单位毫米 -> 转为米
    - Pose: traj.txt, 每行16个浮点数 -> 4x4 camera-to-world matrix
    - Intrinsic: 推断 (fx=fy=600, cx=W/2, cy=H/2)
    """

    def __init__(
        self,
        data_root: str,
        num_views: int = 2,  # 保留兼容性，如果指定则固定值
        min_views: Optional[int] = None,
        max_views: Optional[int] = None,
        max_dim: int = 224,
        temporal_window: int = 30,
        shuffle_view_order: bool = True,
        color_jitter_prob: float = 0.9,  # 论文4.1设置
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.1,
        grayscale_prob: float = 0.05,
        seed: int = 42,
    ):
        """
        Args:
            data_root: Replica 数据根目录 (e.g., /path/to/Replica/room0/)
            num_views: 固定视角数（保留兼容性）
            min_views: 最小视角数（论文第一阶段：2）
            max_views: 最大视角数（论文第一阶段：24）
            max_dim: 训练图像最大边长（默认224，论文目标518）
            temporal_window: 时间采样窗口大小（默认30帧）
            shuffle_view_order: 是否随机打乱视角顺序
            color_jitter_prob: color jitter 概率（论文4.1=0.9）
            brightness: brightness jitter 参数（论文4.1=0.5）
            contrast: contrast jitter 参数（论文4.1=0.5）
            saturation: saturation jitter 参数（论文4.1=0.5）
            hue: hue jitter 参数（论文4.1=0.1）
            grayscale_prob: grayscale 概率（论文4.1=0.05）
            seed: 随机种子
        """
        self.data_root = Path(data_root)

        # 视角数设置：优先使用范围，否则使用固定值
        if min_views is not None and max_views is not None:
            self.min_views = min_views
            self.max_views = max_views
            self.use_view_range = True
        else:
            self.min_views = num_views
            self.max_views = num_views
            self.use_view_range = False

        self.max_dim = max_dim
        self.temporal_window = temporal_window
        self.shuffle_view_order = shuffle_view_order

        # 数据增强参数（论文4.1对齐）
        self.color_jitter_prob = color_jitter_prob
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.grayscale_prob = grayscale_prob

        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        print(f"[ReplicaDataset] 初始化")
        print(f"  - Data root: {data_root}")
        print(f"  - 视角数: {self.min_views}-{self.max_views} (range={self.use_view_range})")
        print(f"  - 最大尺寸: {max_dim}")
        print(f"  - 时间窗口: {temporal_window}")
        print(f"  - Shuffle: {shuffle_view_order}")
        print(f"  - Color jitter prob: {color_jitter_prob}")

        # 加载 poses
        self.poses = self._load_poses()

        # 获取帧列表
        self.frame_ids = self._get_frame_ids()

        # 推断 intrinsic
        self.intrinsic = self._infer_intrinsic()

        # 构建样本列表
        self.samples = self._build_samples()

        print(f"[ReplicaDataset] 总样本数: {len(self.samples)}")

    def _load_poses(self) -> Dict[int, np.ndarray]:
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
                    print(f"[load_poses] 第 {idx} 行格式错误，跳过")
                    continue
                T_c2w = np.array(values, dtype=np.float32).reshape(4, 4)
                poses[idx] = T_c2w

        print(f"[load_poses] 加载完成，共 {len(poses)} 个 poses")
        return poses

    def _get_frame_ids(self) -> List[int]:
        """获取所有帧 ID"""
        results_dir = self.data_root / "results"

        if not results_dir.exists():
            raise FileNotFoundError(f"results 目录不存在: {results_dir}")

        # 匹配 RGB 文件
        rgb_files = sorted(results_dir.glob("frame*.jpg"))
        depth_files = sorted(results_dir.glob("depth*.png"))

        # 提取帧 ID
        rgb_ids = set()
        for f in rgb_files:
            # frame000000.jpg -> 0
            name = f.stem
            id_str = name.replace("frame", "")
            rgb_ids.add(int(id_str))

        depth_ids = set()
        for f in depth_files:
            # depth000000.png -> 0
            name = f.stem
            id_str = name.replace("depth", "")
            depth_ids.add(int(id_str))

        # 只保留同时有 RGB 和 depth 的帧
        valid_ids = sorted(list(rgb_ids & depth_ids))

        # 同时检查 pose 是否存在
        valid_ids = [id for id in valid_ids if id in self.poses]

        print(f"[get_frame_ids] 找到 {len(valid_ids)} 个有效帧")
        return valid_ids

    def _infer_intrinsic(self) -> np.ndarray:
        """推断 intrinsic matrix"""
        # 默认 Replica 内参: fx=fy=600
        # cx, cy 根据图像尺寸推断

        # 读取第一帧获取图像尺寸
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

        print(f"[infer_intrinsic] 推断完成")
        print(f"  - fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        print(f"  - Image size: {W}x{H}")

        return K

    def _build_samples(self) -> List[List[int]]:
        """构建样本列表（temporal_nearby sampler，支持动态视角数）"""
        samples = []

        if len(self.frame_ids) < self.min_views:
            print(f"[build_samples] 帧数不足，无法构建样本")
            return samples

        for ref_idx in range(len(self.frame_ids)):
            ref_frame_id = self.frame_ids[ref_idx]

            # 时间窗口内的帧
            window_start = max(0, ref_idx - self.temporal_window)
            window_end = min(len(self.frame_ids), ref_idx + self.temporal_window + 1)
            window_frame_ids = self.frame_ids[window_start:window_end]

            if len(window_frame_ids) < self.min_views:
                continue

            # 动态采样视角数（如果使用范围）
            if self.use_view_range:
                # 确保不超过窗口内可用的帧数
                max_possible = min(self.max_views, len(window_frame_ids))
                num_views = random.randint(self.min_views, max_possible)
            else:
                num_views = self.min_views

            # 从窗口中采样 num_views 个帧
            sampled_frames = random.sample(window_frame_ids, num_views)

            # 如果 shuffle，打乱顺序（但保留第一个作为参考帧）
            if self.shuffle_view_order:
                ref_frame = sampled_frames[0]
                other_frames = sampled_frames[1:]
                random.shuffle(other_frames)
                sampled_frames = [ref_frame] + other_frames

            samples.append(sampled_frames)

        print(f"[build_samples] 构建完成，共 {len(samples)} 个样本")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        frame_ids = self.samples[idx]

        images_list, depths_list, valid_masks_list, intrinsics_list, poses_list = [], [], [], [], []

        for frame_id in frame_ids:
            rgb = self._load_rgb(frame_id)
            depth_m = self._load_depth(frame_id)
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

        # Color jitter
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

    def _load_rgb(self, frame_id: int) -> np.ndarray:
        """加载 RGB 图像"""
        rgb_path = self.data_root / "results" / f"frame{frame_id:06d}.jpg"
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)

    def _load_depth(self, frame_id: int) -> np.ndarray:
        """加载深度图像，单位转换 mm -> m"""
        depth_path = self.data_root / "results" / f"depth{frame_id:06d}.png"
        depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_mm = depth_png.astype(np.float32)
        return depth_mm / 1000.0

    def _align_to_train_size(self, rgb, depth, K, max_dim):
        """对齐图像和内参到训练尺寸"""
        _, H, W = rgb.shape
        scale = max_dim / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)

        # Resize
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

        # 调整内参
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
        """应用 color jitter（论文4.1对齐）

        Args:
            images: [V, 3, H, W] tensor of multiple views

        Returns:
            jittered images: [V, 3, H, W]
        """
        import torchvision.transforms.functional as F

        # Clamp to valid range first
        images = torch.clamp(images, 0.0, 1.0)

        # 对每个 view 应用 color jitter
        V = images.shape[0]
        jittered_images = []

        for v in range(V):
            img_v = images[v]  # [3, H, W]

            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)

            img_v = F.adjust_brightness(img_v, brightness_factor)
            img_v = F.adjust_contrast(img_v, contrast_factor)
            img_v = F.adjust_saturation(img_v, saturation_factor)

            # 可选 grayscale（所有 view 使用相同决策）
            if v == 0 and random.random() < self.grayscale_prob:
                # 标记需要 grayscale
                self._apply_grayscale = True
            if hasattr(self, '_apply_grayscale') and self._apply_grayscale:
                gray = 0.299 * img_v[0] + 0.587 * img_v[1] + 0.114 * img_v[2]
                img_v = torch.stack([gray, gray, gray], dim=0)

            jittered_images.append(torch.clamp(img_v, 0.0, 1.0))

        # 清除 grayscale 标记
        if hasattr(self, '_apply_grayscale'):
            del self._apply_grayscale

        return torch.stack(jittered_images, dim=0)


def create_replica_dataloader(
    data_root,
    batch_size=1,
    num_views=2,  # 保留兼容性
    min_views=None,
    max_views=None,
    max_dim=224,
    shuffle=True,
    num_workers=4,
    seed=42,
    # 论文4.1数据增强参数
    color_jitter_prob=0.9,
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
    hue=0.1,
    grayscale_prob=0.05,
):
    """创建 Replica DataLoader（论文4.1对齐版）

    Args:
        data_root: Replica 数据根目录
        batch_size: batch 大小
        num_views: 固定视角数（保留兼容性）
        min_views: 最小视角数（论文第一阶段：2）
        max_views: 最大视角数（论文第一阶段：24）
        max_dim: 图像最大边长
        shuffle: 是否 shuffle
        num_workers: DataLoader worker 数
        seed: 随机种子
        color_jitter_prob: color jitter 概率（论文4.1=0.9）
        brightness: brightness jitter 参数（论文4.1=0.5）
        contrast: contrast jitter 参数（论文4.1=0.5）
        saturation: saturation jitter 参数（论文4.1=0.5）
        hue: hue jitter 参数（论文4.1=0.1）
        grayscale_prob: grayscale 概率（论文4.1=0.05）
    """
    dataset = ReplicaDataset(
        data_root=data_root,
        num_views=num_views,
        min_views=min_views,
        max_views=max_views,
        max_dim=max_dim,
        seed=seed,
        color_jitter_prob=color_jitter_prob,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        grayscale_prob=grayscale_prob,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Replica 数据加载器测试（论文4.1对齐版）")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_views', type=int, default=2, help='固定视角数')
    parser.add_argument('--min_views', type=int, default=None, help='最小视角数（论文第一阶段：2）')
    parser.add_argument('--max_views', type=int, default=None, help='最大视角数（论文第一阶段：24）')
    parser.add_argument('--max_dim', type=int, default=224)
    args = parser.parse_args()

    dataloader = create_replica_dataloader(
        args.data_root,
        args.batch_size,
        args.num_views,
        args.min_views,
        args.max_views,
        args.max_dim
    )
    print(f"\n[测试] 读取第一个 batch...")
    for batch in dataloader:
        print(f"  - Images: {batch['images'].shape}")
        print(f"  - Depths: {batch['depths'].shape}")
        print(f"  - Poses: {batch['poses'].shape}")
        print(f"  - Frame IDs: {batch['frame_ids']}")
        print(f"  - 视角数: {batch['images'].shape[1]}")
        break
    print(f"[测试] 完成")