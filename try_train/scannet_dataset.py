"""
ScanNet 数据加载器

根据 try_train/stage1_scannet_training_plan.md 文档的要求，
本模块实现：

1. 加载导出的 ScanNet 数据
2. 数据预处理：
   - depth 单位转换：毫米 -> 米
   - pose 格式统一：camera-to-world
   - RGB/depth/valid_mask/intrinsics 对齐
   - resize/pad 到训练尺寸（14 的整数倍）
3. 多视角采样策略：temporal_nearby sampler
4. pose target 归一化：相对于第一帧

参考文档第4、5节的详细要求。

作者：Claude Code
日期：2026-05-11
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


class ScanNetDataset(Dataset):
    """
    ScanNet 数据集

    用于训练 head-only 模型（冻结 DINOv2 backbone）

    特点：
    - 支持 2-4 views 的多视角采样
    - temporal_nearby sampler（时间窗口内采样）
    - pose target 录一化到第一帧坐标系
    - RGB/depth/intrinsics 对齐到训练尺寸

    数据格式：
    - RGB: JPG, 3-channel
    - Depth: PNG 16-bit, 单位毫米 -> 转为米
    - Pose: 4x4 camera-to-world matrix
    - Intrinsic: 3x3 matrix
    """

    def __init__(
        self,
        metadata_path: str,
        num_views: int = 2,
        max_dim: int = 224,
        temporal_window: int = 30,
        shuffle_view_order: bool = True,
        color_jitter_prob: float = 0.3,
        seed: int = 42,
    ):
        """
        Args:
            metadata_path: metadata JSON 文件路径
            num_views: 每个样本的视角数（默认2）
            max_dim: 训练图像最大边长（默认224，显存允许可改336）
            temporal_window: 时间采样窗口大小（默认30帧）
            shuffle_view_order: 是否随机打乱视角顺序
            color_jitter_prob: color jitter 概率（默认0.3）
            seed: 随机种子
        """
        self.metadata_path = Path(metadata_path)
        self.num_views = num_views
        self.max_dim = max_dim
        self.temporal_window = temporal_window
        self.shuffle_view_order = shuffle_view_order
        self.color_jitter_prob = color_jitter_prob
        self.seed = seed

        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)

        print(f"[ScanNetDataset] 初始化")
        print(f"  - Metadata: {metadata_path}")
        print(f"  - 视角数: {num_views}")
        print(f"  - 最大尺寸: {max_dim}")
        print(f"  - 时间窗口: {temporal_window}")
        print(f"  - Shuffle: {shuffle_view_order}")

        # 加载 metadata
        self.metadata = self._load_metadata()

        # 构建样本列表
        self.samples = self._build_samples()

        print(f"[ScanNetDataset] 总样本数: {len(self.samples)}")

    def _load_metadata(self) -> Dict:
        """加载 metadata JSON 文件"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata 文件不存在: {self.metadata_path}")

        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"[load_metadata] 加载完成")
        print(f"  - Dataset: {metadata['dataset']}")
        print(f"  - Split: {metadata['split']}")
        print(f"  - 场景数: {len(metadata['scenes'])}")
        print(f"  - 总帧数: {sum(len(frames) for frames in metadata['frames'].values())}")

        return metadata

    def _build_samples(self) -> List[Tuple[str, List[int]]]:
        """构建样本列表（temporal_nearby sampler）"""
        samples = []
        scenes = self.metadata['scenes']
        frames_dict = self.metadata['frames']

        for scene_id in scenes:
            frames_info = frames_dict[scene_id]
            frame_ids = [info['frame_id'] for info in frames_info]

            if len(frame_ids) < self.num_views:
                print(f"[build_samples] {scene_id}: 帧数不足，跳过")
                continue

            for ref_idx in range(len(frame_ids)):
                window_start = max(0, ref_idx - self.temporal_window)
                window_end = min(len(frame_ids), ref_idx + self.temporal_window + 1)
                window_frames = frame_ids[window_start:window_end]

                if len(window_frames) < self.num_views:
                    continue

                sampled_frames = random.sample(window_frames, self.num_views)
                if self.shuffle_view_order:
                    ref_frame = sampled_frames[0]
                    other_frames = sampled_frames[1:]
                    random.shuffle(other_frames)
                    sampled_frames = [ref_frame] + other_frames

                samples.append((scene_id, sampled_frames))

        print(f"[build_samples] 构建完成，共 {len(samples)} 个样本")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        scene_id, frame_ids = self.samples[idx]

        images_list, depths_list, valid_masks_list, intrinsics_list, poses_list = [], [], [], [], []

        for frame_id in frame_ids:
            frame_info = self._get_frame_info(scene_id, frame_id)
            rgb = self._load_rgb(frame_info['rgb_path'])
            depth_m = self._load_depth(frame_info['depth_path'])
            T_c2w = np.array(frame_info['T_c2w'], dtype=np.float32)
            K = np.array(frame_info['K'], dtype=np.float32)

            rgb, depth_m, K, valid_mask = self._align_to_train_size(rgb, depth_m, K, self.max_dim)

            images_list.append(rgb)
            depths_list.append(depth_m)
            valid_masks_list.append(valid_mask)
            intrinsics_list.append(K)
            poses_list.append(T_c2w)

        poses_normalized = self._normalize_poses(poses_list)

        images = torch.from_numpy(np.stack(images_list, axis=0))
        depths = torch.from_numpy(np.stack(depths_list, axis=0))
        valid_masks = torch.from_numpy(np.stack(valid_masks_list, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0))
        poses = torch.from_numpy(np.stack(poses_normalized, axis=0))

        if random.random() < self.color_jitter_prob:
            images = self._apply_color_jitter(images)

        return {
            'images': images, 'depths': depths, 'valid_masks': valid_masks,
            'intrinsics': intrinsics, 'poses': poses,
            'scene_id': scene_id, 'frame_ids': frame_ids,
        }

    def _get_frame_info(self, scene_id: str, frame_id: int) -> Dict:
        frames_info = self.metadata['frames'][scene_id]
        for info in frames_info:
            if info['frame_id'] == frame_id:
                return info
        raise ValueError(f"找不到帧: {scene_id}/{frame_id}")

    def _load_rgb(self, rgb_path: str) -> np.ndarray:
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)

    def _load_depth(self, depth_path: str) -> np.ndarray:
        depth_png = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_mm = depth_png.astype(np.float32)
        return depth_mm / 1000.0

    def _align_to_train_size(self, rgb, depth, K, max_dim):
        _, H, W = rgb.shape
        scale = max_dim / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)

        rgb_resized = cv2.resize(rgb.transpose(1,2,0), (new_W, new_H), interpolation=cv2.INTER_LINEAR).transpose(2,0,1)
        depth_resized = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        valid_mask_resized = (depth_resized > 0).astype(np.float32)

        patch_size = 14
        pad_H = (patch_size - new_H % patch_size) % patch_size
        pad_W = (patch_size - new_W % patch_size) % patch_size

        if pad_H > 0 or pad_W > 0:
            rgb_padded = np.pad(rgb_resized, ((0,0),(0,pad_H),(0,pad_W)), mode='constant', constant_values=0)
            depth_padded = np.pad(depth_resized, ((0,pad_H),(0,pad_W)), mode='constant', constant_values=0)
            valid_mask_padded = np.pad(valid_mask_resized, ((0,pad_H),(0,pad_W)), mode='constant', constant_values=0)
        else:
            rgb_padded, depth_padded, valid_mask_padded = rgb_resized, depth_resized, valid_mask_resized

        K_new = K.copy()
        K_new[0,0] = K[0,0] * scale
        K_new[1,1] = K[1,1] * scale
        K_new[0,2] = K[0,2] * scale
        K_new[1,2] = K[1,2] * scale

        return rgb_padded, depth_padded, K_new, valid_mask_padded.astype(bool)

    def _normalize_poses(self, poses_list):
        T_ref = poses_list[0]
        T_ref_inv = np.linalg.inv(T_ref)
        poses_normalized = []
        for T_c2w in poses_list:
            poses_normalized.append(T_ref_inv @ T_c2w)
        return poses_normalized

    def _apply_color_jitter(self, images):
        import torchvision.transforms.functional as F
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        images = F.adjust_brightness(images, brightness_factor)
        images = F.adjust_contrast(images, contrast_factor)
        return F.adjust_saturation(images, saturation_factor)


def create_dataloader(metadata_path, batch_size=1, num_views=2, max_dim=224, shuffle=True, num_workers=4, seed=42):
    dataset = ScanNetDataset(metadata_path=metadata_path, num_views=num_views, max_dim=max_dim, seed=seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_views', type=int, default=2)
    parser.add_argument('--max_dim', type=int, default=224)
    args = parser.parse_args()

    dataloader = create_dataloader(args.metadata, args.batch_size, args.num_views, args.max_dim)
    print(f"\n[测试] 读取第一个 batch...")
    for batch in dataloader:
        print(f"  - Images: {batch['images'].shape}")
        print(f"  - Depths: {batch['depths'].shape}")
        print(f"  - Scene: {batch['scene_id']}")
        break
    print(f"[测试] 完成")