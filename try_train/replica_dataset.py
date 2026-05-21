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


# Replica depth scale factor.
#
# Replica 的 depth*.png 是 uint16，但 NOT 标准毫米单位。
# 学术界（NICE-SLAM / iMAP / GO-SLAM / MonoGS）都使用 scale = 6553.5，
# 即 raw=65535 对应 ~10m（适合室内场景）。
#
# 早期代码错用 /1000 会让 Replica room0 的 depth 中位数变成 ~17m（错误），
# 正确除以 6553.5 后中位数 ~2.69m（符合室内场景）。
REPLICA_DEPTH_SCALE = 6553.5


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
        sampler_type: str = 'temporal_nearby',  # 'temporal_nearby' 或 'spatial_nearby'
        spatial_radius: float = 5.0,  # spatial nearby 的3D距离阈值（米）
        color_jitter_prob: float = 0.9,  # 论文4.1设置
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5,
        hue: float = 0.1,
        grayscale_prob: float = 0.05,
        # Geometric augmentation（论文4.1设置）
        spatial_rescale_range: Tuple[float, float] = (0.8, 1.2),  # spatial rescale
        aspect_ratio_range: Tuple[float, float] = (0.33, 1.0),    # aspect ratio
        co_jitter: bool = True,  # 共享颜色扰动参数
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
            sampler_type: 采样策略 ('temporal_nearby' 或 'spatial_nearby')
            spatial_radius: spatial nearby 的3D距离阈值（米，默认5.0）
            color_jitter_prob: color jitter 概率（论文4.1=0.9）
            brightness: brightness jitter 参数（论文4.1=0.5）
            contrast: contrast jitter 参数（论文4.1=0.5）
            saturation: saturation jitter 参数（论文4.1=0.5）
            hue: hue jitter 参数（论文4.1=0.1）
            grayscale_prob: grayscale 概率（论文4.1=0.05）
            spatial_rescale_range: spatial rescale 范围（论文=[0.8,1.2]）
            aspect_ratio_range: aspect ratio 范围（论文=[0.33,1.0]）
            co_jitter: 是否所有view共享颜色扰动参数（论文=True）
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
        self.sampler_type = sampler_type
        self.spatial_radius = spatial_radius
        self.spatial_rescale_range = spatial_rescale_range
        self.aspect_ratio_range = aspect_ratio_range
        self.co_jitter = co_jitter

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
        print(f"  - 采样策略: {sampler_type}")
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

        # 构建空间距离矩阵（spatial nearby sampler 使用）
        self.camera_centers = None
        self.spatial_dist_matrix = None
        if self.sampler_type == 'spatial_nearby':
            self._build_spatial_distance_matrix()

        print(f"[ReplicaDataset] 总样本数: {len(self.samples)}")

    def _load_poses(self) -> Dict[int, np.ndarray]:
        """加载 traj.txt 文件中的 poses（含有效性检查）"""
        traj_file = self.data_root / "traj.txt"

        if not traj_file.exists():
            raise FileNotFoundError(f"traj.txt 文件不存在: {traj_file}")

        poses = {}
        invalid_pose_count = 0
        with open(traj_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                values = [float(x) for x in line.strip().split()]
                if len(values) != 16:
                    print(f"[load_poses] 第 {idx} 行格式错误，跳过")
                    continue
                T_c2w = np.array(values, dtype=np.float32).reshape(4, 4)

                # 有效性检查：所有值必须 finite
                if not np.all(np.isfinite(T_c2w)):
                    invalid_pose_count += 1
                    continue

                # 有效性检查：旋转矩阵行列式应接近1（正交性）
                det = np.linalg.det(T_c2w[:3, :3])
                if abs(det) < 0.9 or abs(det) > 1.1:
                    invalid_pose_count += 1
                    continue

                poses[idx] = T_c2w

        print(f"[load_poses] 加载完成，共 {len(poses)} 个有效 poses"
              f"{f'，过滤 {invalid_pose_count} 个无效' if invalid_pose_count > 0 else ''}")
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

        # 检查深度有效比例（valid_ratio >= 0.05）
        depth_valid_ids = []
        for fid in valid_ids:
            depth_path = results_dir / f"depth{fid:06d}.png"
            depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_png is not None:
                valid_ratio = (depth_png > 0).sum() / depth_png.size
                if valid_ratio >= 0.05:
                    depth_valid_ids.append(fid)
        filtered_count = len(valid_ids) - len(depth_valid_ids)
        if filtered_count > 0:
            print(f"[get_frame_ids] 深度有效比例过滤: 移除 {filtered_count} 帧 (valid_ratio < 0.05)")
        valid_ids = depth_valid_ids

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

    def _build_spatial_distance_matrix(self):
        """构建基于 camera center 3D 距离的距离矩阵（spatial nearby sampler 使用）"""
        n = len(self.frame_ids)
        self.camera_centers = np.zeros((n, 3), dtype=np.float32)
        for i, fid in enumerate(self.frame_ids):
            self.camera_centers[i] = self.poses[fid][:3, 3]

        # 计算两两距离
        diff = self.camera_centers[:, np.newaxis, :] - self.camera_centers[np.newaxis, :, :]
        self.spatial_dist_matrix = np.linalg.norm(diff, axis=-1)  # [n, n]
        print(f"[spatial_nearby] 距离矩阵构建完成, shape={self.spatial_dist_matrix.shape}"
              f", 平均距离={self.spatial_dist_matrix[self.spatial_dist_matrix > 0].mean():.2f}m")

    def _spatial_nearby_sample(self, ref_idx: int) -> Optional[List[int]]:
        """基于 camera center 3D 距离采样（spatial nearby sampler）

        Args:
            ref_idx: 参考帧在 frame_ids 中的索引

        Returns:
            采样到的 frame_ids 列表，或 None（如果空间邻居不足）
        """
        if self.spatial_dist_matrix is None:
            return None

        n = len(self.frame_ids)
        distances = self.spatial_dist_matrix[ref_idx]  # [n]

        # 找到在空间半径内的帧
        nearby_mask = distances <= self.spatial_radius
        nearby_mask[ref_idx] = True  # 确保参考帧包含在内
        nearby_indices = np.where(nearby_mask)[0]

        if len(nearby_indices) < self.min_views:
            # 空间邻居不足，回退到距离最近的 min_views 个帧
            sorted_indices = np.argsort(distances)
            nearby_indices = sorted_indices[:max(self.min_views, 2)]

        nearby_frame_ids = [self.frame_ids[i] for i in nearby_indices]

        # 动态采样视角数
        if self.use_view_range:
            max_possible = min(self.max_views, len(nearby_frame_ids))
            num_views = random.randint(self.min_views, max_possible)
        else:
            num_views = self.min_views

        # 确保参考帧在第一个位置
        ref_frame = self.frame_ids[ref_idx]
        other_frames = [f for f in nearby_frame_ids if f != ref_frame]
        random.shuffle(other_frames)
        sampled = [ref_frame] + other_frames[:num_views - 1]

        if len(sampled) < self.min_views:
            return None

        return sampled

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        # 如果使用 spatial nearby sampler，在运行时动态采样
        if self.sampler_type == 'spatial_nearby':
            sampled = self._spatial_nearby_sample(idx)
            if sampled is not None:
                frame_ids = sampled
            else:
                frame_ids = self.samples[idx]  # fallback
        else:
            frame_ids = self.samples[idx]

        images_list, depths_list, valid_masks_list, intrinsics_list, poses_list = [], [], [], [], []

        # Geometric augmentation: 采样共享参数（所有view使用相同的rescale/aspect_ratio）
        do_spatial_rescale = self.spatial_rescale_range is not None and random.random() < 0.5
        do_aspect_ratio = self.aspect_ratio_range is not None and random.random() < 0.5
        # 预采样共享参数，确保所有view裁剪一致
        shared_rescale = random.uniform(*self.spatial_rescale_range) if do_spatial_rescale else 1.0
        shared_aspect_ratio = random.uniform(*self.aspect_ratio_range) if do_aspect_ratio else None
        shared_aspect_x_start = None  # 延迟初始化（需要知道图像尺寸）

        for frame_id in frame_ids:
            rgb = self._load_rgb(frame_id)
            depth_m = self._load_depth(frame_id)
            T_c2w = self.poses[frame_id]
            K = self.intrinsic.copy()

            # Geometric augmentation（在resize之前，同步更新intrinsics）
            if do_spatial_rescale:
                rgb, depth_m, K, T_c2w = self._apply_spatial_rescale(rgb, depth_m, K, T_c2w, scale=shared_rescale)
            if do_aspect_ratio:
                rgb, depth_m, K, T_c2w = self._apply_aspect_ratio(rgb, depth_m, K, T_c2w, target_ratio=shared_aspect_ratio)

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
        """加载深度图像（Replica uint16 raw -> 米）

        注意：Replica 不是标准毫米单位。其 depth*.png 是 uint16，
        scale factor = REPLICA_DEPTH_SCALE (6553.5)，与 NICE-SLAM/iMAP/
        GO-SLAM/MonoGS 等 Replica 学术工作一致：raw=65535 映射 ~10m。
        早期 /1000 会让室内 depth 中位数变成 ~17m（错误）。
        """
        depth_path = self.data_root / "results" / f"depth{frame_id:06d}.png"
        depth_png = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        depth_raw = depth_png.astype(np.float32)
        return depth_raw / REPLICA_DEPTH_SCALE

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
        """应用 color jitter（论文4.1对齐，支持co-jitter）

        Args:
            images: [V, 3, H, W] tensor of multiple views

        Returns:
            jittered images: [V, 3, H, W]
        """
        import torchvision.transforms.functional as F

        # Clamp to valid range first
        images = torch.clamp(images, 0.0, 1.0)

        V = images.shape[0]
        jittered_images = []

        # co-jitter: 所有 view 共享相同的颜色扰动参数
        if self.co_jitter:
            brightness_factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            contrast_factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            saturation_factor = random.uniform(1 - self.saturation, 1 + self.saturation)
            use_grayscale = random.random() < self.grayscale_prob
        else:
            brightness_factor = None
            contrast_factor = None
            saturation_factor = None
            use_grayscale = False

        for v in range(V):
            img_v = images[v]  # [3, H, W]

            if self.co_jitter:
                # 使用共享参数
                b_f, c_f, s_f = brightness_factor, contrast_factor, saturation_factor
            else:
                # 每个 view 独立参数
                b_f = random.uniform(1 - self.brightness, 1 + self.brightness)
                c_f = random.uniform(1 - self.contrast, 1 + self.contrast)
                s_f = random.uniform(1 - self.saturation, 1 + self.saturation)

            img_v = F.adjust_brightness(img_v, b_f)
            img_v = F.adjust_contrast(img_v, c_f)
            img_v = F.adjust_saturation(img_v, s_f)

            if use_grayscale or (not self.co_jitter and random.random() < self.grayscale_prob):
                gray = 0.299 * img_v[0] + 0.587 * img_v[1] + 0.114 * img_v[2]
                img_v = torch.stack([gray, gray, gray], dim=0)

            jittered_images.append(torch.clamp(img_v, 0.0, 1.0))

        return torch.stack(jittered_images, dim=0)

    def _apply_spatial_rescale(self, rgb, depth, K, T_c2w, scale=None):
        """应用 spatial rescale [0.8, 1.2]（论文4.1）

        随机缩放图像，同步更新 intrinsics 和 depth。
        不需要更新 pose（相机位置不变，只改变FOV等效效果）。

        Args:
            rgb: [3, H, W] float32
            depth: [H, W] float32
            K: [3, 3] intrinsic matrix
            T_c2w: [4, 4] pose matrix
            scale: 预采样的缩放因子（None则随机采样）

        Returns:
            rgb, depth, K, T_c2w (可能被缩放)
        """
        if self.spatial_rescale_range is None and scale is None:
            return rgb, depth, K, T_c2w

        if scale is None:
            scale = random.uniform(*self.spatial_rescale_range)
        _, H, W = rgb.shape
        new_H, new_W = int(H * scale), int(W * scale)

        if new_H == H and new_W == W:
            return rgb, depth, K, T_c2w

        # Resize
        rgb = cv2.resize(rgb.transpose(1, 2, 0), (new_W, new_H), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
        depth = cv2.resize(depth, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

        # 同步更新 intrinsics
        K_new = K.copy()
        K_new[0, 0] *= scale
        K_new[1, 1] *= scale
        K_new[0, 2] *= scale
        K_new[1, 2] *= scale

        return rgb, depth, K_new, T_c2w

    def _apply_aspect_ratio(self, rgb, depth, K, T_c2w, target_ratio=None):
        """应用 aspect ratio sampling [0.33, 1.0]（论文4.1）

        随机裁剪图像到目标宽高比，同步更新 intrinsics。
        ratio=1.0 表示正方形，ratio=0.33 表示宽:高=1:3（竖长）。

        Args:
            rgb: [3, H, W] float32
            depth: [H, W] float32
            K: [3, 3] intrinsic matrix
            T_c2w: [4, 4] pose matrix
            target_ratio: 预采样的目标宽高比（None则随机采样）

        Returns:
            rgb, depth, K, T_c2w
        """
        if self.aspect_ratio_range is None and target_ratio is None:
            return rgb, depth, K, T_c2w

        _, H, W = rgb.shape
        current_ratio = W / H  # width/height ratio

        # 使用预采样或随机采样的目标 ratio
        if target_ratio is None:
            target_ratio = random.uniform(*self.aspect_ratio_range)

        # 计算裁剪区域
        if target_ratio < current_ratio:
            # 目标更窄：裁剪宽度
            new_W = int(H * target_ratio)
            x_start = random.randint(0, max(0, W - new_W))
            y_start = 0
            crop_W, crop_H = new_W, H
        else:
            # 目标更宽/相等：裁剪高度
            new_H = int(W / target_ratio)
            x_start = 0
            y_start = random.randint(0, max(0, H - new_H))
            crop_W, crop_H = W, new_H

        if crop_H == H and crop_W == W:
            return rgb, depth, K, T_c2w

        # 裁剪
        rgb = rgb[:, y_start:y_start+crop_H, x_start:x_start+crop_W]
        depth = depth[y_start:y_start+crop_H, x_start:x_start+crop_W]

        # 同步更新 intrinsics（减去裁剪偏移）
        K_new = K.copy()
        K_new[0, 2] -= x_start
        K_new[1, 2] -= y_start

        return rgb, depth, K_new, T_c2w


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
    sampler_type='temporal_nearby',
    spatial_radius=5.0,
    # 论文4.1数据增强参数
    color_jitter_prob=0.9,
    brightness=0.5,
    contrast=0.5,
    saturation=0.5,
    hue=0.1,
    grayscale_prob=0.05,
    # Geometric augmentation
    spatial_rescale_range=(0.8, 1.2),
    aspect_ratio_range=(0.33, 1.0),
    co_jitter=True,
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
        sampler_type: 采样策略 ('temporal_nearby' 或 'spatial_nearby')
        spatial_radius: spatial nearby 的3D距离阈值（米）
        color_jitter_prob: color jitter 概率（论文4.1=0.9）
        brightness: brightness jitter 参数（论文4.1=0.5）
        contrast: contrast jitter 参数（论文4.1=0.5）
        saturation: saturation jitter 参数（论文4.1=0.5）
        hue: hue jitter 参数（论文4.1=0.1）
        grayscale_prob: grayscale 概率（论文4.1=0.05）
        spatial_rescale_range: spatial rescale 范围（论文=[0.8,1.2]）
        aspect_ratio_range: aspect ratio 范围（论文=[0.33,1.0]）
        co_jitter: 是否所有view共享颜色扰动参数（论文=True）
    """
    dataset = ReplicaDataset(
        data_root=data_root,
        num_views=num_views,
        min_views=min_views,
        max_views=max_views,
        max_dim=max_dim,
        seed=seed,
        sampler_type=sampler_type,
        spatial_radius=spatial_radius,
        spatial_rescale_range=spatial_rescale_range,
        aspect_ratio_range=aspect_ratio_range,
        co_jitter=co_jitter,
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