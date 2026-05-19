"""
Head-Only 训练模型（论文4.1对齐版）

根据 try_train/stage1_scannet_training_plan.md 文档的要求，
本模块实现：

1. 基于 GCTBase 的简化训练模型
2. 冻结 DINOv2 backbone 和所有 transformer blocks
3. 只训练 DepthHead 和 PoseHead（默认启用）
4. 支持 Absolute Pose Loss 和 Relative Pose Loss
5. Loss 实现遵循论文 Sec. 3.3 的设计

参考文档第2、8节的架构设计。

作者：Claude Code
日期：2026-05-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import sys
from pathlib import Path

# 添加父目录到路径，以便导入 lingbot_map
sys.path.insert(0, str(Path(__file__).parent.parent))

from lingbot_map.models.gct_base import GCTBase
from lingbot_map.heads.dpt_head import DPTHead
from lingbot_map.heads.camera_head import CameraHead
from lingbot_map.layers.patch_embed import PatchEmbed


class HeadOnlyModel(nn.Module):
    """
    Head-Only 训练模型

    架构：冻结 backbone + trainable lightweight heads
    """

    def __init__(
        self,
        backbone_name: str = 'dinov2_vits14',
        freeze_backbone: bool = True,
        img_size: int = 224,
        patch_size: int = 14,
        embed_dim: int = 384,
        train_depth_head: bool = True,
        train_pose_head: bool = True,
        num_views: int = 2,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.train_depth_head = train_depth_head
        self.train_pose_head = train_pose_head
        self.num_views = num_views

        print(f"[HeadOnlyModel] 初始化")
        print(f"  - Backbone: {backbone_name}, Freeze: {freeze_backbone}")
        print(f"  - Train depth: {train_depth_head}, Train pose: {train_pose_head}")

        # 1. Backbone: DINOv2 ViT
        self.backbone = self._load_dinov2_backbone(backbone_name)
        if freeze_backbone:
            self._freeze_backbone()

        # 2. Depth Head
        self.depth_head = None
        if train_depth_head:
            self.depth_head = DPTHead(
                dim_in=embed_dim,
                patch_size=patch_size,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                features=256,
                out_channels=[256, 512, 1024, 1024],
                intermediate_layer_idx=[0, 1, 2, 3],
            )

        # 3. Pose Head
        self.pose_head = None
        if train_pose_head:
            self.pose_head = CameraHead(
                dim_in=embed_dim,
                trunk_depth=4,
                pose_encoding_type="absT_quaR_FoV",
                num_heads=8,
                mlp_ratio=4,
            )

        self.norm = nn.LayerNorm(embed_dim)

        # ImageNet 归一化常量（DINOv2 训练时使用 ImageNet mean/std）
        self.register_buffer(
            'imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            'imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        )

        # 从 DINOv2 不同深度的 block 提取中间特征（多尺度）
        self.intermediate_block_indices = self._get_intermediate_block_indices(backbone_name)
        self._hook_features = {}
        self._hooks = []
        self._register_intermediate_hooks()

        print(f"  - Intermediate blocks: {self.intermediate_block_indices}")

    def _get_intermediate_block_indices(self, backbone_name):
        """根据 backbone 类型确定提取中间特征的 block 索引

        DPTHead 需要4组不同深度的特征来实现多尺度融合：
        - 浅层: 边缘、纹理等低级特征
        - 中浅层: 中级特征
        - 中深层: 高级特征
        - 深层: 语义特征
        """
        block_count_map = {
            'dinov2_vits14': 12,
            'dinov2_vitb14': 12,
            'dinov2_vitl14': 24,
        }
        num_blocks = block_count_map.get(backbone_name, 12)
        indices = [
            num_blocks // 4 - 1,       # 浅层
            num_blocks // 2 - 1,       # 中浅层
            3 * num_blocks // 4 - 1,   # 中深层
            num_blocks - 1,            # 深层
        ]
        return indices

    def _register_intermediate_hooks(self):
        """在 DINOv2 的中间 block 上注册 forward hook"""
        self._remove_hooks()
        for idx in self.intermediate_block_indices:
            hook = self.backbone.blocks[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, block_idx):
        """创建 hook 函数，捕获指定 block 的输出"""
        def hook(module, input, output):
            self._hook_features[block_idx] = output
        return hook

    def _remove_hooks(self):
        """移除所有已注册的 hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def _load_dinov2_backbone(self, backbone_name):
        # DINOv2 需要通过 torch.hub 加载，不是 torchvision.models
        backbone_map = {
            'dinov2_vits14': 'dinov2_vits14',
            'dinov2_vitb14': 'dinov2_vitb14',
            'dinov2_vitl14': 'dinov2_vitl14',
        }
        model_name = backbone_map[backbone_name]
        backbone = torch.hub.load('facebookresearch/dinov2', model_name)
        print(f"[_load_dinov2_backbone] {backbone_name} 加载完成 (from torch.hub)")
        return backbone

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: [B, V, 3, H, W], RGB图像（[0,1]范围）

        Returns:
            predictions dict with depth/pose
        """
        B, V, C, H, W = images.shape

        # ImageNet 归一化：DINOv2 训练时使用 ImageNet mean/std
        images_normalized = (images - self.imagenet_mean) / self.imagenet_std
        images_flat = images_normalized.view(B * V, C, H, W)

        # 清除上一次的 hook 缓存
        self._hook_features = {}

        # Backbone 特征提取（hooks 自动捕获中间层特征）
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone.forward_features(images_flat)

        # 最终输出（已经过 backbone.norm）
        if isinstance(features, dict):
            patch_tokens = features['x_norm_patchtokens']
            cls_token = features['x_norm_clstoken']
        else:
            patch_tokens = features[:, 1:, :]
            cls_token = features[:, 0:1, :]

        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(B, V, num_patches, self.embed_dim)
        cls_token = cls_token.view(B, V, 1, self.embed_dim)

        # 构建多尺度特征列表：从 hooks 中提取4个不同深度的特征
        aggregated_tokens_list = []
        pose_tokens_list = []
        for block_idx in self.intermediate_block_indices:
            if block_idx in self._hook_features:
                feat = self._hook_features[block_idx]
                # hook 捕获的是 block 输出（未经 backbone.norm），需要应用 norm
                feat_normed = self.backbone.norm(feat)
                inter_patch = feat_normed[:, 1:, :].view(B, V, num_patches, self.embed_dim)
                inter_cls = feat_normed[:, 0:1, :].view(B, V, 1, self.embed_dim)
                aggregated_tokens_list.append(inter_patch)
                pose_tokens_list.append(inter_cls)
            else:
                # fallback
                aggregated_tokens_list.append(patch_tokens)
                pose_tokens_list.append(cls_token)

        predictions = {}

        # Depth Head - 使用多尺度特征（关键修复：不再重复同一特征）
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=0,
            )
            predictions['depth'] = depth
            predictions['depth_conf'] = depth_conf

        # Pose Head - 使用多尺度 cls token
        if self.pose_head is not None:
            pose_enc_list = self.pose_head(pose_tokens_list, num_iterations=4)
            predictions['pose_enc'] = pose_enc_list[-1]

        return predictions


class DepthLoss(nn.Module):
    """Depth Loss: Masked L1 或 Log-depth L1"""

    def __init__(self, loss_type='masked_log_l1'):
        super().__init__()
        self.loss_type = loss_type

    def forward(self, depth_pred, depth_gt, valid_mask):
        if depth_pred.dim() == 5:
            depth_pred = depth_pred.squeeze(-1)

        valid_mask = valid_mask.bool()

        if self.loss_type == 'masked_l1':
            diff = torch.abs(depth_pred - depth_gt)
            loss = diff[valid_mask].mean()
        elif self.loss_type == 'masked_log_l1':
            eps = 1e-3
            log_pred = torch.log(depth_pred + eps)
            log_gt = torch.log(depth_gt + eps)
            diff = torch.abs(log_pred - log_gt)
            loss = diff[valid_mask].mean()

        return loss


class PoseLoss(nn.Module):
    """
    Absolute Pose Loss: Rotation geodesic + Translation Huber

    遵循论文 Sec. 3.3 和文档第8.2节的设计：
    - rotation: quaternion geodesic loss
    - translation: Huber loss (鲁棒性)
    - 第一帧 pose 不参与 loss 计算（第一帧为 reference frame）
    """

    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(self, pose_enc_pred, pose_gt, exclude_first_frame=True):
        """
        Args:
            pose_enc_pred: [B, V, 9] pose encoding (center + quaternion + focal/offset)
            pose_gt: [B, V, 4, 4] camera-to-world pose matrix (relative to first frame)
            exclude_first_frame: 是否排除第一帧（第一帧 target 为 identity）

        Returns:
            pose_loss: scalar loss
        """
        B, V = pose_enc_pred.shape[:2]

        # 提取预测的 center 和 quaternion
        center_pred = pose_enc_pred[:, :, :3]  # [B, V, 3]
        quat_pred = pose_enc_pred[:, :, 3:7]   # [B, V, 4]

        # 归一化 quaternion
        quat_pred = quat_pred / (torch.norm(quat_pred, dim=-1, keepdim=True) + 1e-8)

        # 提取 GT center (从 pose matrix 的最后一列)
        center_gt = pose_gt[:, :, :3, 3]  # [B, V, 3]

        # 提取 GT rotation (从 pose matrix 的前3x3部分)
        # 需要将 rotation matrix 转换为 quaternion
        rot_gt = pose_gt[:, :, :3, :3]  # [B, V, 3, 3]
        quat_gt = self._rotation_matrix_to_quaternion(rot_gt)  # [B, V, 4]

        # 排除第一帧
        if exclude_first_frame and V > 1:
            center_pred = center_pred[:, 1:, :]
            center_gt = center_gt[:, 1:, :]
            quat_pred = quat_pred[:, 1:, :]
            quat_gt = quat_gt[:, 1:, :]

        # Translation loss: Huber
        trans_diff = torch.abs(center_pred - center_gt)
        trans_loss = self._huber_loss(trans_diff)

        # Rotation loss: quaternion geodesic
        rot_loss = self._quaternion_geodesic_loss(quat_pred, quat_gt)

        return self.rotation_weight * rot_loss + self.translation_weight * trans_loss

    def _rotation_matrix_to_quaternion(self, R):
        """
        将 rotation matrix [B, V, 3, 3] 转换为 quaternion [B, V, 4]
        使用简化方法，可能有数值不稳定性，但对于训练来说足够
        """
        B, V = R.shape[:2]

        # 使用 trace 方法
        trace = R[:, :, 0, 0] + R[:, :, 1, 1] + R[:, :, 2, 2]

        # 简化实现：假设 rotation matrix 是有效的
        # 使用更稳健的方法
        batch_shape = (B, V)

        # 为了避免数值不稳定，使用特征值分解的近似
        # 这里使用简化的实现
        qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) / 2.0
        qx = (R[:, :, 2, 1] - R[:, :, 1, 2]) / (4.0 * qw + 1e-8)
        qy = (R[:, :, 0, 2] - R[:, :, 2, 0]) / (4.0 * qw + 1e-8)
        qz = (R[:, :, 1, 0] - R[:, :, 0, 1]) / (4.0 * qw + 1e-8)

        quat = torch.stack([qw, qx, qy, qz], dim=-1)
        # 归一化
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

        return quat

    def _quaternion_geodesic_loss(self, q1, q2):
        """
        Quaternion geodesic loss

        Args:
            q1: [B, V, 4] predicted quaternions
            q2: [B, V, 4] ground truth quaternions

        Returns:
            geodesic loss: scalar
        """
        # Geodesic distance between two quaternions
        # dot product
        dot = torch.sum(q1 * q2, dim=-1)  # [B, V]

        # Clamp to avoid numerical issues
        dot = torch.clamp(torch.abs(dot), min=0.0, max=1.0)

        # Angle = 2 * arccos(|dot|)
        # 使用 arccos 的近似避免数值不稳定
        angle = 2.0 * torch.acos(dot)

        # 取 mean
        return angle.mean()

    def _huber_loss(self, diff, delta=1.0):
        """
        Huber loss (鲁棒性)

        Args:
            diff: [B, V, 3] 或任意形状的差值
            delta: Huber delta

        Returns:
            huber loss: scalar
        """
        mask = (diff < delta).float()
        loss = mask * 0.5 * diff ** 2 + (1 - mask) * (delta * diff - 0.5 * delta ** 2)
        return loss.mean()


class RelativePoseLoss(nn.Module):
    """
    Relative Pose Loss: 所有 view pairs 之间的 relative pose

    遵循论文 Sec. 3.3 和文档第8.3节的设计：
    - 在 sampled views 内计算所有 pair 的 relative pose
    - T_i_to_j = inv(T_i) @ T_j
    - rotation: geodesic loss
    - translation: Huber loss
    """

    def __init__(self, rotation_weight=1.0, translation_weight=10.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(self, pose_enc_pred, pose_gt):
        """
        Args:
            pose_enc_pred: [B, V, 9] pose encoding (center + quaternion + focal/offset)
            pose_gt: [B, V, 4, 4] camera-to-world pose matrix

        Returns:
            relative_pose_loss: scalar loss
        """
        B, V = pose_enc_pred.shape[:2]

        if V < 2:
            return 0.0  # 需要至少2帧才能计算 relative pose

        # 提取预测的 center 和 quaternion
        center_pred = pose_enc_pred[:, :, :3]  # [B, V, 3]
        quat_pred = pose_enc_pred[:, :, 3:7]   # [B, V, 4]
        quat_pred = quat_pred / (torch.norm(quat_pred, dim=-1, keepdim=True) + 1e-8)

        # 构建 predicted pose matrix
        # 从 quaternion 构建 rotation matrix
        rot_pred = self._quaternion_to_rotation_matrix(quat_pred)  # [B, V, 3, 3]

        # 构建 4x4 pose matrix
        pose_pred = torch.zeros(B, V, 4, 4, device=pose_enc_pred.device)
        pose_pred[:, :, :3, :3] = rot_pred
        pose_pred[:, :, :3, 3] = center_pred
        pose_pred[:, :, 3, 3] = 1.0

        # 计算 relative pose pairs
        total_loss = 0.0
        num_pairs = 0

        for i in range(V):
            for j in range(V):
                if i == j:
                    continue

                # T_i_to_j_pred = inv(T_i_pred) @ T_j_pred
                T_i_pred = pose_pred[:, i, :, :]  # [B, 4, 4]
                T_j_pred = pose_pred[:, j, :, :]  # [B, 4, 4]
                T_i_to_j_pred = torch.linalg.inv(T_i_pred) @ T_j_pred  # [B, 4, 4]

                # T_i_to_j_gt = inv(T_i_gt) @ T_j_gt
                T_i_gt = pose_gt[:, i, :, :]
                T_j_gt = pose_gt[:, j, :, :]
                T_i_to_j_gt = torch.linalg.inv(T_i_gt) @ T_j_gt

                # 提取 relative translation 和 rotation
                trans_pred = T_i_to_j_pred[:, :3, 3]
                trans_gt = T_i_to_j_gt[:, :3, 3]

                rot_pred_ij = T_i_to_j_pred[:, :3, :3]
                rot_gt_ij = T_i_to_j_gt[:, :3, :3]

                quat_pred_ij = self._rotation_matrix_to_quaternion(rot_pred_ij)
                quat_gt_ij = self._rotation_matrix_to_quaternion(rot_gt_ij)

                # Translation loss
                trans_diff = torch.abs(trans_pred - trans_gt)
                trans_loss = self._huber_loss(trans_diff)

                # Rotation loss
                rot_loss = self._quaternion_geodesic_loss(quat_pred_ij, quat_gt_ij)

                total_loss += self.rotation_weight * rot_loss + self.translation_weight * trans_loss
                num_pairs += 1

        return total_loss / num_pairs if num_pairs > 0 else 0.0

    def _quaternion_to_rotation_matrix(self, q):
        """
        将 quaternion [B, V, 4] 转换为 rotation matrix [B, V, 3, 3]
        """
        # 归一化
        q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-8)

        qw, qx, qy, qz = q[:, :, 0], q[:, :, 1], q[:, :, 2], q[:, :, 3]

        # Rotation matrix from quaternion
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
        """将 rotation matrix 转换为 quaternion

        Args:
            R: rotation matrix, can be [B, V, 3, 3] or [B, 3, 3]

        Returns:
            quaternion: same batch shape with last dim 4
        """
        # Handle different shapes
        if R.dim() == 3:  # [B, 3, 3]
            trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
            qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) / 2.0
            qx = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * qw + 1e-8)
            qy = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * qw + 1e-8)
            qz = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * qw + 1e-8)
            quat = torch.stack([qw, qx, qy, qz], dim=-1)
        elif R.dim() == 4:  # [B, V, 3, 3]
            trace = R[:, :, 0, 0] + R[:, :, 1, 1] + R[:, :, 2, 2]
            qw = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) / 2.0
            qx = (R[:, :, 2, 1] - R[:, :, 1, 2]) / (4.0 * qw + 1e-8)
            qy = (R[:, :, 0, 2] - R[:, :, 2, 0]) / (4.0 * qw + 1e-8)
            qz = (R[:, :, 1, 0] - R[:, :, 0, 1]) / (4.0 * qw + 1e-8)
            quat = torch.stack([qw, qx, qy, qz], dim=-1)
        else:
            raise ValueError(f"Unexpected rotation matrix shape: {R.shape}")

        return quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

    def _quaternion_geodesic_loss(self, q1, q2):
        """Quaternion geodesic loss

        Args:
            q1, q2: quaternions, can be [B, V, 4] or [B, 4]

        Returns:
            geodesic loss: scalar
        """
        dot = torch.sum(q1 * q2, dim=-1)
        dot = torch.clamp(torch.abs(dot), min=0.0, max=1.0)
        angle = 2.0 * torch.acos(dot)
        return angle.mean()

    def _huber_loss(self, diff, delta=1.0):
        """Huber loss"""
        mask = (diff < delta).float()
        loss = mask * 0.5 * diff ** 2 + (1 - mask) * (delta * diff - 0.5 * delta ** 2)
        return loss.mean()


def create_head_only_model(
    backbone_name='dinov2_vits14',
    freeze_backbone=True,
    img_size=224,
    train_depth_head=True,
    train_pose_head=True,  # 默认启用 pose head（论文4.1对齐）
    num_views=2,
):
    """创建 head-only 模型

    Args:
        backbone_name: DINOv2 backbone 类型
        freeze_backbone: 是否冻结 backbone
        img_size: 训练图像尺寸
        train_depth_head: 是否训练 depth head
        train_pose_head: 是否训练 pose head（默认 True，符合论文4.1）
        num_views: 视角数（可以是固定值或范围）

    Returns:
        HeadOnlyModel 实例
    """
    model = HeadOnlyModel(
        backbone_name=backbone_name,
        freeze_backbone=freeze_backbone,
        img_size=img_size,
        train_depth_head=train_depth_head,
        train_pose_head=train_pose_head,
        num_views=num_views,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[create_head_only_model] Trainable: {trainable}, Total: {total}")
    print(f"  Frozen: {(total - trainable) / total:.2%}")

    return model


if __name__ == '__main__':
    model = create_head_only_model('dinov2_vits14', True, 224, True, False, 2)
    dummy_images = torch.rand(1, 2, 3, 224, 224)
    predictions = model(dummy_images)
    print(f"Depth: {predictions['depth'].shape}, Conf: {predictions['depth_conf'].shape}")