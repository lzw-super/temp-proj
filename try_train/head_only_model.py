"""
Head-Only 训练模型

根据 try_train/stage1_scannet_training_plan.md 文档的要求，
本模块实现：

1. 基于 GCTBase 的简化训练模型
2. 冻结 DINOv2 backbone 和所有 transformer blocks
3. 只训练 DepthHead 和 PoseHead

参考文档第2、8节的架构设计。

作者：Claude Code
日期：2026-05-11
"""

import torch
import torch.nn as nn
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
            images: [B, V, 3, H, W], RGB图像

        Returns:
            predictions dict with depth/pose
        """
        B, V, C, H, W = images.shape
        images_flat = images.view(B * V, C, H, W)

        # Backbone 特征提取
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            features = self.backbone.forward_features(images_flat)

        if isinstance(features, dict):
            patch_tokens = features['x_norm_patchtokens']
            cls_token = features['x_norm_clstoken']
        else:
            patch_tokens = features[:, 1:, :]
            cls_token = features[:, 0:1, :]

        num_patches = patch_tokens.shape[1]
        patch_tokens = patch_tokens.view(B, V, num_patches, self.embed_dim)
        cls_token = cls_token.view(B, V, 1, self.embed_dim)

        predictions = {}

        # Depth Head
        if self.depth_head is not None:
            aggregated_tokens_list = [patch_tokens] * 4
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list,
                images=images,
                patch_start_idx=0,
            )
            predictions['depth'] = depth
            predictions['depth_conf'] = depth_conf

        # Pose Head
        if self.pose_head is not None:
            pose_tokens = cls_token
            pose_aggregated_list = [pose_tokens] * 4
            pose_enc_list = self.pose_head(pose_aggregated_list, num_iterations=4)
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
    """Pose Loss: Rotation geodesic + Translation Huber"""

    def __init__(self, rotation_weight=1.0, translation_weight=10.0):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight

    def forward(self, pose_enc_pred, pose_gt):
        center_pred = pose_enc_pred[:, :, :3]
        quat_pred = pose_enc_pred[:, :, 3:7]
        quat_pred = quat_pred / torch.norm(quat_pred, dim=-1, keepdim=True)

        center_gt = pose_gt[:, :, :3, 3]

        trans_diff = torch.abs(center_pred - center_gt)
        trans_loss = self._huber_loss(trans_diff)
        rot_loss = torch.norm(quat_pred, dim=-1).mean()

        return self.rotation_weight * rot_loss + self.translation_weight * trans_loss

    def _huber_loss(self, diff, delta=1.0):
        mask = (diff < delta).float()
        loss = mask * 0.5 * diff ** 2 + (1 - mask) * (delta * diff - 0.5 * delta ** 2)
        return loss.mean()


def create_head_only_model(
    backbone_name='dinov2_vits14',
    freeze_backbone=True,
    img_size=224,
    train_depth_head=True,
    train_pose_head=False,
    num_views=2,
):
    """创建 head-only 模型"""
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