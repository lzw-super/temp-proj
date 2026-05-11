# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
DPTHead - Dense Prediction Transformer Head 密集预测头

本模块实现了 DPT (Dense Prediction Transformer) 架构，用于从 Vision Transformer
特征中生成密集预测（深度、3D点云等）。

灵感来源: https://github.com/DepthAnything/Depth-Anything-V2
参考论文: "Vision Transformers for Dense Prediction" (https://arxiv.org/abs/2103.13413)

╔════════════════════════════════════════════════════════════════════╗
║                    DPTHead 架构概览                                 ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  【核心思想】                                                      ║
║                                                                    ║
║  ViT 的输出是 patch-level 的 token 序列，                           ║
║  DPT 通过多尺度特征融合将这些 token 转换为像素级的密集预测。          ║
║                                                                    ║
║  【处理流程】                                                      ║
║                                                                    ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │ 输入: aggregated_tokens_list                                 │   ║
║  │       - 来自 Aggregator 的多层 token 特征                    │   ║
║  │       - 通常取 4 个中间层的特征                               │   ║
║  │       - 每层特征: [B, S, P, C]                                │   ║
║  │         B: batch size                                        │   ║
║  │         S: sequence length (帧数)                            │   ║
║  │         P: patch 数量 = (H/patch_size) * (W/patch_size)     │   ║
║  │         C: 特征维度                                           │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║      │                                                              ║
║      ↓                                                              ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │ 步骤1: 提取 patch tokens                                      │   ║
║  │                                                               │   ║
║  │   tokens = aggregated_tokens_list[layer_idx]                 │   ║
║  │   patch_tokens = tokens[:, :, patch_start_idx:]              │   ║
║  │                                                               │   ║
║  │   patch_start_idx: 跳过 camera/register 等特殊 token         │   ║
║  │   只保留 patch tokens                                          │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║      │                                                              ║
║      ↓                                                              ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │ 步骤2: 重塑为 2D 特征图                                        │   ║
║  │                                                               │   ║
║  │   # [B, S, P, C] -> [B*S, P, C]                              │   ║
║  │   x = x.reshape(B * S, -1, C)                                │   ║
║  │                                                               │   ║
║  │   # LayerNorm 标准化                                          │   ║
║  │   x = norm(x)                                                 │   ║
║  │                                                               │   ║
║  │   # [B*S, P, C] -> [B*S, C, H_patch, W_patch]               │   ║
║  │   x = x.permute(0, 2, 1).reshape(B*S, C, patch_h, patch_w)  │   ║
║  │                                                               │   ║
║  │   patch_h = H // patch_size (e.g., 378/14 = 27)             │   ║
║  │   patch_w = W // patch_size (e.g., 518/14 = 37)             │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║      │                                                              ║
║      ↓                                                              ║
║  ┌─────────────────────────────────────────────────────────────┐   ║
║  │ 步骤3: 通道投影                                                │   ║
║  │                                                               │   ║
║  │   x = project[dpt_idx](x)                                    │   ║
║  │   # Conv2d(C_in, C_out, kernel=1)                            │   ║
║  │                                                               │   ║
║  │   投影到不同的通道数:                                          │   ║
║  │   - layer 0: C_in -> 256                                     │   ║
║  │   - layer 1: C_in -> 512                                     │   ║
║  │   - layer 2: C_in -> 1024                                    │   ║
║  │   - layer 3: C_in -> 1024                                    │   ║
║  │                                                               │   ║
║  │   不同通道数代表不同尺度的特征                                  │   ║
║  └─────────────────────────────────────────────────────────────┘   ║
║      │                                                              ║
║      ↓                                                              ╌───────────────────┐
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤4: 分辨率调整 (Resize)                                    │   ║│
║  │                                                               │   ║│
║  │   resize_layers[dpt_idx](x)                                  │   ║│
║  │                                                               │   ║│
║  │   - layer 0: ConvTranspose2d(4x 上采样)                      │   ║│ 108x148
║  │     kernel=4, stride=4                                       │   ║│ (27*4, 37*4)
║  │     output: 108 x 148                                        │   ║│
║  │                                                               │   ║│
║  │   - layer 1: ConvTranspose2d(2x 上采样)                      │   ║│ 54x74
║  │     kernel=2, stride=2                                       │   ║│ (27*2, 37*2)
║  │     output: 54 x 74                                          │   ║│
║  │                                                               │   ║│
║  │   - layer 2: Identity (保持不变)                             │   ║│ 27x37
║  │     output: 27 x 37                                          │   ║│
║  │                                                               │   ║│
║  │   - layer 3: Conv2d(2x 下采样)                               │   ║│ 14x19
║  │     kernel=3, stride=2, padding=1                            │   ║│ (27/2, 37/2)
║  │     output: 14 x 19                                          │   ║│
║  │                                                               │   ║│
║  │   所有层调整为不同分辨率，为融合做准备                          │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤5: 位置编码 (可选)                                        │   ║│
║  │                                                               │   ║│
║  │   if pos_embed:                                              │   ║│
║  │       x = _apply_pos_embed(x, W, H)                          │   ║│
║  │                                                               │   ║│
║  │   生成 UV grid 并转换为位置嵌入                                 │   ║│
║  │   添加空间位置信息                                             │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  【多尺度特征融合】                                                │   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤6: 特征融合 (scratch_forward)                             │   ║│
║  │                                                               │   ║│
║  │   # 通道调整                                                   │   ║│
║  │   layer_1_rn = layer1_rn(layer_1)  # 256 -> 256              │   ║│
║  │   layer_2_rn = layer2_rn(layer_2)  # 512 -> 256              │   ║│
║  │   layer_3_rn = layer3_rn(layer_3)  # 1024 -> 256            │   ║│
║  │   layer_4_rn = layer4_rn(layer_4)  # 1024 -> 256            │   ║│
║  │                                                               │   ║│
║  │   所有特征统一到 256 通道                                       │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤7: 自底向上融合 (RefineNet)                               │   ║│
║  │                                                               │   ║│
║  │   # 从最粗 (layer_4) 到最细 (layer_1)                        │   ║│
║  │                                                               │   ║│
║  │   # RefineNet4: layer_4_rn (14x19) -> 上采样到 27x37        │   ║│
║  │   out = refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])   │   ║│
║  │                                                               │   ║│
║  │   # RefineNet3: out + layer_3_rn -> 上采样到 54x74          │   ║│
║  │   out = refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])║│
║  │                                                               │   ║│
║  │   # RefineNet2: out + layer_2_rn -> 上采样到 108x148        │   ║│
║  │   out = refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])║│
║  │                                                               │   ║│
║  │   # RefineNet1: out + layer_1_rn -> 保持 108x148            │   ║│
║  │   out = refinenet1(out, layer_1_rn)                          │   ║│
║  │                                                               │   ║│
║  │   【RefineNet 内部】                                          │   ║│
║  │   ┌─────────────────────────────────────────────────────────┐║│
║  │   │ FeatureFusionBlock:                                      │║│
║  │   │                                                          │║│
║  │   │ 1. 如果有两个输入:                                        │║│
║  │   │    res = ResidualConvUnit(xs[1])                         │║│
║  │   │    output = xs[0] + res  # 残差连接                      │║│
║  │   │                                                          │║│
║  │   │ 2. ResidualConvUnit(output)                              │║│
║  │   │                                                          │║│
║  │   │ 3. 上采样 (2x 或指定 size)                                │║│
║  │   │                                                          │║│
║  │   │ 4. 1x1 Conv 输出                                          │║│
║  │   └─────────────────────────────────────────────────────────┘║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤8: 输出卷积                                                │   ║│
║  │                                                               │   ║│
║  │   out = output_conv1(out)                                    │   ║│
║  │   # Conv2d(256, 128, kernel=3)                               │   ║│
║  │                                                               │   ║│
║  │   output_conv2:                                               │   ║│
║  │   Conv2d(128, 32, kernel=3) -> ReLU                          │   ║│
║  │   Conv2d(32, output_dim, kernel=1)                           │   ║│
║  │                                                               │   ║│
║  │   output_dim:                                                 │   ║│
║  │   - depth prediction: 1                                       │   ║│
║  │   - point prediction: 3                                       │   ║│
║  │   - depth + point: 4                                          │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤9: 双线性插值到目标分辨率                                  │   ║│
║  │                                                               │   ║│
║  │   out = interpolate(out, (H/down_ratio, W/down_ratio))      │   ║│
║  │                                                               │   ║│
║  │   默认 down_ratio=1:                                          │   ║│
║  │   output 分辨率 = 原图像分辨率                                  │   ║│
║  │   H = 378, W = 518                                            │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 步骤10: 激活函数                                               │   ║│
║  │                                                               │   ║│
║  │   preds, conf = activate_head(out, activation, conf_activation)║│
║  │                                                               │   ║│
║  │   activation="inv_log":                                       │   ║│
║  │   │   depth = 1 / log(1 + exp(pred))                         │   ║│
║  │   │   转换为实际深度值                                          │   ║│
║  │                                                               │   ║│
║  │   conf_activation="expp1":                                    │   ║│
║  │   │   conf = exp(pred) + 1                                    │   ║│
║  │   │   确保置信度 >= 1                                          │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║      │                                                              ║│
║      ↓                                                              ║│
║  ┌─────────────────────────────────────────────────────────────┐   ║│
║  │ 输出:                                                         │   ║│
║  │                                                               │   ║│
║  │   preds: [B, S, H, W, output_dim]                            │   ║│
║  │   │   - 深度预测 [B, S, H, W, 1]                              │   ║│
║  │   │   - 3D点预测 [B, S, H, W, 3]                              │   ║│
║  │                                                               │   ║│
║  │   conf: [B, S, H, W]                                          │   ║│
║  │   │   - 深度置信度                                             │   ║│
║  │   │   - 点云置信度                                             │   ║│
║  └─────────────────────────────────────────────────────────────┘   ║│
║                                                                    ║
║  【分辨率变化示意图】                                              ║
║                                                                    ║
║  ViT Patch 分辨率:      27 x 37  (layer 2)                        ║
║  │                     54 x 74  (layer 1 上采样后)               ║
║  │                    108 x 148 (layer 0 上采样后)               ║
║  │                      14 x 19  (layer 3 下采样后)              ║
║                                                                    ║
║  融合后分辨率:          108 x 148 (最细层)                        ║
║  最终输出分辨率:        378 x 518 (原图分辨率, down_ratio=1)      ║
║                                                                    ║
║  【多尺度融合的优势】                                              ║
║                                                                    ║
║  1. 低分辨率特征 (layer 3/4):                                      ║
║  │   - 全局上下文信息                                              ║
║  │   - 大范围空间关系                                              ║
║  │   - 鲁棒但粗糙                                                  ║
║                                                                    ║
║  2. 高分辨率特征 (layer 0/1):                                      ║
║  │   - 局部细节信息                                                ║
║  │   - 精确空间定位                                                ║
║  │   - 精细但可能噪声                                              ║
║                                                                    ║
║  3. 融合效果:                                                      ║
║  │   - 结合全局和局部                                              ║
║  │   - 既有大范围一致性又有精细细节                                 ║
║  │   - 密集预测的关键                                              ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class DPTHead(nn.Module):
    """
    【DPT 密集预测头】

    DPT (Dense Prediction Transformer) Head 用于从 Vision Transformer 特征
    生成像素级的密集预测（深度、3D点云等）。

    【核心架构】

    DPT 通过多尺度特征融合将 patch-level 的 ViT 特征转换为像素级预测:
    1. 从多个 Transformer 层提取特征
    2. 调整各层特征到不同分辨率
    3. 自底向上融合多尺度特征
    4. 生成最终密集预测

    【使用场景】

    在 GCT 模型中，DPTHead 用于:
    - _predict_depth: 预测深度图
    - _predict_points: 预测 3D 世界坐标点

    【参数说明】

    Args:
        dim_in (int): 输入维度 (channels).
            - 来自 aggregated_tokens 的特征维度
            - 通常为 2048 (frame + global 特征拼接)

        patch_size (int, optional): ViT patch 大小. 默认 14.
            - DINOv2 ViT-L/14 的 patch_size

        output_dim (int, optional): 输出通道数. 默认 4.
            - 1: 仅深度
            - 3: 仅3D点
            - 4: 深度 + 3D点

        activation (str, optional): 输出激活类型. 默认 "inv_log".
            - "inv_log": 深度激活 1/log(1+exp(x))
            - 将预测转换为实际深度值

        conf_activation (str, optional): 置信度激活类型. 默认 "expp1".
            - "expp1": exp(x) + 1
            - 确保置信度 >= 1

        features (int, optional): 中间特征通道数. 默认 256.
            - 所有融合层统一到这个通道数

        out_channels (List[int], optional): 各层输出通道数.
            - 默认 [256, 512, 1024, 1024]
            - 对应 layer 0, 1, 2, 3

        intermediate_layer_idx (List[int], optional): 使用的 Transformer 层索引.
            - 默认 [0, 1, 2, 3]
            - 取第 0, 1, 2, 3 个 block 的特征

        pos_embed (bool, optional): 是否使用位置编码. 默认 True.
            - 在投影后添加空间位置嵌入

        feature_only (bool, optional): 是否只返回特征. 默认 False.
            - True: 返回融合特征，不经过输出头
            - False: 返回最终预测

        down_ratio (int, optional): 输出下采样比例. 默认 1.
            - 1: 输出与原图分辨率相同
            - 2: 输出为原图一半分辨率

    【模块组成】

    1. norm: LayerNorm 标准化
    2. projects: 4个 1x1 Conv 投影层
    3. resize_layers: 4个分辨率调整层
    4. scratch: 特征融合模块
       - layer1_rn ~ layer4_rn: 通道调整 Conv
       - refinenet1 ~ refinenet4: FeatureFusionBlock
       - output_conv1, output_conv2: 输出 Conv

    【输出格式】

    Returns:
        - 如果 feature_only=True: 特征图 [B, S, C, H, W]
        - 否则: Tuple(predictions, confidence)
          - predictions: [B, S, H, W, output_dim]
          - confidence: [B, S, H, W]
    """

    def __init__(
        self,
        dim_in: int,  # 输入维度 (来自 aggregated_tokens)
        patch_size: int = 14,  # ViT patch 大小 (DINOv2: 14)
        output_dim: int = 4,  # 输出通道数 (深度1 + 点3 = 4)
        activation: str = "inv_log",  # 深度激活类型
        conf_activation: str = "expp1",  # 置信度激活类型
        features: int = 256,  # 融合特征通道数
        out_channels: List[int] = [256, 512, 1024, 1024],  # 各层输出通道
        intermediate_layer_idx: List[int] = [0, 1, 2, 3],  # 使用的 Transformer 层
        pos_embed: bool = True,  # 是否使用位置编码
        feature_only: bool = False,  # 是否只返回特征
        down_ratio: int = 1,  # 输出下采样比例
    ) -> None:
        super(DPTHead, self).__init__()

        # ════════════════════════════════════════════════════════════════════
        # 保存配置参数
        # ════════════════════════════════════════════════════════════════════
        self.patch_size = patch_size  # 用于计算 patch_h, patch_w
        self.activation = activation  # 深度激活函数类型
        self.conf_activation = conf_activation  # 置信度激活函数类型
        self.pos_embed = pos_embed  # 是否添加空间位置编码
        self.feature_only = feature_only  # 是否只输出特征
        self.down_ratio = down_ratio  # 输出分辨率下采样比例
        self.intermediate_layer_idx = intermediate_layer_idx  # 使用哪些层的特征

        # ════════════════════════════════════════════════════════════════════
        # LayerNorm: 标准化输入 tokens
        # ════════════════════════════════════════════════════════════════════
        self.norm = nn.LayerNorm(dim_in)

        # ════════════════════════════════════════════════════════════════════
        # 投影层: 将 tokens 投影到不同通道数
        # ════════════════════════════════════════════════════════════════════
        # 4 个 1x1 Conv，对应 4 个 Transformer 层
        # 每个投影到不同的通道数，代表不同尺度的特征
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0)
                for oc in out_channels
            ]
        )

        # ════════════════════════════════════════════════════════════════════
        # 分辨率调整层: 上采样或下采样
        # ════════════════════════════════════════════════════════════════════
        # 将各层特征调整到不同分辨率，为融合做准备
        #
        # resize_layers[0]: 4x 上采样 (27x37 -> 108x148)
        #   ConvTranspose2d kernel=4, stride=4
        # resize_layers[1]: 2x 上采样 (27x37 -> 54x74)
        #   ConvTranspose2d kernel=2, stride=2
        # resize_layers[2]: 保持不变 (27x37)
        #   Identity
        # resize_layers[3]: 2x 下采样 (27x37 -> 14x19)
        #   Conv2d kernel=3, stride=2, padding=1
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )

        # ════════════════════════════════════════════════════════════════════
        # Scratch: 特征融合模块
        # ════════════════════════════════════════════════════════════════════
        # 包含通道调整层和融合块
        self.scratch = _make_scratch(out_channels, features, expand=False)

        # ════════════════════════════════════════════════════════════════════
        # RefineNet: 自底向上特征融合块
        # ════════════════════════════════════════════════════════════════════
        # 从最粗层 (layer_4) 到最细层 (layer_1) 逐步融合
        #
        # refinenet4: 只处理 layer_4，无残差输入 (has_residual=False)
        # refinenet3: 融合 layer_3 和上层输出
        # refinenet2: 融合 layer_2 和上层输出
        # refinenet1: 融合 layer_1 和上层输出
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # ════════════════════════════════════════════════════════════════════
        # 输出卷积头
        # ════════════════════════════════════════════════════════════════════
        head_features_1 = features  # 256
        head_features_2 = 32  # 最终输出前的中间通道

        if feature_only:
            # 只输出特征模式
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            # 正常输出模式
            # output_conv1: 256 -> 128
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2  # 128

            # output_conv2: 128 -> 32 -> output_dim
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        【前向传播】支持分帧处理的密集预测

        处理流程支持两种模式:
        1. 批量处理: 所有帧一次处理
        2. 分帧处理: 每次处理 frames_chunk_size 帧，降低显存占用

        Args:
            aggregated_tokens_list (List[Tensor]): 多层 token 特征列表
                - 每个元素: [B, S, P, 2C]
                - 来自 Aggregator 的多个 Transformer block

            images (Tensor): 输入图像 [B, S, 3, H, W], 范围 [0, 1]
                - 用于获取图像尺寸信息

            patch_start_idx (int): patch token 起始索引
                - 用于跳过 camera/register 等特殊 token
                - 只提取 patch tokens

            frames_chunk_size (int, optional): 分帧处理的帧数
                - None 或 >= S: 批量处理所有帧
                - 否则: 每次处理 frames_chunk_size 帧

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - feature_only=True: 特征图 [B, S, C, H, W]
                - 否则: Tuple(predictions, confidence)
                  - predictions: [B, S, H, W, output_dim]
                  - confidence: [B, S, H, W]
        """
        B, _, _, H, W = images.shape
        S = aggregated_tokens_list[0].shape[1]  # 帧数

        # ════════════════════════════════════════════════════════════════════
        # 批量处理模式
        # ════════════════════════════════════════════════════════════════════
        # 如果 frames_chunk_size 未指定或大于总帧数，一次性处理所有帧
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # ════════════════════════════════════════════════════════════════════
        # 分帧处理模式
        # ════════════════════════════════════════════════════════════════════
        # 逐块处理帧，降低显存峰值使用
        assert frames_chunk_size > 0

        # 存储各块的结果
        all_preds = []
        all_conf = []

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # 处理当前帧块
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)

        # 沿序列维度拼接各块结果
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        【核心前向实现】处理特定帧块的密集预测

        这是 DPTHead 的核心实现，处理多层 ViT 特征并生成密集预测。

        ╔════════════════════════════════════════════════════════════════════╗
        ║               _forward_impl 处理流程详解                           ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【步骤1: 多层特征提取和重塑】                                      ║
        ║                                                                    ║
        ║  for layer_idx in [0, 1, 2, 3]:                                    ║
        ║      │   # 提取 patch tokens                                        ║
        ║      │   x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]║
        ║      │   # 跳过 camera/register 等 token                            ║
        ║      │                                                              ║
        ║      │   # 分帧处理: 只取指定范围                                    ║
        ║      │   if frames_start_idx is not None:                          ║
        ║      │       x = x[:, frames_start_idx:frames_end_idx]             ║
        ║      │                                                              ║
        ║      │   # 重塑为 2D                                                ║
        ║      │   B, S = x.shape[0], x.shape[1]                             ║
        ║      │   x = x.reshape(B * S, -1, C)  # 合并 batch 和 frame        ║
        ║      │                                                              ║
        ║      │   # LayerNorm                                                ║
        ║      │   x = norm(x)                                                 ║
        ║      │                                                              ║
        ║      │   # 转换为 2D 特征图                                          ║
        ║      │   x = x.permute(0, 2, 1)  # [B*S, C, P]                     ║
        ║      │   x = x.reshape(B*S, C, patch_h, patch_w)                   ║
        ║      │   # [B*S, C, 27, 37]                                         ║
        ║      │                                                              ║
        ║      │   # 通道投影                                                  ║
        ║      │   x = projects[dpt_idx](x)                                  ║
        ║      │   # [B*S, out_channels[dpt_idx], 27, 37]                    ║
        ║      │                                                              ║
        ║      │   # 位置编码 (可选)                                           ║
        ║      │   if pos_embed:                                              ║
        ║      │       x = _apply_pos_embed(x, W, H)                         ║
        ║      │                                                              ║
        ║      │   # 分辨率调整                                                ║
        ║      │   x = resize_layers[dpt_idx](x)                             ║
        ║      │   # dpt_idx=0: [B*S, 256, 108, 148]  (4x 上采样)            ║
        ║      │   # dpt_idx=1: [B*S, 512, 54, 74]    (2x 上采样)            ║
        ║      │   # dpt_idx=2: [B*S, 1024, 27, 37]  (保持)                  ║
        ║      │   # dpt_idx=3: [B*S, 1024, 14, 19]  (2x 下采样)             ║
        ║      │                                                              ║
        ║      │   out.append(x)                                              ║
        ║                                                                    ║
        ║  【步骤2: 多尺度特征融合】                                         ║
        ║                                                                    ║
        ║  out = scratch_forward(out)                                        ║
        ║  │   # 输入: 4个不同分辨率的特征图                                  ║
        ║  │   # 输出: 256 通道的融合特征图                                   ║
        ║                                                                    ║
        ║  【步骤3: 双线性插值到目标分辨率】                                  ║
        ║                                                                    ║
        ║  out = interpolate(out, (H/down_ratio, W/down_ratio))             ║
        ║  │   # 默认: 输出到原图分辨率                                       ║
        ║  │   # H=378, W=518                                                ║
        ║                                                                    ║
        ║  【步骤4: 再次位置编码 (可选)】                                    ║
        ║                                                                    ║
        ║  if pos_embed:                                                     ║
        ║      out = _apply_pos_embed(out, W, H)                            ║
        ║                                                                    ║
        ║  【步骤5: 输出处理】                                               ║
        ║                                                                    ║
        ║  if feature_only:                                                  ║
        ║      return out.view(B, S, C, H, W)  # 只返回特征                  ║
        ║                                                                    ║
        ║  # 输出卷积                                                        ║
        ║  out = output_conv2(out)                                           ║
        ║  │   # Conv(256->128) -> ReLU -> Conv(128->32) -> ReLU             ║
        ║  │   # -> Conv(32->output_dim)                                     ║
        ║                                                                    ║
        ║  # 激活函数                                                        ║
        ║  preds, conf = activate_head(out, activation, conf_activation)     ║
        ║  │   # preds: [B*S, H, W, output_dim]                              ║
        ║  │   # conf: [B*S, H, W]                                           ║
        ║                                                                    ║
        ║  # 重塑回 [B, S, ...]                                              ║
        ║  preds = preds.view(B, S, H, W, output_dim)                        ║
        ║  conf = conf.view(B, S, H, W)                                      ║
        ║                                                                    ║
        ║  return preds, conf                                                ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            aggregated_tokens_list: 多层 token 特征列表
            images: 输入图像 [B, S, 3, H, W]
            patch_start_idx: patch token 起始索引
            frames_start_idx: 帧块起始索引 (分帧处理)
            frames_end_idx: 帧块结束索引 (分帧处理)

        Returns:
            特征图或 (预测, 置信度) tuple
        """

        B, _, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size  # 27, 37

        out = []  # 存储各层处理后的特征
        dpt_idx = 0

        # ════════════════════════════════════════════════════════════════════
        # 步骤1: 处理每个 Transformer 层的特征
        # ════════════════════════════════════════════════════════════════════
        for layer_idx in self.intermediate_layer_idx:
            # 提取 patch tokens (跳过特殊 token)
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]

            # 分帧处理: 只取指定范围
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx]

            B, S = x.shape[0], x.shape[1]

            # 合并 batch 和 frame 维度: [B, S, P, C] -> [B*S, P, C]
            x = x.reshape(B * S, -1, x.shape[-1])

            # LayerNorm 标准化
            x = self.norm(x)

            # 转换为 2D 特征图: [B*S, P, C] -> [B*S, C, patch_h, patch_w]
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            # 通道投影: 1x1 Conv
            x = self.projects[dpt_idx](x)

            # 位置编码 (可选)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            # 分辨率调整
            x = self.resize_layers[dpt_idx](x)

            out.append(x)
            dpt_idx += 1

        # ════════════════════════════════════════════════════════════════════
        # 步骤2: 多尺度特征融合
        # ════════════════════════════════════════════════════════════════════
        out = self.scratch_forward(out)

        # ════════════════════════════════════════════════════════════════════
        # 步骤3: 双线性插值到目标分辨率
        # ════════════════════════════════════════════════════════════════════
        # 默认输出到原图分辨率 (down_ratio=1)
        out = custom_interpolate(
            out,
            (int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        # ════════════════════════════════════════════════════════════════════
        # 步骤4: 再次位置编码 (可选)
        # ════════════════════════════════════════════════════════════════════
        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        # ════════════════════════════════════════════════════════════════════
        # 步骤5: 输出处理
        # ════════════════════════════════════════════════════════════════════
        if self.feature_only:
            # 只返回特征模式
            return out.view(B, S, *out.shape[1:])

        # 输出卷积头
        out = self.scratch.output_conv2(out)

        # 激活函数: 转换为实际深度值和置信度
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation)

        # 重塑回原始形状 [B, S, H, W, output_dim] 和 [B, S, H, W]
        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        return preds, conf

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        【应用空间位置编码】

        在特征图上添加 UV 位置嵌入，增强空间位置信息。

        处理流程:
        1. 创建 UV grid: [patch_w, patch_h, 2] 归一化坐标
        2. 转换为位置嵌入: [patch_w, patch_h, C]
        3. 缩放并添加到特征图

        Args:
            x: 特征图 [B*S, C, patch_h, patch_w]
            W: 原图宽度
            H: 原图高度
            ratio: 位置编码缩放比例 (默认 0.1)

        Returns:
            添加位置编码后的特征图
        """
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]

        # 创建 UV grid: 归一化坐标 [patch_w, patch_h, 2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)

        # 转换为位置嵌入: [patch_w, patch_h, C]
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])

        # 缩放位置编码
        pos_embed = pos_embed * ratio

        # 转换为 [1, C, patch_h, patch_w] 并广播
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)

        return x + pos_embed

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        【多尺度特征融合】

        自底向上融合多尺度特征，从最粗层到最细层。

        ╔══════════════════════════════════════════════════════════════════╗
        ║               scratch_forward 融合流程                            ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║                                                                  ║
        ║  输入:                                                           ║
        ║  │   layer_1: [B*S, 256, 108, 148]   (最细，高分辨率)           ║
        ║  │   layer_2: [B*S, 512, 54, 74]                                  ║
        ║  │   layer_3: [B*S, 1024, 27, 37]                                 ║
        ║  │   layer_4: [B*S, 1024, 14, 19]    (最粗，低分辨率)           ║
        ║                                                                  ║
        ║  【通道调整】                                                     ║
        ║                                                                  ║
        ║  layer_1_rn = Conv2d(256 -> 256, kernel=3)                      ║
        ║  layer_2_rn = Conv2d(512 -> 256, kernel=3)                      ║
        ║  layer_3_rn = Conv2d(1024 -> 256, kernel=3)                     ║
        ║  layer_4_rn = Conv2d(1024 -> 256, kernel=3)                     ║
        ║                                                                  ║
        ║  所有特征统一到 256 通道                                          ║
        ║                                                                  ║
        ║  【自底向上融合】                                                 ║
        ║                                                                  ║
        ║  # RefineNet4: 从最粗层开始                                       ║
        ║  out = refinenet4(layer_4_rn, size=layer_3.shape[2:])           ║
        ║  │   # 输入: layer_4_rn (14x19)                                  ║
        ║  │   # 上采样到 27x37                                             ║
        ║  │   # 输出: [B*S, 256, 27, 37]                                   ║
        ║                                                                  ║
        ║  # RefineNet3: 融合 layer_3                                       ║
        ║  out = refinenet3(out, layer_3_rn, size=layer_2.shape[2:])      ║
        ║  │   # 输入: out (27x37) + layer_3_rn (27x37)                    ║
        ║  │   # 上采样到 54x74                                             ║
        ║  │   # 输出: [B*S, 256, 54, 74]                                   ║
        ║                                                                  ║
        ║  # RefineNet2: 融合 layer_2                                       ║
        ║  out = refinenet2(out, layer_2_rn, size=layer_1.shape[2:])      ║
        ║  │   # 输入: out (54x74) + layer_2_rn (54x74)                    ║
        ║  │   # 上采样到 108x148                                           ║
        ║  │   # 输出: [B*S, 256, 108, 148]                                 ║
        ║                                                                  ║
        ║  # RefineNet1: 融合 layer_1                                       ║
        ║  out = refinenet1(out, layer_1_rn)                               ║
        ║  │   # 输入: out (108x148) + layer_1_rn (108x148)                ║
        ║  │   # 保持 108x148                                               ║
        ║  │   # 输出: [B*S, 256, 108, 148]                                 ║
        ║                                                                  ║
        ║  # 输出卷积                                                       ║
        ║  out = output_conv1(out)                                         ║
        ║  │   # Conv2d(256 -> 128, kernel=3)                              ║
        ║  │   # 输出: [B*S, 128, 108, 148]                                 ║
        ║                                                                  ║
        ║  返回: [B*S, 128, 108, 148]                                       ║
        ║                                                                  ║
        ╚══════════════════════════════════════════════════════════════════╝

        【融合策略】

        1. 低分辨率特征先处理:
           │   - 全局上下文信息
           │   - 大范围空间关系

        2. 逐步添加高分辨率特征:
           │   - 局部细节信息
           │   - 精细空间定位

        3. 残差连接:
           │   - 每次融合保留上一层信息
           │   - 稳定训练过程

        Args:
            features: 4个不同分辨率的特征图列表

        Returns:
            融合后的特征图 [B*S, 128, H_fusion, W_fusion]
        """
        layer_1, layer_2, layer_3, layer_4 = features

        # ════════════════════════════════════════════════════════════════════
        # 通道调整: 所有层统一到 256 通道
        # ════════════════════════════════════════════════════════════════════
        layer_1_rn = self.scratch.layer1_rn(layer_1)  # 256 -> 256
        layer_2_rn = self.scratch.layer2_rn(layer_2)  # 512 -> 256
        layer_3_rn = self.scratch.layer3_rn(layer_3)  # 1024 -> 256
        layer_4_rn = self.scratch.layer4_rn(layer_4)  # 1024 -> 256

        # ════════════════════════════════════════════════════════════════════
        # 自底向上融合: 最粗 -> 最细
        # ════════════════════════════════════════════════════════════════════
        # RefineNet4: 处理最粗层 (14x19 -> 27x37)
        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4  # 释放内存

        # RefineNet3: 融合 layer_3 (27x37 -> 54x74)
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        # RefineNet2: 融合 layer_2 (54x74 -> 108x148)
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        # RefineNet1: 融合 layer_1 (保持 108x148)
        out = self.scratch.refinenet1(out, layer_1_rn)
        del layer_1_rn, layer_1

        # 输出卷积: 256 -> 128
        out = self.scratch.output_conv1(out)
        return out


################################################################################
# 辅助模块 - 特征融合和通道调整
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    """
    【创建特征融合块】

    构建 FeatureFusionBlock 用于多尺度特征融合。

    Args:
        features: 特征通道数 (256)
        size: 目标输出尺寸 (可选)
        has_residual: 是否有残差输入 (True)
        groups: 分组卷积的组数 (1)

    Returns:
        FeatureFusionBlock 模块
    """
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    """
    【创建 Scratch 模块】

    Scratch 包含通道调整层，将不同通道的特征统一到同一通道数。

    Args:
        in_shape: 各层输入通道数 [256, 512, 1024, 1024]
        out_shape: 输出通道数 (256)
        groups: 分组卷积组数 (1)
        expand: 是否扩展通道 (False)

    Returns:
        包含 layer1_rn ~ layer4_rn 的 Module
    """
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    # expand=True 时通道数逐层增加
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    # 各层通道调整 Conv (统一到 out_shape)
    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """
    【残差卷积单元】

    包含两个 3x3 Conv 的残差块，用于特征融合块中。

    结构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   input x                                                       │
    │      │                                                          │
    │      ├──────────────────────────────────────────────────────┐  │
    │      │                                                      │  │
    │      ↓                                                      │  │
    │   activation (ReLU)                                          │  │
    │      │                                                      │  │
    │      ↓                                                      │  │
    │   Conv1 (3x3)                                                │  │
    │      │                                                      │  │
    │      ↓                                                      │  │
    │   activation (ReLU)                                          │  │
    │      │                                                      │  │
    │      ↓                                                      │  │
    │   Conv2 (3x3)                                                │  │
    │      │                                                      │  │
    │      ↓                                                      │  │
    │   output = Conv2_output + x  (残差连接)─────────────────────┘  │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    Args:
        features: 特征通道数
        activation: 激活函数 (ReLU)
        bn: 是否使用 BatchNorm (False)
        groups: 分组卷积组数 (1)
    """

    def __init__(self, features, activation, bn, groups=1):
        super().__init__()

        self.bn = bn
        self.groups = groups

        # 两个 3x3 Conv
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        【前向传播】残差卷积处理

        Args:
            x: 输入特征图

        Returns:
            残差连接后的输出
        """
        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)  # 残差连接


class FeatureFusionBlock(nn.Module):
    """
    【特征融合块】

    融合两个不同尺度的特征，并上采样到目标分辨率。

    结构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │   输入1 (来自上层): xs[0]                                       │
    │   输入2 (当前层特征): xs[1] (可选)                              │
    │                                                                 │
    │   if has_residual and len(xs) == 2:                            │
    │       res = ResidualConvUnit(xs[1])  # 处理输入2                │
    │       output = xs[0] + res         # 残差连接                   │
    │   else:                                                         │
    │       output = xs[0]              # 只有输入1                   │
    │                                                                 │
    │   output = ResidualConvUnit(output)  # 再次处理                 │
    │                                                                 │
    │   output = interpolate(output, size or scale_factor=2)         │
    │   │   # 上采样到目标分辨率                                       │
    │                                                                 │
    │   output = out_conv(output)        # 1x1 Conv 输出              │
    │                                                                 │
    │   返回融合并上采样的特征                                          │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘

    【参数说明】

    Args:
        features: 特征通道数
        activation: 激活函数 (ReLU)
        deconv: 是否使用反卷积上采样 (False)
        bn: 是否使用 BatchNorm (False)
        expand: 是否扩展通道 (False)
        align_corners: 插值对齐方式 (True)
        size: 目标输出尺寸 (可选)
        has_residual: 是否有残差输入 (True)
        groups: 分组卷积组数 (1)
    """

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        # 输出 Conv: 1x1
        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        # 残差卷积单元
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """
        【前向传播】特征融合和上采样

        Args:
            xs: 输入特征列表
                - xs[0]: 来自上层的特征
                - xs[1]: 当前层特征 (可选)
            size: 目标输出尺寸 (覆盖 self.size)

        Returns:
            融合并上采样的特征图
        """
        output = xs[0]

        # 如果有残差输入，进行融合
        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        # 再次残差卷积处理
        output = self.resConfUnit2(output)

        # 上采样
        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}  # 默认 2x 上采样
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)

        # 输出 Conv
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    【安全插值函数】

    自定义插值函数，避免 INT_MAX 限制导致的溢出问题。

    当输出尺寸非常大时，nn.functional.interpolate 可能因
    INT_MAX (1610612736) 限制而失败。此函数通过分块处理
    大尺寸输入来避免此问题。

    Args:
        x: 输入张量 [N, C, H, W]
        size: 目标尺寸 (H_out, W_out)
        scale_factor: 缩放因子 (如果 size 为 None)
        mode: 插值模式 ("bilinear")
        align_corners: 角点对齐方式 (True)

    Returns:
        插值后的张量
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736

    # 计算总元素数
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    # 如果超过 INT_MAX，分块处理
    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

class DPTHead_Update(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead_Update, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block_slam(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block_slam(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block_slam(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block_slam(features, use_bn)
        
        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )
    
    def forward(self, out_features, patch_h, patch_w, return_intermediate=True):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        if return_intermediate:
            return out, path_1, path_2, path_3, path_4
        else:
            out = self.scratch.output_conv2(out)
            return out

def _make_fusion_block_slam(features, use_bn, size=None):
    return FeatureFusionBlock_slam(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class FeatureFusionBlock_slam(nn.Module):
    """Feature fusion block.
    """

    def __init__(
        self, 
        features, 
        activation, 
        deconv=False, 
        bn=False, 
        expand=False, 
        align_corners=True,
        size=None
    ):
        """Init.
        
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_slam, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        
        output = self.out_conv(output)

        return output