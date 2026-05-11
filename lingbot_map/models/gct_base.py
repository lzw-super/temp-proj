"""
GCTBase - GCT模型实现的基类

提供共享功能:
- 预测头 (camera/相机, depth/深度, point/点云)
- 前向传播结构
- 模型库混合类 (PyTorchModelHubMixin)

【架构概览】

本类采用抽象基类设计模式，定义了GCT模型的核心架构框架:
1. Aggregator (聚合器): 负责图像特征提取和跨帧注意力计算
   - 子类需实现 _build_aggregator() 方法
2. Prediction Heads (预测头):
   - Camera Head: 相机位姿预测 (9维编码: 中心+四元数+焦距)
   - Depth Head: 深度图预测
   - Point Head: 3D世界坐标点预测
   - Local Point Head: 相机坐标系下的3D点预测

【关键抽象方法】

子类必须实现以下方法:
1. _build_aggregator(): 构建聚合器模块
   - GCTStream 使用 AggregatorStream (带KV cache的流式处理)
2. _build_camera_head(): 构建相机预测头
   - GCTStream 使用 CameraCausalHead (带因果注意力)
3. _aggregate_features(): 特征聚合核心流程
   - 调用 aggregator 处理图像序列
   - 返回多尺度特征列表和patch起始索引

【forward() 执行流程】

forward() -> _aggregate_features() -> aggregator.forward()
         -> _predict_camera() -> CameraCausalHead
         -> _predict_depth() -> DPTHead
         -> _predict_points() -> DPTHead

详细流程图:

┌─────────────────────────────────────────────────────────────────────┐
│                         forward() 总体流程                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  images [B,S,3,H,W]                                                 │
│       │                                                             │
│       ↓                                                             │
│  _normalize_input() ─ 标准化输入形状                                │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ _aggregate_features() 【核心特征聚合】                        │ │
│  │                                                               │ │
│  │  调用 self.aggregator(images, ...)                            │ │
│  │                                                               │ │
│  │  AggregatorStream.forward() 详细流程:                        │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ 1. _embed_images()                                      │ │ │
│  │  │    - 图像归一化 (ResNet mean/std)                        │ │ │
│  │  │    - DINOv2 ViT-L/14 Patch Embedding                    │ │ │
│  │  │      输入: [B*S, 3, H, W]                                │ │ │
│  │  │      输出: [B*S, P_patch, C]                             │ │ │
│  │  │    - 添加特殊tokens:                                     │ │ │
│  │  │      [camera_token] [register_tokens] [scale_token]     │ │ │
│  │  │    - 输出: [B*S, P, C] 其中 P = 特殊token数 + patch数   │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ 2. _get_positions()                                     │ │ │
│  │  │    - 生成2D位置编码 (用于RoPE旋转位置编码)               │ │ │
│  │  │    - 输出: [B*S, P, 2]                                   │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │  ┌─────────────────────────────────────────────────────────┐ │ │
│  │  │ 3. 交替执行 frame_blocks 和 global_blocks               │ │ │
│  │  │                                                          │ │ │
│  │  │    for block_group in range(aa_block_num):              │ │ │
│  │  │        frame_blocks:                                     │ │ │
│  │  │          - 帧内自注意力 (每帧独立处理)                   │ │ │
│  │  │          - 输入/输出: [B*S, P, C]                        │ │ │
│  │  │          - 使用标准 Transformer Block + RoPE            │ │ │
│  │  │                                                          │ │ │
│  │  │        global_blocks:                                     │ │ │
│  │  │          - 跨帧因果注意力                                │ │ │
│  │  │          - 使用 FlashInferBlock 或 SDPABlock            │ │ │
│  │  │          - 带 KV cache 实现高效流式推理                  │ │ │
│  │  │          - scale token 控制前N帧的双向注意力             │ │ │
│  │  │                                                          │ │ │
│  │  │    输出: frame_inter + global_inter [B,S,P,2C]          │ │ │
│  │  └─────────────────────────────────────────────────────────┘ │ │
│  │                                                               │ │
│  │  返回: (aggregated_tokens_list, patch_start_idx)             │ │
│  │        - aggregated_tokens_list: 多个block输出的列表         │ │
│  │          selected_idx=[4,11,17,23] 指定输出哪些block        │ │
│  │        - patch_start_idx: patch tokens的起始索引             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ _predict_camera() 【相机位姿预测】                            │ │
│  │                                                               │ │
│  │  CameraCausalHead 处理流程:                                   │ │
│  │  - 提取 camera token: tokens[:,:,0]                           │ │
│  │  - 迭代优化位姿 (4次迭代):                                    │ │
│  │    CameraBlock + 因果注意力 + KV cache                        │ │
│  │  - 输出: pose_enc [B,S,9]                                     │ │
│  │    9维编码 = [center(3), quaternion(4), focal+offset(2)]     │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ _predict_depth() 【深度预测】                                 │ │
│  │                                                               │ │
│  │  DPTHead (Dense Prediction Transformer):                      │ │
│  │  - 从多尺度特征解码深度图                                     │ │
│  │  - 使用 selected_idx=[4,11,17,23] 的特征                     │ │
│  │  - 输出: depth [B,S,H,W,1], depth_conf [B,S,H,W]             │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ _predict_points() 【3D点云预测】                              │ │
│  │                                                               │ │
│  │  DPTHead (output_dim=4):                                      │ │
│  │  - 输出: world_points [B,S,H,W,3] + conf [B,S,H,W]           │ │
│  │  - activation="inv_log": 逆向log变换得到真实坐标              │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  predictions = {pose_enc, depth, depth_conf,                       │
│                 world_points, world_points_conf, images}            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【特殊Token说明】

每帧的token序列结构:
┌────────────────────────────────────────────────────────────────────┐
│ [camera_token] [register_tokens(4个)] [scale_token] [patch_tokens] │
│     idx=0          idx=1~4            idx=5         idx=6~end      │
│                                                                    │
│ camera_token: 用于预测相机位姿                                      │
│ register_tokens: DINOv2的register tokens，提高特征质量             │
│ scale_token: 控制尺度估计帧的双向注意力                             │
│ patch_tokens: 图像patch的特征，数量 = (H/14)*(W/14)                │
│                                                                    │
│ patch_start_idx = 1 + num_register + 1 = 6                        │
│ num_special_tokens = 1 + 4 + 1 = 6                                │
└────────────────────────────────────────────────────────────────────┘

【Scale Token工作原理】

scale_frames (默认8帧) 的处理:
- 这些帧之间使用双向注意力（而非因果注意力）
- 通过 scale_token 的激活状态实现
- 目的: 建立稳定的全局坐标系和尺度基准
- 类似 SLAM 的初始化阶段

┌────────────────────────────────────────────────────────────────────┐
│ Scale Frames (帧0~7):                                              │
│                                                                    │
│   scale_token 激活 -> 双向注意力                                   │
│   帧0可以看到帧7，帧7可以看到帧0                                   │
│                                                                    │
│ 后续帧 (帧8+):                                                     │
│                                                                    │
│   scale_token 非激活 -> 因果注意力                                 │
│   帧t只能看到帧0~t                                                 │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

【KV Cache机制】

FlashInfer/SDPA KV cache 实现:
- 存储历史帧的 Key 和 Value
- 新帧只需与缓存中的KV计算注意力
- 支持滑动窗口: 超过窗口大小时自动驱逐旧帧
- 保留 scale frames 和 special tokens 不被驱逐

┌────────────────────────────────────────────────────────────────────┐
│ KV Cache 工作流程:                                                 │
│                                                                    │
│ [K1,V1] [K2,V2] [K3,V3] ... [Kn,Vn]  <- 已缓存的KV                │
│         └─────────────────────────                                 │
│                    + 当前帧的 Q, K, V                              │
│                    ↓                                               │
│              Attention(Q, [K_cache + K_curr], [V_cache + V_curr]) │
│                                                                    │
│ 如果是关键帧: 将当前KV存入缓存                                     │
│ 如果不是关键帧: 只参与计算，不存入缓存                             │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from huggingface_hub import PyTorchModelHubMixin

from lingbot_map.heads.dpt_head import DPTHead
from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
from lingbot_map.utils.geometry import closed_form_inverse_se3

logger = logging.getLogger(__name__)


class GCTBase(nn.Module, PyTorchModelHubMixin, ABC):
    """
    GCT模型实现的基类

    处理共享组件:
    - 预测头 (camera, depth, point)
    - 前向传播结构
    - 输入标准化

    子类必须实现:
    - _build_aggregator(): 创建模式特定的聚合器
    - _build_camera_head(): 创建模式特定的相机预测头

    【设计模式】
    本类使用抽象基类模式，将架构的核心部分定义为抽象方法，
    允许子类（如GCTStream）实现具体的聚合器和相机预测头逻辑。

    【参数说明】
    Architecture parameters (架构参数):
    - img_size: 输入图像尺寸，默认518（ViT-L/14的标准尺寸）
    - patch_size: Patch大小，默认14（DINOv2 ViT-L使用14x14 patch）
    - embed_dim: 嵌入维度，默认1024（ViT-L的隐藏层维度）
    - patch_embed: Patch embedding类型，使用DINOv2 ViT-L/14
    - disable_global_rope: 是否禁用全局注意力中的RoPE

    Head configuration (预测头配置):
    - enable_camera: 是否启用相机位姿预测
    - enable_point: 是否启用世界坐标系3D点预测
    - enable_local_point: 是否启用相机坐标系3D点预测
    - enable_depth: 是否启用深度预测
    - enable_track: 是否启用跟踪功能

    Camera head sliding window:
    - enable_camera_sliding_window: 相机头是否使用滑动窗口注意力

    3D RoPE:
    - enable_3d_rope: 是否启用3D旋转位置编码（用于时间维度的位置编码）

    Context Parallelism (已弃用):
    - enable_ulysses_cp: 保持兼容性但不使用

    Normalization:
    - enable_normalize: 是否启用特征归一化
    - pred_normalization: 预测结果归一化（训练时使用）
    - pred_normalization_detach_scale: 归一化时是否detach scale参数

    Gradient checkpointing:
    - use_gradient_checkpoint: 是否使用梯度检查点（节省内存）
    """

    def __init__(
        self,
        # Architecture parameters (架构参数)
        img_size: int = 518,  # 输入图像尺寸，ViT-L/14的标准尺寸
        patch_size: int = 14,  # Patch大小，DINOv2使用14x14
        embed_dim: int = 1024,  # 嵌入维度，ViT-L隐藏层维度
        patch_embed: str = 'dinov2_vitl14_reg',  # Patch embedding类型
        disable_global_rope: bool = False,  # 是否禁用全局注意力RoPE
        # Head configuration (预测头配置)
        enable_camera: bool = True,  # 启用相机位姿预测
        enable_point: bool = True,  # 启用世界坐标3D点预测
        enable_local_point: bool = False,  # 启用相机坐标3D点预测
        enable_depth: bool = True,  # 启用深度预测
        enable_track: bool = False,  # 启用跟踪功能
        # Camera head sliding window (相机头滑动窗口)
        enable_camera_sliding_window: bool = False,
        # 3D RoPE (3D旋转位置编码)
        enable_3d_rope: bool = False,
        # Context Parallelism (上下文并行，已弃用但保留兼容)
        enable_ulysses_cp: bool = False,
        # Normalization (归一化设置)
        enable_normalize: bool = False,
        # Prediction normalization (预测归一化)
        pred_normalization: bool = False,
        pred_normalization_detach_scale: bool = False,
        # Gradient checkpointing (梯度检查点)
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()

        # ════════════════════════════════════════════════════════════════════
        # 存储配置参数
        # ════════════════════════════════════════════════════════════════════

        # 架构参数
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = patch_embed
        self.disable_global_rope = disable_global_rope

        # 上下文并行（在独立包中禁用）
        self.enable_ulysses_cp = False  # CP disabled in standalone package
        self.enable_normalize = enable_normalize
        self.pred_normalization = pred_normalization
        self.pred_normalization_detach_scale = pred_normalization_detach_scale
        self.use_gradient_checkpoint = use_gradient_checkpoint

        # 预测头启用标志
        self.enable_camera = enable_camera
        self.enable_point = enable_point
        self.enable_local_point = enable_local_point
        self.enable_depth = enable_depth
        self.enable_track = enable_track
        self.enable_camera_sliding_window = enable_camera_sliding_window
        self.enable_3d_rope = enable_3d_rope

        # ════════════════════════════════════════════════════════════════════
        # 构建核心模块
        # ════════════════════════════════════════════════════════════════════

        # 聚合器: 子类特定实现 (如 AggregatorStream)
        # 聚合器负责:
        # 1. 图像特征提取 (DINOv2 patch embedding)
        # 2. 特殊token添加 (camera, register, scale)
        # 3. 帧内注意力 (frame_blocks)
        # 4. 跨帧注意力 (global_blocks with KV cache)
        self.aggregator = self._build_aggregator()

        # 预测头: 子类特定实现
        # Camera Head: 预测相机位姿 (9维编码)
        self.camera_head = self._build_camera_head() if enable_camera else None
        # Point Head: 预测世界坐标系3D点
        self.point_head = self._build_point_head() if enable_point else None
        # Local Point Head: 预测相机坐标系3D点
        self.local_point_head = self._build_local_point_head() if enable_local_point else None
        # Depth Head: 预测深度图
        self.depth_head = self._build_depth_head() if enable_depth else None

    # ════════════════════════════════════════════════════════════════════════
    # 抽象方法：子类必须实现
    # ════════════════════════════════════════════════════════════════════════

    @abstractmethod
    def _build_aggregator(self) -> nn.Module:
        """
        【抽象方法】构建聚合器模块

        子类实现说明:
        - GCTStream 实现: 返回 AggregatorStream
          - 使用 FlashInferBlock 或 SDPABlock
          - 支持 KV cache 的流式推理
          - 包含 frame_blocks (帧内注意力) 和 global_blocks (跨帧注意力)

        聚合器的核心功能:
        1. Patch Embedding (DINOv2 ViT-L/14):
           - 输入图像 [B*S, 3, H, W]
           - 输出 patch tokens [B*S, P_patch, C]
           - P_patch = (H/14) * (W/14) 个patch

        2. 特殊Token处理:
           - camera_token: 用于位姿预测
           - register_tokens: 提高特征质量
           - scale_token: 控制尺度帧的双向注意力

        3. 注意力处理:
           - frame_blocks: 每帧独立的自注意力
           - global_blocks: 跨帧因果注意力（带KV cache）

        Returns:
            聚合器模块 (nn.Module)
        """
        pass

    @abstractmethod
    def _build_camera_head(self) -> nn.Module:
        """
        【抽象方法】构建相机位姿预测头

        子类实现说明:
        - GCTStream 实现: 返回 CameraCausalHead
          - 使用因果注意力进行位姿预测
          - 支持迭代优化（4次迭代）
          - 使用 KV cache 提高效率

        Camera Head 输出:
        - pose_enc: 9维位姿编码 [B, S, 9]
          - [:3]: 相机中心位置 (x, y, z)
          - [3:7]: 四元数旋转 (qw, qx, qy, qz)
          - [7:9]: 焦距和偏移参数

        位姿解码:
        - 使用 pose_encoding_to_extri_intri() 转换为外参/内参矩阵

        Returns:
            相机预测头模块 (nn.Module)
        """
        pass

    # ════════════════════════════════════════════════════════════════════════
    # 预测头构建方法（基类提供的默认实现）
    # ════════════════════════════════════════════════════════════════════════

    def _build_depth_head(self) -> nn.Module:
        """
        【深度预测头构建】使用DPTHead进行深度图预测

        DPTHead (Dense Prediction Transformer Head):
        - 源自DPT架构，用于密集预测任务
        - 从多尺度特征解码深度图
        - 使用 selected_idx=[4,11,17,23] 的block输出

        参数说明:
        - dim_in: 输入维度 = 2*embed_dim (frame + global 特征拼接)
        - patch_size: 14 (与patch embedding一致)
        - output_dim: 2 (深度值 + 置信度)
        - activation: "exp" (指数激活，确保深度为正)
        - conf_activation: "expp1" (置信度激活)

        Returns:
            DPTHead 模块
        """
        return DPTHead(
            dim_in=2 * self.embed_dim,  # 输入维度：frame+global特征拼接
            patch_size=self.patch_size,
            output_dim=2,  # 深度值 + 置信度
            activation="exp",  # 指数激活：depth = e^(pred)，确保正值
            conf_activation="expp1"  # 置信度激活：conf = e^(pred) + 1
        )

    def _build_point_head(self) -> nn.Module:
        """
        【3D点云预测头构建】预测世界坐标系下的3D点坐标

        DPTHead配置:
        - output_dim: 4 (x, y, z 坐标 + 置信度)
        - activation: "inv_log" (逆向log变换)

        输出:
        - world_points: [B, S, H, W, 3] 世界坐标3D点
        - world_points_conf: [B, S, H, W] 置信度

        坐标变换说明:
        - 世界坐标系由scale frames建立的基准定义
        - 点云可以直接用于3D重建和可视化

        Returns:
            DPTHead 模块
        """
        return DPTHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            output_dim=4,  # 3D坐标(x,y,z) + 置信度
            activation="inv_log",  # 逆向log变换：pts = 1 / log(pred + 1)
            conf_activation="expp1"
        )

    def _build_local_point_head(self) -> nn.Module:
        """
        【局部点云预测头构建】预测相机坐标系下的3D点坐标

        与 point_head 类似，但输出在相机坐标系:
        - cam_points: [B, S, H, W, 3] 相机坐标3D点
        - cam_points_conf: [B, S, H, W] 置信度

        用途:
        - 可用于验证深度预测的一致性
        - 可用于相机坐标系下的局部重建

        Returns:
            DPTHead 模块
        """
        return DPTHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            output_dim=4,
            activation="inv_log",
            conf_activation="expp1"
        )

    # ════════════════════════════════════════════════════════════════════════
    # 输入标准化方法
    # ════════════════════════════════════════════════════════════════════════

    def _normalize_input(self, images: torch.Tensor, query_points=None):
        """
        【输入标准化】确保输入形状符合模型要求

        处理逻辑:
        1. 如果 images 是 [S, 3, H, W] (无batch维度):
           - 添加 batch 维度变成 [1, S, 3, H, W]
        2. 如果 query_points 是 [N, 2] (无batch维度):
           - 添加 batch 维度变成 [1, N, 2]

        Args:
            images: 输入图像，可能是 [S,3,H,W] 或 [B,S,3,H,W]
            query_points: 查询点（可选）

        Returns:
            (images, query_points): 标准化后的输入
        """
        # 处理图像：添加batch维度（如果缺失）
        if len(images.shape) == 4:
            images = images.unsqueeze(0)  # [S,3,H,W] -> [1,S,3,H,W]
        # 处理查询点：添加batch维度（如果缺失）
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)  # [N,2] -> [1,N,2]
        return images, query_points

    @abstractmethod
    def _aggregate_features(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        view_graphs: Optional[torch.Tensor] = None,
        causal_graphs: Optional[Union[torch.Tensor, List[np.ndarray]]] = None,
        ordered_video: Optional[torch.Tensor] = None,
        is_cp_sliced: bool = False,
    ) -> tuple:
        """
        【核心抽象方法】特征聚合 - GCT模型的核心处理流程

        这是整个模型最关键的方法，负责从图像序列提取和聚合特征。
        子类（GCTStream）会调用 aggregator 来实现具体的聚合逻辑。

        ╔════════════════════════════════════════════════════════════════════╗
        ║                 _aggregate_features 完整处理流程                    ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【输入】                                                          ║
        ║  images: [B, S, 3, H, W]                                           ║
        ║  - B: batch大小（通常为1）                                         ║
        ║  - S: 序列长度（帧数）                                             ║
        ║  - H, W: 图像尺寸（默认518x378）                                  ║
        ║                                                                    ║
        ║  【处理步骤】                                                      ║
        ║                                                                    ║
        ║  步骤1: _embed_images() - 图像嵌入                                ║
        ║  ┌─────────────────────────────────────────────────────────────┐  ║
        ║  │  输入: images [B, S, 3, H, W]                                │  ║
        ║  │                                                              │  ║
        ║  │  1.1 图像归一化:                                              │  ║
        ║  │      images = (images - ResNet_mean) / ResNet_std           │  ║
        ║  │      - ResNet_mean = [0.485, 0.456, 0.406]                  │  ║
        ║  │      - ResNet_std = [0.229, 0.224, 0.225]                   │  ║
        ║  │                                                              │  ║
        ║  │  1.2 Reshape: [B, S, 3, H, W] -> [B*S, 3, H, W]             │  ║
        ║  │      将所有帧合并处理                                         │  ║
        ║  │                                                              │  ║
        ║  │  1.3 DINOv2 Patch Embedding:                                 │  ║
        ║  │      使用 DINOv2 ViT-L/14 提取patch特征                      │  ║
        ║  │      - patch_size = 14                                       │  ║
        ║  │      - 每个patch变成1024维向量                               │  ║
        ║  │      - 输出: [B*S, P_patch, C]                               │  ║
        ║  │        P_patch = (H/14) * (W/14) ≈ 37*27 = 999个patch       │  ║
        ║  │                                                              │  ║
        ║  │  1.4 添加特殊Tokens:                                          │  ║
        ║  │      special_tokens = _prepare_special_tokens(B, S, C)      │  ║
        ║  │      - camera_token [B*S, 1, C]                             │  ║
        ║  │      - register_tokens [B*S, 4, C]                          │  ║
        ║  │      - scale_token [B*S, 1, C]                              │  ║
        ║  │                                                              │  ║
        ║  │  1.5 拼接所有tokens:                                          │  ║
        ║  │      tokens = cat([special_tokens, patch_tokens])           │  ║
        ║  │      - 输出: [B*S, P, C]                                     │  ║
        ║  │        P = 1 + 4 + 1 + P_patch = 6 + 999 ≈ 1005             │  ║
        ║  └─────────────────────────────────────────────────────────────┘  ║
        ║                                                                    ║
        ║  步骤2: _get_positions() - 位置编码                               ║
        ║  ┌─────────────────────────────────────────────────────────────┐  ║
        ║  │  生成2D旋转位置编码 (RoPE):                                   │  ║
        ║  │  - 为每个patch生成位置信息                                    │  ║
        ║  │  - 输出: pos [B*S, P, 2]                                      │  ║
        ║  │  - 用于注意力计算中的相对位置信息                             │  ║
        ║  └─────────────────────────────────────────────────────────────┘  ║
        ║                                                                    ║
        ║  步骤3: 交替执行 frame_blocks 和 global_blocks                   ║
        ║  ┌─────────────────────────────────────────────────────────────┐  ║
        ║  │                                                              │  ║
        ║  │  aa_order = ["frame", "global"]                             │  ║
        ║  │  aa_block_num = depth / aa_block_size = 24 / 1 = 24         │  ║
        ║  │                                                              │  ║
        ║  │  for block_group_idx in range(24):                          │  ║
        ║  │                                                              │  ║
        ║  │      ┌───────────────────────────────────────────────────┐  │  ║
        ║  │      │ frame_blocks[frame_idx]: 帧内自注意力              │  │  ║
        ║  │      │                                                   │  │  ║
        ║  │      │  特点:                                             │  │  ║
        ║  │      │  - 每帧独立处理，帧间不通信                        │  │  ║
        ║  │      │  - tokens: [B*S, P, C] -> [B*S, P, C]             │  │  ║
        ║  │      │  - 使用 RoPE 位置编码                              │  │  ║
        ║  │      │                                                   │  │  ║
        ║  │      │  计算过程:                                         │  │  ║
        ║  │      │  Q = tokens @ W_q                                  │  │  ║
        ║  │      │  K = tokens @ W_k                                  │  │  ║
        ║  │      │  V = tokens @ W_v                                  │  │  ║
        ║  │      │  Attention(Q, K, V) with RoPE                     │  │  ║
        ║  │      │  -> FFN -> LayerNorm -> 输出                       │  │  ║
        ║  │      └───────────────────────────────────────────────────┘  │  ║
        ║  │                                                              │  ║
        ║  │      ┌───────────────────────────────────────────────────┐  │  ║
        ║  │      │ global_blocks[global_idx]: 跨帧因果注意力         │  │  ║
        ║  │      │                                                   │  │  ║
        ║  │      │  特点:                                             │  │  ║
        ║  │      │  - 使用 FlashInferBlock 或 SDPABlock              │  │  ║
        ║  │      │  - 带 KV cache 实现高效流式推理                    │  │  ║
        ║  │      │  - 因果注意力: 帧t只能看到帧0~t                    │  │  ║
        ║  │      │                                                   │  │  ║
        ║  │      │  tokens reshape: [B*S, P, C] -> [B, S*P, C]       │  │  ║
        ║  │      │                                                   │  │  ║
        ║  │      │  KV Cache机制:                                     │  │  ║
        ║  │      │  ┌───────────────────────────────────────────────┐│  │  ║
        ║  │      │  │ cache: {k_0:[B,1,S_cache,P,C], v_0:...}       ││  │  ║
        ║  │      │  │                                               ││  │  ║
        ║  │      │  │ 当前帧的Q: [B, 1, P, C]                        ││  │  ║
        ║  │      │  │ 缓存的K/V: [B, 1, S_cache*P, C]               ││  │  ║
        ║  │      │  │                                               ││  │  ║
        ║  │      │  │ Attention(Q_cur, [K_cache + K_cur],          ││  │  ║
        ║  │      │  │           [V_cache + V_cur])                  ││  │  ║
        ║  │      │  │                                               ││  │  ║
        ║  │      │  │ 如果是关键帧:                                  ││  │  ║
        ║  │      │  │   将K_cur, V_cur存入cache                     ││  │  ║
        ║  │      │  │ 如果超出sliding_window:                       ││  │  ║
        ║  │      │  │   驱逐最旧帧的KV（保留scale frames）          ││  │  ║
        ║  │      │  └───────────────────────────────────────────────┘│  │  ║
        ║  │      │                                                   │  │  ║
        ║  │      │  Scale Token逻辑:                                 │  │  ║
        ║  │      │  - 前 num_frame_for_scale 帧使用双向注意力        │  │  ║
        ║  │      │  - scale_token 激活使得这些帧可以互相看到         │  │  ║
        ║  │      │  - 目的: 建立全局坐标系的尺度基准                 │  │  ║
        ║  │      └───────────────────────────────────────────────────┘  │  ║
        ║  │                                                              │  ║
        ║  │      收集输出:                                                │  ║
        ║  │      if block_group_idx in selected_idx:                   │  ║
        ║  │          concat = cat([frame_out, global_out], dim=-1)     │  ║
        ║  │          output_list.append(concat)  # [B, S, P, 2C]       │  ║
        ║  │                                                              │  ║
        ║  └─────────────────────────────────────────────────────────────┘  ║
        ║                                                                    ║
        ║  【输出】                                                          ║
        ║  aggregated_tokens_list: List of [B, S, P, 2C]                   ║
        ║  - selected_idx=[4, 11, 17, 23] 选择4个block的输出               ║
        ║  - 2C = frame特征(C) + global特征(C) 拼接                        ║
        ║                                                                    ║
        ║  patch_start_idx: int = 6                                         ║
        ║  - patch tokens的起始索引（跳过special tokens）                  ║
        ║  - 用于 depth_head 和 point_head 提取patch特征                   ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        【子类实现】

        GCTStream._aggregate_features():
        ```python
        def _aggregate_features(self, images, ...):
            # 直接调用 aggregator
            aggregated_tokens_list, patch_start_idx = self.aggregator(
                images,
                selected_idx=[4, 11, 17, 23],  # 选择这4个block的输出
                num_frame_for_scale=num_frame_for_scale,
                sliding_window_size=sliding_window_size,
                num_frame_per_block=num_frame_per_block,
            )
            return aggregated_tokens_list, patch_start_idx
        ```

        Args:
            images: 输入图像 [B, S, 3, H, W]，值域[0, 1]
                - B: batch大小
                - S: 序列长度（帧数）
                - H, W: 图像尺寸（默认518x378）
            num_frame_for_scale: 尺度估计帧数
                - 前 N 帧一起处理，使用双向注意力
                - 用于建立全局坐标系
                - 默认值: self.num_frame_for_scale (通常为8)
            sliding_window_size: 滑动窗口大小（blocks）
                - KV cache的最大帧数
                - 超过时自动驱逐最旧帧
                - -1 表示无限制（全因果）
            num_frame_per_block: 每个block处理的帧数
                - scale frames时: num_frame_per_block = num_frame_for_scale
                - 逐帧处理时: num_frame_per_block = 1
            view_graphs: 视图图（可选，用于非因果模式）
            causal_graphs: 因果图（可选，自定义注意力掩码）
            ordered_video: 有序视频标记（可选）
            is_cp_sliced: 是否被CP切片（已弃用）

        Returns:
            tuple: (aggregated_tokens_list, patch_start_idx)
                - aggregated_tokens_list: 多尺度特征列表
                  - 每个元素: [B, S, P, 2C]
                  - 长度取决于 selected_idx (通常4个)
                - patch_start_idx: patch tokens起始索引
                  - 用于预测头提取patch特征
        """
        pass

    def _predict_camera(
        self,
        aggregated_tokens_list: list,
        mask: Optional[torch.Tensor] = None,
        causal_inference: bool = False,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        【相机位姿预测】使用 CameraCausalHead 预测相机位姿

        CameraCausalHead 处理流程:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                   │
        │  输入: aggregated_tokens_list [B, S, P, 2C]                      │
        │                                                                   │
        │  步骤1: 提取 camera token                                        │
        │  │   camera_token = tokens[:, :, 0, :]  # 第一个token           │
        │  │   shape: [B, S, C]                                            │
        │                                                                   │
        │  步骤2: 迭代优化位姿预测 (4次迭代)                               │
        │  │   for iteration in range(4):                                 │
        │  │       # 初始化位姿编码                                        │
        │  │       pose_enc = empty_pose_tokens if iteration == 0         │
        │  │       else pose_enc_prev                                     │
        │  │                                                               │
        │  │       # CameraBlock: 因果注意力 + KV cache                   │
        │  │       # camera_token 与历史 camera_tokens 计算注意力         │
        │  │       pose_enc_new = CameraBlock(                            │
        │  │           camera_token, pose_enc,                            │
        │  │           kv_cache=camera_kv_cache                           │
        │  │       )                                                       │
        │  │                                                               │
        │  │       pose_enc_prev = pose_enc_new                           │
        │                                                                   │
        │  输出: pose_enc [B, S, 9]                                        │
        │  │   位姿编码结构:                                               │
        │  │   [:3]   - 相机中心位置 (x, y, z)                            │
        │  │   [3:7]  - 四元数旋转 (qw, qx, qy, qz)                       │
        │  │   [7:9]  - 焦距和偏移参数                                    │
        │                                                                   │
        │  位姿解码:                                                        │
        │  │   使用 pose_encoding_to_extri_intri() 转换为:               │
        │  │   - extrinsics: [B, S, 3, 4] 外参矩阵                       │
        │  │   - intrinsics: [B, S, 3, 3] 内参矩阵                       │
        │                                                                   │
        └──────────────────────────────────────────────────────────────────┘

        Args:
            aggregated_tokens_list: 聚合后的特征列表
                - 每个元素: [B, S, P, 2C]
                - 包含 frame 和 global 特征的拼接
            mask: 掩码（可选，用于有序视频处理）
            causal_inference: 是否使用因果推理模式
                - True: 使用 KV cache 的流式推理
            num_frame_for_scale: 尺度估计帧数
            sliding_window_size: 滑动窗口大小
                - 如果 enable_camera_sliding_window=True 则使用
            num_frame_per_block: 每个block处理的帧数
            gather_outputs: 是否收集输出

        Returns:
            Dict: {"pose_enc": pose_enc, "pose_enc_list": pose_enc_list}
                - pose_enc: 最终位姿预测 [B, S, 9]
                - pose_enc_list: 各迭代步骤的位姿列表
        """
        if self.camera_head is None:
            return {}

        # 转换为FP32以避免数值精度问题
        # 位姿预测对精度敏感，使用FP32可以提高预测质量
        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]

        # 确定相机头是否使用滑动窗口
        # 如果 enable_camera_sliding_window=True 且指定了滑动窗口大小，
        # 则相机头也使用滑动窗口注意力
        camera_sliding_window = sliding_window_size if self.enable_camera_sliding_window else -1

        # 禁用autocast，确保使用FP32精度
        # 这是为了避免在低精度下位姿预测出现数值问题
        with torch.amp.autocast('cuda', enabled=False):
            pose_enc_list = self.camera_head(
                aggregated_tokens_list_fp32,
                mask=mask,
                causal_inference=causal_inference,
                num_frame_for_scale=num_frame_for_scale if num_frame_for_scale is not None else -1,
                sliding_window_size=camera_sliding_window,
                num_frame_per_block=num_frame_per_block,
            )

        # 返回位姿编码
        # pose_enc_list[-1] 是最后一次迭代的结果（最精确）
        return {
            "pose_enc": pose_enc_list[-1],  # 最终位姿预测 [B, S, 9]
            "pose_enc_list": pose_enc_list,  # 各迭代的中间结果
        }

    def _predict_depth(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        【深度图预测】使用 DPTHead 预测深度图

        DPTHead 处理流程:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                   │
        │  输入:                                                            │
        │  - aggregated_tokens_list: 多尺度特征 [B, S, P, 2C]              │
        │  - images: 原始图像 [B, S, 3, H, W]                              │
        │  - patch_start_idx: patch起始索引                                │
        │                                                                   │
        │  Dense Prediction Transformer (DPT) 结构:                       │
        │                                                                   │
        │  步骤1: 特征提取                                                  │
        │  │   提取 patch tokens (跳过 special tokens)                    │
        │  │   patch_tokens = tokens[:, :, patch_start_idx:, :]           │
        │  │   shape: [B, S, P_patch, 2C]                                 │
        │                                                                   │
        │  步骤2: 多尺度特征重组                                            │
        │  │   将不同block的特征组合成多尺度表示                           │
        │  │   selected_idx=[4, 11, 17, 23] 提供不同尺度                  │
        │                                                                   │
        │  步骤3: 解码器处理                                                │
        │  │   逐步上采样特征到原始分辨率                                   │
        │  │   最终输出: [B, S, H, W, 2]                                  │
        │                                                                   │
        │  步骤4: 激活函数                                                  │
        │  │   depth = exp(pred[:, :, :, :, 0])  # 确保正值               │
        │  │   conf = exp(pred[:, :, :, :, 1]) + 1  # 置信度              │
        │                                                                   │
        │  输出:                                                            │
        │  - depth: [B, S, H, W, 1] 深度图                                 │
        │  - depth_conf: [B, S, H, W] 置信度                               │
        │                                                                   │
        └──────────────────────────────────────────────────────────────────┘

        Args:
            aggregated_tokens_list: 聚合特征列表
            images: 原始图像（用于确定分辨率）
            patch_start_idx: patch tokens的起始索引
            gather_outputs: 是否收集输出

        Returns:
            Dict: {"depth": depth, "depth_conf": depth_conf}
                - depth: 深度图 [B, S, H, W, 1]
                - depth_conf: 置信度 [B, S, H, W]
        """
        if self.depth_head is None:
            return {}

        # 转换为FP32精度
        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]
        images_fp32 = images.float()

        # 禁用autocast，使用FP32
        with torch.amp.autocast('cuda', enabled=False):
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list_fp32,
                images=images_fp32,
                patch_start_idx=patch_start_idx
            )

        return {"depth": depth, "depth_conf": depth_conf}

    def _predict_points(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        【世界坐标3D点云预测】使用 DPTHead 预测世界坐标系下的3D点

        处理流程与 depth_head 类似，但:
        - output_dim = 4 (x, y, z + confidence)
        - activation = "inv_log" (逆向log变换)

        输出:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                   │
        │  world_points: [B, S, H, W, 3]                                   │
        │  │   世界坐标系下的3D点坐标                                      │
        │  │   由 scale frames 建立的全局坐标系                           │
        │                                                                   │
        │  world_points_conf: [B, S, H, W]                                 │
        │  │   每个点的置信度                                              │
        │                                                                   │
        │  坐标变换:                                                        │
        │  │   可以结合位姿和深度得到相机坐标系点                          │
        │  │   world_pts = c2w @ (depth * K_inv @ pixel_coords)          │
        │                                                                   │
        └──────────────────────────────────────────────────────────────────┘

        Args:
            aggregated_tokens_list: 聚合特征列表
            images: 原始图像
            patch_start_idx: patch起始索引
            gather_outputs: 是否收集输出

        Returns:
            Dict: {"world_points": pts3d, "world_points_conf": pts3d_conf}
        """
        if self.point_head is None:
            return {}

        # 转换为FP32精度
        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]
        images_fp32 = images.float()

        with torch.amp.autocast('cuda', enabled=False):
            pts3d, pts3d_conf = self.point_head(
                aggregated_tokens_list_fp32,
                images=images_fp32,
                patch_start_idx=patch_start_idx
            )

        return {"world_points": pts3d, "world_points_conf": pts3d_conf}

    def _predict_local_points(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
        gather_outputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        【相机坐标3D点云预测】预测相机坐标系下的3D点

        与 world_points 类似，但输出在相机坐标系:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                   │
        │  cam_points: [B, S, H, W, 3]                                     │
        │  │   相机坐标系下的3D点                                          │
        │  │   Z轴指向前方，X轴向右，Y轴向下                              │
        │                                                                   │
        │  cam_points_conf: [B, S, H, W]                                   │
        │                                                                   │
        │  用途:                                                            │
        │  - 验证深度预测的一致性                                          │
        │  - 局部重建任务                                                  │
        │                                                                   │
        └──────────────────────────────────────────────────────────────────┘

        Args:
            aggregated_tokens_list: 聚合特征列表
            images: 原始图像
            patch_start_idx: patch起始索引
            gather_outputs: 是否收集输出

        Returns:
            Dict: {"cam_points": pts3d, "cam_points_conf": pts3d_conf}
        """
        if self.local_point_head is None:
            return {}

        aggregated_tokens_list_fp32 = [t.float() for t in aggregated_tokens_list]
        images_fp32 = images.float()

        with torch.amp.autocast('cuda', enabled=False):
            pts3d, pts3d_conf = self.local_point_head(
                aggregated_tokens_list_fp32,
                images=images_fp32,
                patch_start_idx=patch_start_idx
            )

        return {"cam_points": pts3d, "cam_points_conf": pts3d_conf}

    def _unproject_depth_to_world(
        self,
        depth: torch.Tensor,
        pose_enc: torch.Tensor,
    ) -> torch.Tensor:
        """
        【深度反投影】将深度图转换为世界坐标系下的3D点

        计算流程:
        ┌──────────────────────────────────────────────────────────────────┐
        │                                                                   │
        │  输入:                                                            │
        │  - depth: [B, S, H, W, 1] 深度图                                 │
        │  - pose_enc: [B, S, 9] 位姿编码                                  │
        │                                                                   │
        │  步骤1: 解码位姿                                                  │
        │  │   extrinsics, intrinsics = pose_encoding_to_extri_intri()   │
        │  │   ext: [B, S, 3, 4] 外参矩阵                                 │
        │  │   int: [B, S, 3, 3] 内参矩阵                                 │
        │                                                                   │
        │  步骤2: 计算 camera-to-world 变换                                │
        │  │   ext_4x4: [B*S, 4, 4] 完整外参矩阵                          │
        │  │   c2w = closed_form_inverse_se3(ext_4x4)                     │
        │  │   c2w: [B, S, 4, 4] 相机到世界变换                           │
        │                                                                   │
        │  步骤3: 构建像素坐标网格                                          │
        │  │   对于每个像素 (u, v):                                        │
        │  │   pixel_coords = [u, v, 1]                                   │
        │                                                                   │
        │  步骤4: 反投影到相机坐标系                                        │
        │  │   camera_coords = K_inv @ pixel_coords                       │
        │  │   camera_points = camera_coords * depth                      │
        │                                                                   │
        │  步骤5: 变换到世界坐标系                                          │
        │  │   world_points = c2w @ camera_points_h                       │
        │                                                                   │
        │  输出:                                                            │
        │  world_points: [B, S, H, W, 3]                                   │
        │                                                                   │
        └──────────────────────────────────────────────────────────────────┘

        Args:
            depth: 深度图 [B, S, H, W, 1]
            pose_enc: 位姿编码 [B, S, 9]

        Returns:
            world_points: 世界坐标系3D点 [B, S, H, W, 3]
        """
        B, S, H, W, _ = depth.shape
        device = depth.device
        dtype = depth.dtype

        # 图像尺寸
        image_size_hw = (H, W)

        # 解码位姿编码为外参和内参矩阵
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=image_size_hw, build_intrinsics=True
        )

        # 构建完整的4x4外参矩阵
        extrinsics_flat = extrinsics.view(B * S, 3, 4)
        extrinsics_4x4 = torch.zeros(B * S, 4, 4, device=device, dtype=dtype)
        extrinsics_4x4[:, :3, :] = extrinsics_flat
        extrinsics_4x4[:, 3, 3] = 1.0

        # 计算 camera-to-world 变换矩阵
        # c2w = inv(w2c)，其中 w2c = extrinsics_4x4
        c2w = closed_form_inverse_se3(extrinsics_4x4).view(B, S, 4, 4)

        # 构建像素坐标网格
        # 对于每个像素，创建齐次坐标 [u, v, 1]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing='ij'
        )
        pixel_coords = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)

        # 反投影到相机坐标系
        # camera_coords = K_inv @ pixel_coords，然后乘以深度
        intrinsics_inv = torch.inverse(intrinsics)
        camera_coords = torch.einsum('bsij,hwj->bshwi', intrinsics_inv, pixel_coords)
        camera_points = camera_coords * depth

        # 变换到世界坐标系
        # world_points = c2w @ camera_points_h
        ones = torch.ones_like(camera_points[..., :1])
        camera_points_h = torch.cat([camera_points, ones], dim=-1)  # 齐次坐标
        world_points_h = torch.einsum('bsij,bshwj->bshwi', c2w, camera_points_h)

        return world_points_h[..., :3]  # 返回非齐次坐标

    def forward(
        self,
        images: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        mask: Optional[torch.Tensor] = None,
        causal_inference: bool = False,
        ordered_video: Optional[torch.Tensor] = None,
        gather_outputs: bool = True,
        point_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        【前向传播】GCT模型的核心推理入口

        本方法串联了整个模型的处理流程:
        1. 特征聚合 (_aggregate_features)
        2. 相机位姿预测 (_predict_camera)
        3. 深度预测 (_predict_depth)
        4. 3D点云预测 (_predict_points)
        5. 局部点云预测 (_predict_local_points)

        ╔════════════════════════════════════════════════════════════════════╗
        ║                      forward() 执行流程图                          ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  images [B,S,3,H,W]  或  [S,3,H,W]                                 ║
        ║       │                                                             ║
        ║       │  (如果是4维，添加batch维度)                                 ║
        ║       ↓                                                             ║
        ║  _normalize_input() ───────────────────────────────────────────    ║
        ║       │                                                             ║
        ║       │  images: [1, S, 3, H, W]                                   ║
        ║       ↓                                                             ║
        ║  ┌─────────────────────────────────────────────────────────────┐   ║
        ║  │ _aggregate_features()  【核心特征聚合】                     │   ║
        ║  │                                                             │   ║
        ║  │  self.aggregator(images, ...)                               │   ║
        ║  │                                                             │   ║
        ║  │  内部流程:                                                   │   ║
        ║  │  1. 图像归一化 + DINOv2 Patch Embedding                     │   ║
        ║  │     images [B,S,3,H,W] -> patch_tokens [B*S,P_patch,C]     │   ║
        ║  │                                                             │   ║
        ║  │  2. 添加特殊tokens                                           │   ║
        ║  │     [camera][register][scale] + patch_tokens                │   ║
        ║  │     -> tokens [B*S, P, C]                                    │   ║
        ║  │                                                             │   ║
        ║  │  3. frame_blocks: 帧内自注意力                              │   ║
        ║  │     每帧独立处理                                              │   ║
        ║  │                                                             │   ║
        ║  │  4. global_blocks: 跨帧因果注意力                           │   ║
        ║  │     - KV cache 实现高效流式推理                              │   ║
        ║  │     - scale token 控制前N帧双向注意力                       │   ║
        ║  │                                                             │   ║
        ║  │  输出:                                                       │   ║
        ║  │  - aggregated_tokens_list: List[[B,S,P,2C]]                │   ║
        ║  │    (selected_idx=[4,11,17,23] 选择4个block输出)             │   ║
        ║  │  - patch_start_idx: 6                                        │   ║
        ║  └─────────────────────────────────────────────────────────────┘   ║
        ║       │                                                             ║
        ║       │  aggregated_tokens_list, patch_start_idx                  ║
        ║       ↓                                                             ║
        ║  ┌─────────────────────────────────────────────────────────────┐   ║
        ║  │ _predict_camera()  【相机位姿预测】                         │   ║
        ║  │                                                             │   ║
        ║  │  CameraCausalHead:                                          │   ║
        ║  │  - 提取 camera_token                                        │   ║
        ║  │  - 4次迭代优化位姿                                          │   ║
        ║  │  - KV cache 因果注意力                                      │   ║
        ║  │                                                             │   ║
        ║  │  输出:                                                       │   ║
        ║  │  - pose_enc: [B, S, 9]                                      │   ║
        ║  │    位姿编码 = center(3) + quaternion(4) + focal/offset(2)  │   ║
        ║  │  - pose_enc_list: 各迭代步骤的位姿列表                      │   ║
        ║  └─────────────────────────────────────────────────────────────┘   ║
        ║       │                                                             ║
        ║       │  pose_enc [B, S, 9]                                        ║
        ║       ↓                                                             ║
        ║  ┌─────────────────────────────────────────────────────────────┐   ║
        ║  │ _predict_depth()  【深度预测】                              │   ║
        ║  │                                                             │   ║
        ║  │  DPTHead:                                                    │   ║
        ║  │  - Dense Prediction Transformer                             │   ║
        ║  │  - 多尺度特征解码                                           │   ║
        ║  │  - activation="exp" 确保深度为正                            │   ║
        ║  │                                                             │   ║
        ║  │  输出:                                                       │   ║
        ║  │  - depth: [B, S, H, W, 1]                                   │   ║
        ║  │  - depth_conf: [B, S, H, W]                                 │   ║
        ║  └─────────────────────────────────────────────────────────────┘   ║
        ║       │                                                             ║
        ║       │  depth, depth_conf                                        ║
        ║       ↓                                                             ║
        ║  ┌─────────────────────────────────────────────────────────────┐   ║
        ║  │ _predict_points()  【3D点云预测】                           │   ║
        ║  │                                                             │   ║
        ║  │  DPTHead (output_dim=4):                                    │   ║
        ║  │  - 输出世界坐标系下的3D点坐标                               │   ║
        ║  │  - activation="inv_log"                                     │   ║
        ║  │                                                             │   ║
        ║  │  输出:                                                       │   ║
        ║  │  - world_points: [B, S, H, W, 3]                            │   ║
        ║  │  - world_points_conf: [B, S, H, W]                          │   ║
        ║  └─────────────────────────────────────────────────────────────┘   ║
        ║       │                                                             ║
        ║       │  world_points, world_points_conf                         ║
        ║       ↓                                                             ║
        ║  ┌─────────────────────────────────────────────────────────────┐   ║
        ║  │ _predict_local_points()  【局部点云预测】                   │   ║
        ║  │                                                             │   ║
        ║  │  输出相机坐标系下的3D点                                     │   ║
        ║  │  - cam_points: [B, S, H, W, 3]                              │   ║
        ║  │  - cam_points_conf: [B, S, H, W]                            │   ║
        ║  └─────────────────────────────────────────────────────────────┘   ║
        ║       │                                                             ║
        ║       │  所有预测结果                                             ║
        ║       ↓                                                             ║
        ║  predictions = {                                                    ║
        ║      "pose_enc": pose_enc,                                          ║
        ║      "depth": depth,                                                ║
        ║      "depth_conf": depth_conf,                                      ║
        ║      "world_points": world_points,                                  ║
        ║      "world_points_conf": world_points_conf,                        ║
        ║      "cam_points": cam_points,                                      ║
        ║      "cam_points_conf": cam_points_conf,                            ║
        ║      "images": images  (推理时保存)                                 ║
        ║  }                                                                  ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝

        【推理模式说明】

        1. Scale Frames 处理 (num_frame_for_scale):
           - 前 N 帧一起处理，使用双向注意力
           - 目的: 建立全局坐标系的尺度基准
           - 类似 SLAM 的初始化阶段

        2. Causal Inference (causal_inference=True):
           - 使用 KV cache 实现流式推理
           - 每帧只看到历史帧（因果注意力）
           - 推理速度约 20 FPS

        3. Sliding Window (sliding_window_size):
           - 限制 KV cache 的帧数
           - 超过窗口时自动驱逐最旧帧
           - 保留 scale frames 和 special tokens

        Args:
            images: 输入图像 [S, 3, H, W] 或 [B, S, 3, H, W]
                - 值域: [0, 1]（浮点数）
                - H, W: 默认 518x378（patch_size=14）
                - 如果是4维，会自动添加batch维度

            query_points: 查询点（可选）
                - 用于跟踪任务
                - shape: [N, 2] 或 [B, N, 2]

            num_frame_for_scale: 尺度估计帧数
                - 前 N 帧使用双向注意力
                - 用于建立全局坐标系
                - 默认: self.num_frame_for_scale (通常为8)

            sliding_window_size: 滑动窗口大小
                - KV cache 的最大帧数
                - -1 表示无限制

            num_frame_per_block: 每个block处理的帧数
                - scale frames时: = num_frame_for_scale
                - 逐帧处理时: = 1

            mask: 掩码（可选）
                - 用于有序视频处理

            causal_inference: 是否使用因果推理
                - True: 启用 KV cache 流式推理

            ordered_video: 有序视频标记（可选）

            gather_outputs: 是否收集输出

            point_masks: 点掩码（可选）

            **kwargs: 其他参数

        Returns:
            predictions 字典:
                - pose_enc: 相机位姿编码 [B, S, 9]
                           可通过 pose_encoding_to_extri_intri() 转换为外参/内参
                - depth: 深度图 [B, S, H, W, 1]
                - depth_conf: 深度置信度 [B, S, H, W]
                - world_points: 3D世界坐标点 [B, S, H, W, 3]
                - world_points_conf: 点置信度 [B, S, H, W]
                - cam_points: 相机坐标3D点 [B, S, H, W, 3]
                - cam_points_conf: 置信度 [B, S, H, W]
                - images: 原始图像 [B, S, 3, H, W]（推理时保存）

        【位姿编码解码示例】

        ```python
        # 将位姿编码转换为外参和内参矩阵
        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            pose_enc, image_size_hw=(H, W)
        )
        # extrinsics: [B, S, 3, 4] 外参矩阵
        # intrinsics: [B, S, 3, 3] 内参矩阵

        # 构建 camera-to-world 变换
        ext_4x4 = torch.zeros(B*S, 4, 4)
        ext_4x4[:, :3, :] = extrinsics.view(B*S, 3, 4)
        ext_4x4[:, 3, 3] = 1.0
        c2w = closed_form_inverse_se3(ext_4x4)  # [B*S, 4, 4]
        ```
        """
        # 【步骤1】输入标准化
        # 确保输入形状符合模型要求: [B, S, 3, H, W]
        images, query_points = self._normalize_input(images, query_points)

        # 【步骤2】特征聚合 - 核心处理
        # 调用 aggregator 进行图像特征提取和跨帧注意力
        aggregated_tokens_list, patch_start_idx = self._aggregate_features(
            images,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )

        # 【步骤3】初始化预测结果字典
        predictions = {}

        # 【步骤4】相机位姿预测
        # 使用 CameraCausalHead 预测相机位姿
        predictions.update(self._predict_camera(
            aggregated_tokens_list,
            mask=ordered_video,
            causal_inference=causal_inference,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
            gather_outputs=gather_outputs,
        ))

        # 【步骤5】深度预测
        # 使用 DPTHead 预测深度图
        predictions.update(self._predict_depth(
            aggregated_tokens_list, images, patch_start_idx,
            gather_outputs=gather_outputs,
        ))

        # 【步骤6】世界坐标3D点云预测
        # 使用 DPTHead 预测世界坐标系下的3D点
        predictions.update(self._predict_points(
            aggregated_tokens_list, images, patch_start_idx,
            gather_outputs=gather_outputs,
        ))

        # 【步骤7】相机坐标3D点云预测
        # 使用 DPTHead 预测相机坐标系下的3D点
        predictions.update(self._predict_local_points(
            aggregated_tokens_list, images, patch_start_idx,
            gather_outputs=gather_outputs,
        ))

        # 【步骤8】推理时保存原始图像
        # 用于可视化时将点云颜色映射回原图
        if not self.training:
            predictions["images"] = images

        return predictions
