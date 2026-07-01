"""
GCTStream - Streaming GCT with KV cache for online inference.

Provides streaming inference functionality:
- Temporal causal attention with KV cache
- Sliding window support
- Efficient frame-by-frame processing
- 3D RoPE support for temporal consistency
"""

import logging
import time
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from tqdm.auto import tqdm

from lingbot_map.heads.camera_head import CameraCausalHead
from lingbot_map.models.gct_base import GCTBase
from lingbot_map.aggregator.stream import AggregatorStream

logger = logging.getLogger(__name__)


class GCTStream(GCTBase):
    """
    Streaming GCT model with KV cache for efficient online inference.

    Features:
    - AggregatorStream with KV cache support (FlashInfer backend)
    - CameraCausalHead for pose refinement
    - Sliding window attention for memory efficiency
    - Frame-by-frame streaming inference
    """

    def __init__(
        self,
        # Architecture parameters
        img_size: int = 518,
        patch_size: int = 14,
        embed_dim: int = 1024,
        patch_embed: str = 'dinov2_vitl14_reg',
        pretrained_path: str = '',
        aggregator_depth: int = 24,
        selected_idx: Optional[List[int]] = None,
        disable_global_rope: bool = False,
        # Head configuration
        enable_camera: bool = True,
        enable_point: bool = True,
        enable_local_point: bool = False,
        enable_depth: bool = True,
        enable_track: bool = False,
        enable_camera_sliding_window: bool = False,
        # Normalization
        enable_normalize: bool = False,
        # Prediction normalization
        pred_normalization: bool = False,
        # Stream-specific parameters
        sliding_window_size: int = -1,
        num_frame_for_scale: int = 1,
        num_random_frames: int = 0,
        attend_to_special_tokens: bool = False,
        attend_to_scale_frames: bool = False,
        enable_stream_inference: bool = True,  # Default to True for streaming
        enable_3d_rope: bool = False,
        max_frame_num: int = 1024,
        # Camera head 3D RoPE (separate from aggregator 3D RoPE)
        enable_camera_3d_rope: bool = False,
        camera_rope_theta: float = 10000.0,
        camera_trunk_depth: int = 4,
        camera_num_heads: Optional[int] = None,
        # Streaming context token configuration
        use_anchor_token: bool = False,
        use_scale_token: bool = True,
        # KV cache parameters
        kv_cache_sliding_window: int = 64,
        kv_cache_scale_frames: int = 8,
        kv_cache_cross_frame_special: bool = True,
        kv_cache_include_scale_frames: bool = True,
        kv_cache_camera_only: bool = False,
        # Backend selection
        use_sdpa: bool = False,  # If True, use SDPA (no flashinfer needed); default: FlashInfer
        # Gradient checkpointing
        use_gradient_checkpoint: bool = True,
    ):
        """
        Initialize GCTStream.

        Args:
            img_size: Input image size
            patch_size: Patch size for embedding
            embed_dim: Embedding dimension
            patch_embed: Patch embedding type ("dinov2_vitl14_reg", "conv", etc.)
            pretrained_path: Path to pretrained DINOv2 weights
            disable_global_rope: Disable RoPE in global attention
            enable_camera/point/depth/track: Enable prediction heads
            enable_normalize: Enable normalization
            sliding_window_size: Sliding window size in blocks (-1 for full causal)
            num_frame_for_scale: Number of scale estimation frames
            num_random_frames: Number of random frames for long-range dependencies
            attend_to_special_tokens: Enable cross-frame special token attention
            attend_to_scale_frames: Whether to attend to scale frames
            enable_stream_inference: Enable streaming inference with KV cache
            enable_3d_rope: Enable 3D RoPE for temporal consistency
            max_frame_num: Maximum number of frames for 3D RoPE
            use_anchor_token: Add a learnable anchor token for GCA anchor context
            use_scale_token: Add the legacy learnable scale token
            kv_cache_sliding_window: Sliding window size for KV cache eviction
            kv_cache_scale_frames: Number of scale frames to keep in KV cache
            kv_cache_cross_frame_special: Keep special tokens from evicted frames
            kv_cache_include_scale_frames: Include scale frames in KV cache
            kv_cache_camera_only: Only keep camera tokens from evicted frames
        """
        # Store stream-specific parameters before calling super().__init__()
        self.pretrained_path = pretrained_path
        self.aggregator_depth = aggregator_depth
        self.selected_idx = list(selected_idx) if selected_idx is not None else self._default_selected_idx(aggregator_depth)
        self._validate_selected_idx(self.selected_idx, aggregator_depth)
        self.sliding_window_size = sliding_window_size
        self.num_frame_for_scale = num_frame_for_scale
        self.num_random_frames = num_random_frames
        self.attend_to_special_tokens = attend_to_special_tokens
        self.attend_to_scale_frames = attend_to_scale_frames
        self.enable_stream_inference = enable_stream_inference
        self.enable_3d_rope = enable_3d_rope
        self.max_frame_num = max_frame_num
        self.use_anchor_token = use_anchor_token
        self.use_scale_token = use_scale_token
        # Camera head 3D RoPE settings
        self.enable_camera_3d_rope = enable_camera_3d_rope
        self.camera_rope_theta = camera_rope_theta
        self.camera_trunk_depth = camera_trunk_depth
        self.camera_num_heads = camera_num_heads
        # KV cache parameters
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only
        self.use_sdpa = use_sdpa

        # Call base class __init__ (will call _build_aggregator)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            patch_embed=patch_embed,
            disable_global_rope=disable_global_rope,
            enable_camera=enable_camera,
            enable_point=enable_point,
            enable_local_point=enable_local_point,
            enable_depth=enable_depth,
            enable_track=enable_track,
            enable_camera_sliding_window=enable_camera_sliding_window,
            enable_normalize=enable_normalize,
            pred_normalization=pred_normalization,
            enable_3d_rope=enable_3d_rope,
            use_gradient_checkpoint=use_gradient_checkpoint,
        )

    def _build_aggregator(self) -> nn.Module:
        """
        Build streaming aggregator with KV cache support (FlashInfer backend).

        Returns:
            AggregatorStream module
        """
        # Dynamically compute num_heads based on embed_dim
        # ViT-L: embed_dim=1024 -> num_heads=16, head_dim=64
        # ViT-B: embed_dim=768 -> num_heads=12, head_dim=64
        # ViT-S: embed_dim=384 -> num_heads=6, head_dim=64
        num_heads = self.embed_dim // 64

        return AggregatorStream(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            patch_embed=self.patch_embed,
            pretrained_path=self.pretrained_path,
            depth=self.aggregator_depth,
            disable_global_rope=self.disable_global_rope,
            sliding_window_size=self.sliding_window_size,
            num_frame_for_scale=self.num_frame_for_scale,
            num_random_frames=self.num_random_frames,
            attend_to_special_tokens=self.attend_to_special_tokens,
            attend_to_scale_frames=self.attend_to_scale_frames,
            enable_stream_inference=self.enable_stream_inference,
            enable_3d_rope=self.enable_3d_rope,
            max_frame_num=self.max_frame_num,
            use_anchor_token=self.use_anchor_token,
            use_scale_token=self.use_scale_token,
            # Backend: FlashInfer (default) or SDPA (fallback)
            use_flashinfer=not self.use_sdpa,
            use_sdpa=self.use_sdpa,
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
            use_gradient_checkpoint=self.use_gradient_checkpoint,
        )

    def _build_camera_head(self) -> nn.Module:
        """
        Build causal camera head for streaming inference.

        Returns:
            CameraCausalHead module or None
        """
        return CameraCausalHead(
            dim_in=2 * self.embed_dim,
            trunk_depth=self.camera_trunk_depth,
            num_heads=self.camera_num_heads if self.camera_num_heads is not None else 16,
            sliding_window_size=self.sliding_window_size,
            attend_to_scale_frames=self.attend_to_scale_frames,
            # KV cache parameters
            kv_cache_sliding_window=self.kv_cache_sliding_window,
            kv_cache_scale_frames=self.kv_cache_scale_frames,
            kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
            kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
            kv_cache_camera_only=self.kv_cache_camera_only,
            # Camera head 3D RoPE parameters
            enable_3d_rope=self.enable_camera_3d_rope,
            max_frame_num=self.max_frame_num,
            rope_theta=self.camera_rope_theta,
        )

    def _aggregate_features(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        **kwargs,
    ) -> tuple:
        """
        Run aggregator to get multi-scale features.

        Args:
            images: Input images [B, S, 3, H, W]
            num_frame_for_scale: Number of frames for scale estimation
            sliding_window_size: Override sliding window size
            num_frame_per_block: Number of frames per block

        Returns:
            (aggregated_tokens_list, patch_start_idx)
        """
        aggregated_tokens_list, patch_start_idx = self.aggregator(
            images,
            selected_idx=self.selected_idx,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=sliding_window_size,
            num_frame_per_block=num_frame_per_block,
        )
        return aggregated_tokens_list, patch_start_idx

    @staticmethod
    def _default_selected_idx(aggregator_depth: int) -> List[int]:
        """Default four feature taps for DPT-style heads."""
        if aggregator_depth < 4:
            raise ValueError("aggregator_depth must be at least 4 when using DPT heads")
        if aggregator_depth == 24:
            return [4, 11, 17, 23]
        if aggregator_depth == 12:
            return [2, 5, 8, 11]
        return [
            max(0, int(aggregator_depth * fraction) - 1)
            for fraction in (0.25, 0.50, 0.75, 1.00)
        ]

    @staticmethod
    def _validate_selected_idx(selected_idx: List[int], aggregator_depth: int) -> None:
        if len(selected_idx) < 4:
            raise ValueError("selected_idx must provide at least four feature taps for DPT heads")
        invalid = [idx for idx in selected_idx if idx < 0 or idx >= aggregator_depth]
        if invalid:
            raise ValueError(
                f"selected_idx contains out-of-range indices {invalid}; "
                f"aggregator_depth={aggregator_depth}"
            )

    def clean_kv_cache(self):
        """
        Clean KV cache in aggregator.

        Call this method when starting a new video sequence to clear
        cached key-value pairs from previous sequences.
        """
        if hasattr(self.aggregator, 'clean_kv_cache'):
            self.aggregator.clean_kv_cache()
        else:
            logger.warning("Aggregator does not support KV cache cleaning")
        if hasattr(self.camera_head, 'kv_cache'):
            self.camera_head.clean_kv_cache()
        else:
            logger.warning("Camera head does not support KV cache cleaning")

    def _set_skip_append(self, skip: bool):
        """Set _skip_append flag on all KV caches (aggregator + camera head).

        When skip=True, attention layers will attend to [cached_kv + current_kv]
        but will NOT store the current frame's KV in cache. This is used for
        non-keyframe processing in keyframe-based streaming inference.

        Args:
            skip: If True, subsequent forward passes will not append KV to cache.
        """
        if hasattr(self.aggregator, 'kv_cache') and self.aggregator.kv_cache is not None:
            self.aggregator.kv_cache["_skip_append"] = skip
        if self.camera_head is not None and hasattr(self.camera_head, 'kv_cache') and self.camera_head.kv_cache is not None:
            for cache_dict in self.camera_head.kv_cache:
                cache_dict["_skip_append"] = skip

    def get_kv_cache_info(self) -> Dict[str, Any]:
        """
        Get information about current KV cache state.

        Returns:
            Dictionary with cache statistics:
                - num_cached_blocks: Number of blocks with cached KV
                - cache_memory_mb: Approximate memory usage in MB
        """
        if not hasattr(self.aggregator, 'kv_cache') or self.aggregator.kv_cache is None:
            return {"num_cached_blocks": 0, "cache_memory_mb": 0.0}

        kv_cache = self.aggregator.kv_cache
        num_cached = sum(1 for k in kv_cache.keys() if k.startswith('k_') and not k.endswith('_special'))

        # Estimate memory usage
        total_elements = 0
        for _, v in kv_cache.items():
            if v is not None and torch.is_tensor(v):
                total_elements += v.numel()

        # Assume bfloat16 (2 bytes per element)
        cache_memory_mb = (total_elements * 2) / (1024 * 1024)

        return {
            "num_cached_blocks": num_cached,
            "cache_memory_mb": round(cache_memory_mb, 2)
        }

    @torch.no_grad()
    def inference_streaming(
        self,
        images: torch.Tensor,
        num_scale_frames: Optional[int] = None,
        keyframe_interval: int = 1,
        output_device: Optional[torch.device] = None,
        test_resolution: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        【流式推理核心函数】处理图像序列进行3D重建

        执行流程详解:

        ╔════════════════════════════════════════════════════════════════════╗
        ║                    inference_streaming 总体流程                      ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐     ║
        ║  │ 清空KV缓存   │ -> │ Phase 1    │ -> │ Phase 2             │     ║
        ║  │ clean_kv_   │    │ 处理scale  │    │逐帧处理剩余帧       │     ║
        ║  │ cache()     │    │ frames     │    │(带KV cache推理)     │     ║
        ║  └─────────────┘    └─────────────┘    └─────────────────────┘     ║
        ║                                            |                       ║
        ║                                            v                       ║
        ║                              ┌─────────────────────┐              ║
        ║                              │ 合并所有预测结果     │              ║
        ║                              │ torch.cat(all_*)    │              ║
        ║                              └─────────────────────┘              ║
        ║                                                                    ║
        ╚════════════════════════════════════════════════════════════════════╝

        【核心概念说明】

        1. Scale Frames (尺度估计帧):
           - 前 num_scale_frames 帧图像一起处理
           - 通过 scale token 实现双向注意力（帧间可互相看到）
           - 用于建立全局坐标系和尺度基准
           - 类似于 SLAM 中的初始化阶段

        2. KV Cache (键值缓存):
           - 存储历史帧的注意力 Key 和 Value
           - 新帧只需与缓存中的 KV 计算注意力，无需重新编码历史帧
           - 实现高效的在线推理（约20 FPS）
           - 支持滑动窗口策略：当帧数超过 kv_cache_sliding_window 时，
             自动驱逐最旧帧的 KV（但保留 scale frames 和特殊 token）

        3. Keyframe Mode (关键帧模式):
           - 当 keyframe_interval > 1 时启用
           - 只有每隔 keyframe_interval 帧才会存储 KV 到缓存
           - 非关键帧：参与注意力计算，但不更新缓存
           - 目的：减少长序列的内存占用

        【forward() 调用链】

        inference_streaming -> forward() -> _aggregate_features()
                                           -> _predict_camera()
                                           -> _predict_depth()
                                           -> _predict_points()

        forward() 详细流程:
        ┌────────────────────────────────────────────────────────────────┐
        │  images [B,S,3,H,W]                                             │
        │       │                                                         │
        │       v                                                         │
        │  _normalize_input() - 标准化输入形状                            │
        │       │                                                         │
        │       v                                                         │
        │  ┌─────────────────────────────────────────────────────────┐   │
        │  │ aggregator (AggregatorStream)                           │   │
        │  │                                                          │   │
        │  │  1. _embed_images()                                      │   │
        │  │     - 图像归一化 (ResNet mean/std)                        │   │
        │  │     - DINOv2 ViT-L/14 Patch Embedding                    │   │
        │  │     - 添加 special tokens (camera + register + scale)    │   │
        │  │                                                          │   │
        │  │  2. 交替执行 frame_blocks 和 global_blocks                │   │
        │  │     - frame_blocks: 帧内自注意力 (独立处理每帧)            │   │
        │  │     - global_blocks: 跨帧因果注意力 (带 KV cache)          │   │
        │  │                                                          │   │
        │  │  3. 返回 aggregated_tokens_list [B,S,P,2C]               │   │
        │  └─────────────────────────────────────────────────────────┘   │
        │       │                                                         │
        │       v                                                         │
        │  ┌─────────────────────────────────────────────────────────┐   │
        │  │ _predict_camera() (CameraCausalHead)                     │   │
        │  │                                                          │   │
        │  │  1. 提取 camera token (tokens[:,:,0])                    │   │
        │  │  2. 迭代优化位姿预测 (4次迭代)                             │   │
        │  │     - 使用 empty_pose_tokens 初始化                       │   │
        │  │     - CameraBlock 带因果注意力 + KV cache                 │   │
        │  │     - 输出 pose_enc [B,S,9] (中心+四元数+焦距/偏移)        │   │
        │  └─────────────────────────────────────────────────────────┘   │
        │       │                                                         │
        │       v                                                         │
        │  ┌─────────────────────────────────────────────────────────┐   │
        │  │ _predict_depth() (DPTHead)                               │   │
        │  │                                                          │   │
        │  │  - Dense Prediction Transformer 结构                     │   │
        │  │  - 从多尺度特征解码深度图                                  │   │
        │  │  - 输出 depth [B,S,H,W,1], depth_conf [B,S,H,W]          │   │
        │  └─────────────────────────────────────────────────────────┘   │
        │       │                                                         │
        │       v                                                         │
        │  ┌─────────────────────────────────────────────────────────┐   │
        │  │ _predict_points() (DPTHead)                              │   │
        │  │                                                          │   │
        │  │  - 类似 depth head，输出维度为4                            │   │
        │  │  - 输出 world_points [B,S,H,W,3] + confidence            │   │
        │  └─────────────────────────────────────────────────────────┘   │
        │       │                                                         │
        │       v                                                         │
        │  predictions = {pose_enc, depth, depth_conf, world_points,     │
        │                 world_points_conf, images}                     │
        └────────────────────────────────────────────────────────────────┘

        Args:
            images: 输入图像序列 [S, 3, H, W] 或 [B, S, 3, H, W], 值域 [0, 1]
                    S = 序列长度（帧数）
                    H, W = 图像高度和宽度（默认 518x378）
            num_scale_frames: 尺度估计帧数（前N帧一起处理建立坐标系）
                            默认值来自 self.num_frame_for_scale (通常为8)
            keyframe_interval: 关键帧间隔。每N帧存储KV到缓存（1=每帧都存）
            output_device: 输出存储设备。设为 cpu 可避免长序列GPU内存溢出
            test_resolution: (Deprecated) Test resolution parameter.
                           Should be set in load_and_preprocess_images instead.
                           Options: "240p", "360p", "480p"

        Returns:
            predictions 字典:
                - pose_enc: 相机位姿编码 [B, S, 9]
                           可通过 pose_encoding_to_extri_intri() 转换为外参/内参
                - depth: 深度图 [B, S, H, W, 1]
                - depth_conf: 深度置信度 [B, S, H, W]
                - world_points: 3D世界坐标点 [B, S, H, W, 3]
                - world_points_conf: 点置信度 [B, S, H, W]
                - images: 原始图像（用于可视化）
        """
        # ════════════════════════════════════════════════════════════════════
        # 第一阶段：输入标准化和参数准备
        # ════════════════════════════════════════════════════════════════════

        # 【输入形状标准化】
        # 如果输入是 [S, 3, H, W] (无batch维度)，添加batch维度变成 [1, S, 3, H, W]
        # 这样后续处理统一使用带batch的形式
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        B, S, C, H, W = images.shape
        # B = batch大小 (通常为1)
        # S = 序列长度（帧数）
        # C = 通道数 (3, RGB)
        # H, W = 图像高度和宽度

        # 【测试分辨率参数警告】
        # test_resolution 应在 load_and_preprocess_images 阶段设置，而非在此处
        resize_time = 0.0
        if test_resolution is not None:
            logger.warning(f'test_resolution={test_resolution} is deprecated in inference_streaming.')
            logger.warning(f'Please set test_resolution in load_and_preprocess_images instead.')
            logger.warning(f'Ignoring test_resolution, using input image size: {W}x{H}')

        # 【确定scale frames数量】
        # scale frames是前N帧，用于建立全局坐标系的尺度基准
        # 这几帧会一起处理，使用双向注意力（而非因果注意力）
        scale_frames = num_scale_frames if num_scale_frames is not None else self.num_frame_for_scale
        scale_frames = min(scale_frames, S)  # 确保不超过实际帧数

        # 【输出设备辅助函数】
        # 如果设置了output_device (如cpu)，会将预测结果移动到该设备
        # 用于长序列推理时避免GPU内存溢出
        def _to_out(t: torch.Tensor) -> torch.Tensor:
            if output_device is not None:
                return t.to(output_device)
            return t

        # 【清空KV缓存】
        # 开始处理新视频序列前，必须清空上一序列的缓存
        # 这会重置 aggregator 和 camera_head 中的所有 KV cache
        # 确保推理从干净状态开始
        self.clean_kv_cache()

        # ════════════════════════════════════════════════════════════════════
        # 计时：记录总处理开始时间
        # ════════════════════════════════════════════════════════════════════
        start_time = time.time()
        scale_start_time = start_time

        # ════════════════════════════════════════════════════════════════════
        # 第二阶段 Phase 1：处理 Scale Frames（尺度估计帧）
        # ════════════════════════════════════════════════════════════════════

        """
        Scale Frames 处理说明:
        ┌─────────────────────────────────────────────────────────────────┐
        │                                                                  │
        │   Scale Frames = 前 num_scale_frames 帧 (默认8帧)               │
        │                                                                  │
        │   特点:                                                          │
        │   1. 所有 scale frames 一起作为一个 block 处理                  │
        │   2. 通过 scale token 实现双向注意力                             │
        │      - 第1帧可以看到第8帧，第8帧也可以看到第1帧                  │
        │      - 这与后续帧的因果注意力不同                                │
        │   3. 目的：建立稳定的全局坐标系和尺度基准                         │
        │      - 类似 SLAM 的初始化阶段                                    │
        │      - 需要多帧观测来确定场景的物理尺度                          │
        │                                                                  │
        │   Scale Token 工作原理:                                         │
        │   [camera_token] [register_tokens] [scale_token] [patch_tokens] │
        │                                                                  │
        │   scale_token 在前 scale_frames 帧是激活状态                    │
        │   使得这些帧之间可以双向通信                                     │
        │   后续帧的 scale_token 是非激活状态                             │
        │                                                                  │
        └─────────────────────────────────────────────────────────────────┘
        """

        logger.info(f'Processing {scale_frames} scale frames...')
        # 提取前 scale_frames 帧图像
        scale_images = images[:, :scale_frames]

        # 【核心调用】执行 forward() 进行 scale frames 推理
        # forward() 会调用 aggregator 进行特征聚合，然后通过各个 head 进行预测
        # 关键参数:
        # - num_frame_for_scale=scale_frames: 启用 scale token 双向注意力
        # - num_frame_per_block=scale_frames: 整个 scale frames 作为一个 block 处理
        # - causal_inference=True: 标记为因果推理模式（会启用 KV cache）
        scale_output = self.forward(
            scale_images,
            num_frame_for_scale=scale_frames,
            num_frame_per_block=scale_frames,  # 处理所有 scale frames 作为一个 block
            causal_inference=True,
        )

        # 【保存 scale frames 的预测结果】
        # 初始化输出列表，保存预测结果
        # 如果设置了 output_device，结果会被移动到该设备（通常是cpu）
        # 这样可以在推理过程中释放GPU内存
        all_pose_enc = [_to_out(scale_output["pose_enc"])]  # 位姿编码 [B, scale_frames, 9]
        all_depth = [_to_out(scale_output["depth"])] if "depth" in scale_output else []  # 深度图
        all_depth_conf = [_to_out(scale_output["depth_conf"])] if "depth_conf" in scale_output else []  # 深度置信度
        all_world_points = [_to_out(scale_output["world_points"])] if "world_points" in scale_output else []  # 3D点云
        all_world_points_conf = [_to_out(scale_output["world_points_conf"])] if "world_points_conf" in scale_output else []  # 点置信度
        del scale_output  # 释放中间变量内存

        # 【打印 Scale Frames 处理时间】
        scale_end_time = time.time()
        scale_time = scale_end_time - scale_start_time
        logger.info(f'Scale frames ({scale_frames} frames) processed in {scale_time:.2f}s, avg {scale_time/scale_frames:.3f}s/frame')
        streaming_start_time = time.time()

        # ════════════════════════════════════════════════════════════════════
        # 第三阶段 Phase 2：逐帧处理剩余帧（带 KV Cache 的因果推理）
        # ════════════════════════════════════════════════════════════════════

        """
        逐帧因果推理说明:
        ┌─────────────────────────────────────────────────────────────────┐
        │                                                                  │
        │   因果注意力 (Causal Attention):                                 │
        │   - 第 t 帧只能看到第 1~t 帧（包括 scale frames）                 │
        │   - 不能看到第 t+1, t+2... 等未来帧                              │
        │   - 这使得推理可以在线进行（不需要等待后续帧）                    │
        │                                                                  │
        │   KV Cache 工作流程:                                             │
        │                                                                  │
        │   ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                       │
        │   │ K1  │ │ K2  │ │ K3  │ │ K4  │ │ K5  │  <- 已缓存的 Key     │
        │   │ V1  │ │ V2  │ │ V3  │ │ V4  │ │ V5  │  <- 已缓存的 Value   │
        │   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘                       │
        │      └───────────────────────────────────┘                      │
        │                    + 当前帧的 Q, K, V                           │
        │                    ↓                                           │
        │              Attention(Q, [K1..K5+K_curr], [V1..V5+V_curr])     │
        │                                                                  │
        │   优势:                                                          │
        │   1. 不需要重新编码历史帧，只需计算当前帧的 Q, K, V             │
        │   2. 推理速度约 20 FPS，适合实时应用                             │
        │   3. 滑动窗口：超过窗口大小时自动驱逐旧帧的 KV                   │
        │      但保留 scale frames 和 camera/register tokens              │
        │                                                                  │
        │   Keyframe Mode (keyframe_interval > 1):                        │
        │   ┌───────────────────────────────────────────────────┐         │
        │   │ 帧索引:  8  9  10  11  12  13  14  15  16  17      │         │
        │   │ 类型:   SC  K   N   N   K   N   N   K   N   N      │         │
        │   │         │   │   └──┘   │   └──┘   │   └──┘         │         │
        │   │         │   │         │         │                 │         │
        │   │ SC = scale frame (始终缓存)                       │         │
        │   │ K  = keyframe (存储KV到缓存)                      │         │
        │   │ N  = non-keyframe (不存储KV，但参与注意力计算)    │         │
        │   └───────────────────────────────────────────────────┘         │
        │                                                                  │
        │   效果: 内存占用减少约 1/keyframe_interval 倍                   │
        │                                                                  │
        └─────────────────────────────────────────────────────────────────┘
        """

        # 设置进度条，显示逐帧处理进度
        pbar = tqdm(
            range(scale_frames, S),  # 从 scale_frames 开始，处理剩余帧
            desc='Streaming inference',
            initial=scale_frames,
            total=S,
        )

        # 【逐帧循环】处理每一帧
        for i in pbar:
            # 提取当前帧图像 [B, 1, 3, H, W]
            # 每次只处理单帧，这是流式推理的核心
            frame_image = images[:, i:i+1]

            # 【判断是否为关键帧】
            # 如果 keyframe_interval <= 1，所有帧都是关键帧
            # 否则，每隔 keyframe_interval 帧是一个关键帧
            # 关键帧的 KV 会被持久化存储，非关键帧的 KV 只临时参与计算
            is_keyframe = (keyframe_interval <= 1) or ((i - scale_frames) % keyframe_interval == 0)

            # 【设置跳过缓存模式】
            # 对于非关键帧，设置 _skip_append = True
            # 这意味着当前帧会参与注意力计算，但不会存储到 KV cache
            # 这减少了内存占用，适用于长序列推理
            if not is_keyframe:
                self._set_skip_append(True)  # 设置 aggregator 和 camera_head 的缓存跳过模式

            # 【核心调用】执行 forward() 进行单帧推理
            # forward() 流程:
            #   1. aggregator:
            #      - DINOv2 patch embedding 提取当前帧特征
            #      - frame_blocks: 帧内自注意力
            #      - global_blocks: 与 KV cache 中所有历史帧的因果注意力
            #   2. camera_head:
            #      - 提取 camera token
            #      - CameraBlock: 与历史 camera tokens 的因果注意力
            #      - 迭代优化预测位姿
            #   3. depth_head: 从多尺度特征解码深度
            #   4. point_head: 从多尺度特征解码3D点坐标
            frame_output = self.forward(
                frame_image,
                num_frame_for_scale=scale_frames,  # 保持 scale token 逻辑一致
                num_frame_per_block=1,  # 单帧处理
                causal_inference=True,
            )

            # 【恢复缓存模式】
            # 处理完非关键帧后，恢复 _skip_append = False
            # 这样下一帧如果是关键帧，会正常存储 KV
            if not is_keyframe:
                self._set_skip_append(False)

            # 【保存当前帧的预测结果】
            # 将预测添加到输出列表，如果设置了 output_device 则移动到该设备
            all_pose_enc.append(_to_out(frame_output["pose_enc"]))
            if "depth" in frame_output:
                all_depth.append(_to_out(frame_output["depth"]))
            if "depth_conf" in frame_output:
                all_depth_conf.append(_to_out(frame_output["depth_conf"]))
            if "world_points" in frame_output:
                all_world_points.append(_to_out(frame_output["world_points"]))
            if "world_points_conf" in frame_output:
                all_world_points_conf.append(_to_out(frame_output["world_points_conf"]))
            del frame_output  # 释放当前帧输出的内存

        # 【打印逐帧处理时间统计】
        streaming_end_time = time.time()
        streaming_time = streaming_end_time - streaming_start_time
        streaming_frames = S - scale_frames
        if streaming_frames > 0:
            logger.info(f'Streaming frames ({streaming_frames} frames) processed in {streaming_time:.2f}s, avg {streaming_time/streaming_frames:.3f}s/frame')

        # ════════════════════════════════════════════════════════════════════
        # 第四阶段：合并所有预测结果
        # ════════════════════════════════════════════════════════════════════

        """
        结果合并说明:
        ┌─────────────────────────────────────────────────────────────────┐
        │                                                                  │
        │   all_pose_enc = [scale_pose, frame8_pose, frame9_pose, ...]    │
        │                    ↓                                            │
        │   torch.cat(all_pose_enc, dim=1)                                │
        │                    ↓                                            │
        │   pose_enc = [B, S, 9]  <- 包含所有帧的位姿预测                 │
        │                                                                  │
        │   合并后的输出维度:                                              │
        │   - pose_enc:      [B, S, 9]      位姿编码                      │
        │   - depth:         [B, S, H, W, 1] 深度图                       │
        │   - depth_conf:    [B, S, H, W]   深度置信度                    │
        │   - world_points:  [B, S, H, W, 3] 3D世界坐标                   │
        │   - world_points_conf: [B, S, H, W] 点置信度                    │
        │   - images:        [B, S, 3, H, W] 原始图像                     │
        │                                                                  │
        │   注意: 如果设置了 output_device='cpu'，所有张量在cpu上        │
        │   这样可以在推理过程中释放GPU内存，处理长序列                   │
        │                                                                  │
        └─────────────────────────────────────────────────────────────────┘
        """

        # 【释放GPU内存】
        # 如果设置了 output_device（如cpu），将图像移动到该设备
        # 然后清空 KV cache 并释放 GPU 内存
        # 这对于处理长序列（数千帧）非常重要
        if output_device is not None:
            # 将图像移动到输出设备
            images_out = _to_out(images)
            del images  # 删除 GPU 上的图像引用
            # 清理 KV cache（推理完成后不再需要）
            self.clean_kv_cache()
            # 强制清空 CUDA 缓存，释放内存给其他任务使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            images_out = images

        # 【沿序列维度合并所有预测】
        # torch.cat 将列表中的张量在第1维（序列维度）上合并
        # 例如: [B, scale_frames, 9] + [B, 1, 9] + ... -> [B, S, 9]
        predictions = {
            "pose_enc": torch.cat(all_pose_enc, dim=1),  # [B, S, 9]
        }
        del all_pose_enc  # 释放临时列表

        if all_depth:
            predictions["depth"] = torch.cat(all_depth, dim=1)  # [B, S, H, W, 1]
        del all_depth

        if all_depth_conf:
            predictions["depth_conf"] = torch.cat(all_depth_conf, dim=1)  # [B, S, H, W]
        del all_depth_conf

        if all_world_points:
            predictions["world_points"] = torch.cat(all_world_points, dim=1)  # [B, S, H, W, 3]
        del all_world_points

        if all_world_points_conf:
            predictions["world_points_conf"] = torch.cat(all_world_points_conf, dim=1)  # [B, S, H, W]
        del all_world_points_conf

        # 【保存原始图像】
        # 用于可视化时将点云颜色映射回原图
        predictions["images"] = images_out

        # 【预测归一化】（可选）
        # 如果启用了 pred_normalization，会对预测结果进行归一化处理
        # 主要用于训练时的稳定性，推理时通常不启用
        if self.pred_normalization:
            predictions = self._normalize_predictions(predictions)

        # ════════════════════════════════════════════════════════════════════
        # 计时：打印总处理时间统计
        # ════════════════════════════════════════════════════════════════════
        total_time = time.time() - start_time
        avg_time_per_frame = total_time / S
        fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
        logger.info(f'Total: {S} frames ({W}x{H}) processed in {total_time:.2f}s, avg {avg_time_per_frame:.3f}s/frame ({fps:.1f} FPS)')

        return predictions
