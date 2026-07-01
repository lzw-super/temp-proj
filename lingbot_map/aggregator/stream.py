"""
AggregatorStream - 带 FlashInfer KV Cache 的流式因果聚合器

提供:
- 时间因果注意力 (Temporal causal attention)
- 滑动窗口支持 (Sliding window support)
- Scale Token 用于尺度估计帧
- 使用 FlashInfer 分页 KV Cache 的流式推理

【核心功能】

本模块实现了 GCT 模型的流式推理核心:
1. 因果注意力: 帧 t 只能看到帧 0~t，实现真正的在线推理
2. KV Cache: 存储历史帧的 Key/Value，避免重复计算
3. Scale Token: 前 N 帧使用双向注意力，建立全局坐标系
4. 滑动窗口: 超过窗口大小时自动驱逐旧帧，控制内存

【流式推理流程】

┌─────────────────────────────────────────────────────────────────────┐
│                    流式推理完整流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  【Phase 1: Scale Frames 处理】                                     │
│                                                                     │
│  前 num_frame_for_scale 帧 (默认8帧) 一起处理:                      │
│  │   - 使用双向注意力（通过 scale token 激活）                       │
│  │   - 帧0可以看到帧7，帧7也可以看到帧0                             │
│  │   - 目的: 建立全局坐标系的尺度基准                               │
│  │   - 类似 SLAM 的初始化阶段                                       │
│  │   - KV cache 存储: scale_frames 的所有 KV                        │
│                                                                     │
│  【Phase 2: 逐帧因果推理】                                           │
│                                                                     │
│  从第 num_frame_for_scale+1 帧开始，逐帧处理:                        │
│  │   - 因果注意力: 帧 t 只能看到帧 0~t                              │
│  │   - KV cache:                                                    │
│  │   │   ┌─────────────────────────────────────────────────────┐    │
│  │   │   │ 存储:                                                  │    │
│  │   │   │   - Scale frames 的 KV (始终保留)                     │    │
│  │   │   │   - 最近 kv_cache_sliding_window 帧的 KV              │    │
│  │   │   │   - 特殊 tokens (camera, register, scale)             │    │
│  │   │   │                                                         │    │
│  │   │   │ 计算:                                                   │    │
│  │   │   │   Q_cur = 当前帧的 Query                               │    │
│  │   │   │   K_cache = 缓存的历史 Key                             │    │
│  │   │   │   V_cache = 缓存的历史 Value                           │    │
│  │   │   │                                                         │    │
│  │   │   │   Attention(Q_cur, [K_scale + K_window + K_cur],      │    │
│  │   │   │             [V_scale + V_window + V_cur])             │    │
│  │   │   │                                                         │    │
│  │   │   │ 存储当前帧 KV:                                          │    │
│  │   │   │   - 如果是关键帧: 存入 cache                           │    │
│  │   │   │   - 如果不是关键帧: 只参与计算，不存储                  │    │
│  │   │   │                                                         │    │
│  │   │   │ 驱逐:                                                   │    │
│  │   │   │   - 超过 sliding_window 时驱逐最旧帧                   │    │
│  │   │   │   - 但保留 scale frames 和 special tokens              │    │
│  │   │   └─────────────────────────────────────────────────────┘    │
│  │   │                                                              │
│  │   - 推理速度: ~20 FPS                                            │
│                                                                     │
│  【KV Cache Backend】                                               │
│                                                                     │
│  两种实现方式:                                                       │
│  │   1. FlashInfer (推荐):                                          │
│  │   │   - 使用分页 KV cache                                        │
│  │   │   - 高效的内存管理                                           │
│  │   │   - 支持变长序列                                             │
│  │   │   - 需要额外安装 flashinfer-python                          │
│  │   │                                                              │
│  │   2. SDPA (PyTorch原生):                                         │
│  │   │   - 基于 dict 的简单 KV cache                                │
│  │   │   - 无需额外依赖                                             │
│  │   │   - 性能略低于 FlashInfer                                    │
│  │   │                                                              │
│                                                                     │
│  【关键帧模式】                                                      │
│                                                                     │
│  keyframe_interval > 1 时启用:                                      │
│  │   - 每隔 N 帧存储 KV 到 cache                                    │
│  │   - 非关键帧: 参与计算但不存储                                    │
│  │   - 内存占用减少约 1/N                                           │
│  │   - 适合长序列推理                                                │
│                                                                     │
│  【Flow-based 关键帧】                                              │
│                                                                     │
│  flow_threshold > 0 时启用:                                         │
│  │   - 计算当前帧与上次关键帧的光流                                 │
│  │   - 光流超过阈值时成为新关键帧                                   │
│  │   - 自适应关键帧选择                                             │
│  │   - 更好的运动估计                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

import logging
import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from lingbot_map.layers.block import Block, FlashInferBlock, SDPABlock
from lingbot_map.layers.rope import WanRotaryPosEmbed
from lingbot_map.aggregator.base import AggregatorBase, slice_expand_and_flatten

logger = logging.getLogger(__name__)


class AggregatorStream(AggregatorBase):
    """
    带 FlashInfer 分页 KV Cache 的流式因果聚合器

    特点:
    - 时间因果注意力（每帧只关注过去帧）
    - 滑动窗口支持限制注意力范围
    - Scale Token 用于尺度估计帧
    - 使用 FlashInfer KV Cache 的流式推理

    【设计说明】

    本类继承 AggregatorBase，实现跨帧因果注意力的具体逻辑:
    - _process_global_attention() 调用 _process_causal_stream()
    - _process_causal_stream() 是流式推理的核心

    KV Cache 的两种 Backend:
    - FlashInfer: 高效的分页 KV cache，推荐使用
    - SDPA: PyTorch 原生实现，dict-based cache

    【参数说明】

    Causal-specific parameters:
    - sliding_window_size: 滑动窗口大小（blocks）
    - num_frame_for_scale: 尺度估计帧数（前N帧双向注意力）
    - num_random_frames: 随机帧数（长距离依赖）
    - attend_to_special_tokens: 跨帧特殊 token 注意力
    - attend_to_scale_frames: 包含 scale frames 的注意力
    - enable_3d_rope: 启用 3D RoPE（时间位置编码）
    - max_frame_num: 3D RoPE 最大帧数

    KV cache parameters:
    - kv_cache_sliding_window: 滑动窗口大小（帧数）
    - kv_cache_scale_frames: scale frames 数量（始终保留）
    - kv_cache_cross_frame_special: 保留被驱逐帧的特殊 tokens
    - kv_cache_include_scale_frames: scale frames 是否包含在 cache
    - kv_cache_camera_only: 仅保留 camera tokens
    """

    def __init__(
        self,
        # Causal-specific parameters (因果模式特定参数)
        sliding_window_size: int = -1,  # -1 表示全因果
        num_frame_for_scale: int = 1,   # 尺度帧数
        num_random_frames: int = 0,     # 随机帧数
        attend_to_special_tokens: bool = False,  # 跨帧特殊token注意力
        attend_to_scale_frames: bool = False,    # 包含scale frames
        enable_3d_rope: bool = False,            # 3D RoPE
        max_frame_num: int = 1024,               # 最大帧数
        use_anchor_token: bool = False,          # Stage2 GCA anchor token
        use_scale_token: bool = True,            # Legacy scale token
        # KV cache parameters (KV cache参数)
        kv_cache_sliding_window: int = 64,       # 滑动窗口（帧数）
        kv_cache_scale_frames: int = 8,          # scale frames数
        kv_cache_cross_frame_special: bool = True,  # 跨帧特殊token
        kv_cache_include_scale_frames: bool = True,  # 包含scale frames
        kv_cache_camera_only: bool = False,      # 仅camera tokens
        # Base class parameters via **kwargs
        **kwargs
    ):
        """
        初始化 AggregatorStream

        Args:
            sliding_window_size: 滑动窗口大小（blocks）
                - -1: 全因果，无窗口限制
                - N: 只关注最近N个blocks
            num_frame_for_scale: 尺度估计帧数
                - 前 N 帧使用双向注意力
                - 用于建立全局坐标系
                - 默认 8
            num_random_frames: 随机帧数
                - 用于长距离依赖（已弃用）
            attend_to_special_tokens: 跨帧特殊token注意力
                - 是否允许不同帧的特殊token互相看到
            attend_to_scale_frames: 包含scale frames的注意力
                - 后续帧是否可以看到scale frames
            enable_3d_rope: 启用3D RoPE
                - 将时间维度纳入位置编码
                - 提高流式推理的时间一致性
            max_frame_num: 最大帧数
                - 用于3D RoPE的位置范围
            kv_cache_sliding_window: KV cache滑动窗口
                - 最大缓存的帧数
                - 超过时驱逐最旧帧
            kv_cache_scale_frames: Scale frames数量
                - 这些帧的KV始终保留
            kv_cache_cross_frame_special: 跨帧特殊token
                - 保留被驱逐帧的特殊tokens
            kv_cache_include_scale_frames: 包含scale frames
                - scale frames是否计入sliding window
            kv_cache_camera_only: 仅camera tokens
                - 驱逐时只保留camera token
            **kwargs: 基类参数
        """
        # ════════════════════════════════════════════════════════════════════
        # 存储因果模式特定参数
        # ════════════════════════════════════════════════════════════════════
        self.sliding_window_size = sliding_window_size
        self.num_frame_for_scale = num_frame_for_scale
        self.num_random_frames = num_random_frames
        self.attend_to_special_tokens = attend_to_special_tokens
        self.attend_to_scale_frames = attend_to_scale_frames
        self.enable_3d_rope = enable_3d_rope
        self.max_frame_num = max_frame_num
        self.use_anchor_token = use_anchor_token
        self.use_scale_token = use_scale_token

        # ════════════════════════════════════════════════════════════════════
        # 存储 KV cache 参数
        # ════════════════════════════════════════════════════════════════════
        self.kv_cache_sliding_window = kv_cache_sliding_window
        self.kv_cache_scale_frames = kv_cache_scale_frames
        self.kv_cache_cross_frame_special = kv_cache_cross_frame_special
        self.kv_cache_include_scale_frames = kv_cache_include_scale_frames
        self.kv_cache_camera_only = kv_cache_camera_only

        # ════════════════════════════════════════════════════════════════════
        # 处理 kwargs，选择 backend
        # ════════════════════════════════════════════════════════════════════
        kwargs.pop('enable_stream_inference', None)
        use_flashinfer = kwargs.pop('use_flashinfer', True)
        kwargs.pop('use_flexflash', None)
        use_sdpa = kwargs.pop('use_sdpa', False)

        # Backend 选择: SDPA (无额外依赖) 或 FlashInfer (分页KV cache)
        self.use_sdpa = use_sdpa
        self.use_flashinfer = not use_sdpa  # FlashInfer 是默认backend

        # 调用父类 __init__
        super().__init__(**kwargs)

        # 初始化 KV cache
        self._init_kv_cache()

        # 初始化 3D RoPE（如果启用）
        if self.enable_3d_rope:
            self._init_3d_rope()

    def _build_blocks(
        self,
        block_fn,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        init_values: float,
        qk_norm: bool,
    ):
        """Build frame and global blocks for streaming causal mode."""
        block_params = dict(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
        )

        # Frame blocks: Standard Block + RoPE
        self.frame_blocks = nn.ModuleList([
            block_fn(**block_params, rope=self.rope)
            for _ in range(depth)
        ])

        # Global blocks: FlashInferBlock (default) or SDPABlock (fallback)
        GlobalBlockCls = SDPABlock if self.use_sdpa else FlashInferBlock
        self.global_blocks = nn.ModuleList([
            GlobalBlockCls(
                **block_params,
                rope=self.rope if not self.disable_global_rope else None,
                kv_cache_sliding_window=self.kv_cache_sliding_window,
                kv_cache_scale_frames=self.kv_cache_scale_frames,
                kv_cache_cross_frame_special=self.kv_cache_cross_frame_special,
                kv_cache_include_scale_frames=self.kv_cache_include_scale_frames,
                kv_cache_camera_only=self.kv_cache_camera_only,
            )
            for _ in range(depth)
        ])

    def _setup_special_tokens(self):
        """Setup camera, register, anchor, and/or scale tokens for causal mode."""
        # Camera token
        self.camera_token = nn.Parameter(
            torch.randn(1, 2, 1, self.embed_dim)
        )

        # Register tokens
        if self.num_register_tokens > 0:
            self.register_token = nn.Parameter(
                torch.randn(1, 2, self.num_register_tokens, self.embed_dim)
            )

        if self.use_anchor_token:
            self.anchor_token = nn.Parameter(
                torch.ones(1, 2, 1, self.embed_dim)
            )

        if self.use_scale_token:
            self.scale_token = nn.Parameter(
                torch.ones(1, 2, 1, self.embed_dim)
            )

        # Initialize
        nn.init.normal_(self.camera_token, std=1e-6)
        if self.num_register_tokens > 0:
            nn.init.normal_(self.register_token, std=1e-6)
        if self.use_anchor_token:
            nn.init.normal_(self.anchor_token, std=1e-6)
        if self.use_scale_token:
            nn.init.normal_(self.scale_token, std=1e-6)

        # Token indexing:
        # camera + register + optional anchor + optional scale + patches.
        self.num_anchor_tokens = 1 if self.use_anchor_token else 0
        self.num_scale_tokens = 1 if self.use_scale_token else 0
        self.patch_start_idx = (
            1
            + self.num_register_tokens
            + self.num_anchor_tokens
            + self.num_scale_tokens
        )
        self.num_special_tokens = self.patch_start_idx

    def _init_kv_cache(self):
        """
        【KV Cache 初始化】为流式推理准备缓存结构

        本方法初始化两种类型的 KV Cache:
        1. FlashInfer (推荐): 懒初始化的分页 KV Cache
        2. SDPA: 基于 dict 的简单 KV Cache

        ╔════════════════════════════════════════════════════════════════════╗
        ║                      KV Cache 结构说明                             ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【FlashInfer KV Cache】                                          ║
        ║                                                                    ║
        ║  结构: FlashInferKVCacheManager                                   ║
        ║  │   - 分页管理: 类似操作系统的虚拟内存                            ║
        ║  │   - 高效内存分配: 动态增长和回收                               ║
        ║  │   - 支持变长序列: 不同帧可能有不同token数                       ║
        ║  │                                                                  ║
        ║  懒初始化:                                                          ║
        ║  │   - kv_cache_manager = None (初始)                              ║
        ║  │   - 首次 forward 时创建 manager                                 ║
        ║  │   - 根据实际图像尺寸确定 tokens_per_frame                       ║
        ║                                                                    ║
        ║  参数:                                                              ║
        ║  │   num_blocks: Transformer层数 (24)                             ║
        ║  │   max_num_frames: 最大缓存帧数                                  ║
        ║  │   tokens_per_frame: 每帧token数                                 ║
        ║  │   num_heads: 注意力头数 (16)                                    ║
        ║  │   head_dim: 每个头的维度 (64)                                   ║
        ║                                                                    ║
        ║  【SDPA KV Cache】                                                 ║
        ║                                                                    ║
        ║  结构: Python dict                                                 ║
        ║  │   kv_cache = {                                                   ║
        ║  │     "k_0": None,  # Block 0 的 Key cache                       ║
        ║  │     "v_0": None,  # Block 0 的 Value cache                     ║
        ║  │     "k_0_special": None,  # Block 0 的特殊token Key            ║
        ║  │     "v_0_special": None,  # Block 0 的特殊token Value          ║
        ║  │     ...                                                           ║
        ║  │     "k_23": None,                                                 ║
        ║  │     "v_23": None,                                                 ║
        ║  │     "k_23_special": None,                                        ║
        ║  │     "v_23_special": None,                                        ║
        ║  │   }                                                               ║
        ║                                                                    ║
        ║  Cache 存储时的形状:                                               ║
        ║  │   k_i: [B, 1, S_cached, P, C]                                    ║
        ║  │   v_i: [B, 1, S_cached, P, C]                                    ║
        ║  │   k_i_special: [B, 1, S_cached, num_special, C]                 ║
        ║  │   v_i_special: [B, 1, S_cached, num_special, C]                 ║
        ║  │                                                                  │
        ║  │   S_cached: 已缓存的帧数                                         ║
        ║  │   P: 每帧token数                                                 ║
        ║  │   num_special: 特殊token数 (6)                                  ║
        ║                                                                    ║
        ║  特殊keys:                                                          ║
        ║  │   "_skip_append": False  # 控制是否存储当前帧KV               ║
        ║  │   "_defer_eviction": False  # 控制是否延迟驱逐               ║
        ║                                                                    ║
        ║  【相关计数器】                                                    ║
        ║                                                                    ║
        ║  total_frames_processed: int                                       ║
        ║  │   - 已处理的累计帧数                                            ║
        ║  │   - 用于 3D RoPE 的全局帧索引                                   ║
        ║  │   - 每次 forward 后更新                                         ║
        ║                                                                    ║
        ║  _cached_pos3d: tensor                                             ║
        ║  │   - 缓存的 3D RoPE 位置编码                                     ║
        ║  │   - 避免重复计算                                                ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        # FlashInfer manager: 懒初始化（首次 forward 时创建）
        self.kv_cache_manager = None

        # SDPA: dict-based cache
        self.kv_cache = {}

        # 已处理帧数计数器
        self.total_frames_processed = 0

        # 缓存的 3D RoPE 位置
        self._cached_pos3d = None

        if self.use_sdpa:
            # SDPA backend: 为每个 block 创建 cache entries
            if hasattr(self, 'depth'):
                for i in range(self.depth):
                    # 每个 block 有两组 cache:
                    # - 常规 cache (用于所有 tokens)
                    # - special cache (用于 camera/register/scale tokens)
                    self.kv_cache[f"k_{i}"] = None
                    self.kv_cache[f"v_{i}"] = None
                    self.kv_cache[f"k_{i}_special"] = None
                    self.kv_cache[f"v_{i}_special"] = None
                logger.info(f"SDPA KV cache initialized with {self.depth} blocks")
        else:
            logger.info("FlashInfer KV cache will be lazily initialized on first forward")

    def _get_flashinfer_manager(self, device, dtype, tokens_per_frame=None):
        """
        【FlashInfer Manager 懒初始化】获取或创建 FlashInfer KV Cache Manager

        FlashInfer 使用分页 KV Cache，在首次使用时才创建。
        这种懒初始化方式可以根据实际输入尺寸确定 cache 参数。

        ╔════════════════════════════════════════════════════════════════════╗
        ║              FlashInferKVCacheManager 创建流程                     ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【触发条件】                                                      ║
        ║  │   kv_cache_manager is None                                      ║
        ║  │   首次调用 _process_causal_stream 时                            ║
        ║                                                                    ║
        ║  【参数计算】                                                      ║
        ║                                                                    ║
        ║  num_heads = embed_dim // 64 = 1024 // 64 = 16                    ║
        ║  │   ViT-L 的标准注意力头数                                        ║
        ║                                                                    ║
        ║  head_dim = 64                                                      ║
        ║  │   ViT-L 的每个头的维度                                          ║
        ║                                                                    ║
        ║  tokens_per_frame:                                                  ║
        ║  │   if None:                                                      ║
        ║  │       tokens_per_frame = (img_size // patch_size)^2 + num_spec │
        ║  │       例如: (518//14)^2 + 6 ≈ 37^2 + 6 ≈ 1375                  ║
        ║  │   else:                                                         ║
        ║  │       tokens_per_frame = 实际值                                 ║
        ║  │       支持非标准图像尺寸                                        ║
        ║                                                                    ║
        ║  max_num_frames:                                                    ║
        ║  │   max_num_frames = scale_frames + sliding_window + 16          │
        ║  │   例如: 8 + 64 + 16 = 88                                        ║
        ║  │   预留一定的 headroom                                           ║
        ║                                                                    ║
        ║  【Manager 结构】                                                  ║
        ║                                                                    ║
        ║  FlashInferKVCacheManager:                                         ║
        ║  │   num_blocks: 24 (Transformer层数)                              ║
        ║  │   max_num_frames: 88 (最大缓存帧数)                             ║
        ║  │   tokens_per_frame: ~1375                                       ║
        ║  │   num_heads: 16                                                  ║
        ║  │   head_dim: 64                                                   ║
        ║  │   num_special_tokens: 6                                          ║
        ║  │   scale_frames: 8                                                ║
        ║  │   sliding_window: 64                                             ║
        ║                                                                    ║
        ║  【分页机制】                                                      ║
        ║                                                                    ║
        ║  类似操作系统的虚拟内存:                                           ║
        ║  │   - 物理页: 固定大小的内存块                                    ║
        ║  │   - 逻辑页: 映射到物理页                                        ║
        ║  │   - 页表: 管理映射关系                                          ║
        ║  │                                                                  │
        ║  │   优势:                                                          │
        ║  │   1. 动态内存分配: 按需分配页                                    │
        ║  │   2. 高效驱逐: 快速回收旧页                                     ║
        ║  │   3. 变长支持: 不同帧可占用不同页数                             ║
        ║                                                                    ║
        ║  【返回值】                                                        ║
        ║                                                                    ║
        ║  manager: FlashInferKVCacheManager                                 ║
        ║  │   - 用于所有 global_blocks 的 KV cache                         ║
        ║  │   - 支持 append_frame, evict, rollback 等操作                  ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            device: 计算设备 (cuda/cpu)
            dtype: 数据类型 (bfloat16/float16/float32)
            tokens_per_frame: 每帧token数（可选）
                - None: 使用默认计算
                - 实际值: 支持非标准尺寸

        Returns:
            FlashInferKVCacheManager: 分页 KV Cache 管理器
        """
        if self.kv_cache_manager is None:
            # 懒初始化: 首次使用时创建
            from lingbot_map.layers.flashinfer_cache import FlashInferKVCacheManager

            # 计算注意力头参数
            num_heads = self.embed_dim // 64  # head_dim = 64 for ViT-L
            head_dim = 64

            # 计算 tokens_per_frame
            if tokens_per_frame is None:
                # 默认: 假设正方形图像
                tokens_per_frame = (self.img_size // self.patch_size) ** 2 + self.num_special_tokens

            # 计算最大缓存帧数
            # max_num_frames = scale frames + sliding window + headroom
            max_num_frames = self.kv_cache_scale_frames + self.kv_cache_sliding_window + 16

            # 创建 FlashInfer KV Cache Manager
            self.kv_cache_manager = FlashInferKVCacheManager(
                num_blocks=self.depth,  # 24 transformer blocks
                max_num_frames=max_num_frames,
                tokens_per_frame=tokens_per_frame,
                num_heads=num_heads,
                head_dim=head_dim,
                dtype=dtype,
                device=device,
                num_special_tokens=self.num_special_tokens,
                scale_frames=self.kv_cache_scale_frames,
                sliding_window=self.kv_cache_sliding_window,
                max_total_frames=self.max_frame_num + 100,  # 预留headroom
                force_fp32=getattr(self, 'kv_cache_force_fp32', False),  # 强制FP32
                fa3=getattr(self, 'kv_cache_fa3', False),  # FlashAttention 3
            )

            logger.info(
                f"FlashInfer KV cache manager initialized: {self.depth} blocks, "
                f"max_frames={max_num_frames}, tokens_per_frame={tokens_per_frame}"
            )

        return self.kv_cache_manager

    def clean_kv_cache(self):
        """
        【清理 KV Cache】开始处理新序列时调用

        当开始处理新的视频序列时，必须清空上一序列的缓存。
        这确保推理从干净状态开始，不会受到旧数据的影响。

        ╔════════════════════════════════════════════════════════════════════╗
        ║                    clean_kv_cache 处理流程                        ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【处理步骤】                                                      ║
        ║                                                                    ║
        ║  步骤1: 重置 FlashInfer manager                                    ║
        ║  │   if kv_cache_manager is not None:                              ║
        ║  │       kv_cache_manager.reset()                                  ║
        ║  │       - 清空所有页                                              ║
        ║  │       - 重置计数器                                              ║
        ║  │       - 释放内存                                                ║
        ║                                                                    ║
        ║  步骤2: 清空 SDPA dict cache                                       ║
        ║  │   if kv_cache:                                                  ║
        ║  │       for key in kv_cache.keys():                               ║
        ║  │           if key == "_skip_append":                             ║
        ║  │               kv_cache[key] = False  # 重置标志                 ║
        ║  │           else:                                                   ║
        ║  │               kv_cache[key] = None   # 清空数据                 ║
        ║                                                                    ║
        ║  步骤3: 重置计数器                                                  ║
        ║  │   total_frames_processed = 0                                    ║
        ║  │   _cached_pos3d = None                                          ║
        ║                                                                    ║
        ║  【调用时机】                                                      ║
        ║                                                                    ║
        ║  inference_streaming() 开始时:                                    ║
        ║  │   self.clean_kv_cache()                                         ║
        ║  │   确保新序列从空 cache 开始                                     ║
        ║                                                                    ║
        ║  【特殊情况】                                                      ║
        ║                                                                    ║
        ║  如果 aggregator 不支持 KV cache 清理:                            ║
        ║  │   logger.warning("Aggregator does not support KV cache cleaning")║
        ║                                                                    ║
        ║  camera_head 也需要清理:                                           ║
        ║  │   camera_head.clean_kv_cache()                                  ║
        ║  │   相机头有独立的 KV cache                                       ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        # 重置 FlashInfer manager
        if self.kv_cache_manager is not None:
            self.kv_cache_manager.reset()

        # 清空 SDPA dict cache
        if self.kv_cache:
            for key in list(self.kv_cache.keys()):
                if key == "_skip_append":
                    # 重置标志位，而不是删除
                    self.kv_cache[key] = False
                else:
                    # 清空 KV 数据
                    self.kv_cache[key] = None

        # 重置计数器
        self.total_frames_processed = 0
        self._cached_pos3d = None

        logger.info("KV cache cleaned")

    def _init_3d_rope(self):
        """Initialize 3D RoPE for streaming inference."""
        if not self.enable_3d_rope:
            self.rope3d = None
            return

        # Dynamically compute num_heads based on embed_dim
        # ViT-L: embed_dim=1024, num_heads=16, head_dim=64
        # ViT-B: embed_dim=768, num_heads=12, head_dim=64
        # ViT-S: embed_dim=384, num_heads=6, head_dim=64
        num_heads = self.embed_dim // 64
        head_dim = 64

        self.rope3d = WanRotaryPosEmbed(
            attention_head_dim=head_dim,
            patch_size=(1, self.patch_size, self.patch_size),
            max_seq_len=self.max_frame_num,
        )
        logger.info(f"3D RoPE initialized for max {self.max_frame_num} frames, head_dim={head_dim}, num_heads={num_heads}")

    def _get_3d_positions_streaming(self, num_frames, H, W, device, f_start, f_end):
        """
        Generate 3D RoPE positions for streaming mode with correct global frame indices.

        Args:
            num_frames: Number of frames in current batch
            H, W: Image height and width
            device: Device to create positions on
            f_start: Global start frame index
            f_end: Global end frame index

        Returns:
            pos3d: [1, 1, num_frames * P, head_dim//2] complex tensor
        """
        if self.rope3d is None:
            return None

        pph = H // self.patch_size
        ppw = W // self.patch_size

        pos3d = self.rope3d(
            ppf=num_frames,
            pph=pph,
            ppw=ppw,
            patch_start_idx=self.num_special_tokens,
            device=device,
            f_start=f_start,
            f_end=f_end
        )
        return pos3d

    def _prepare_special_tokens(
        self,
        B: int,
        S_local: int,
        S_global: int,
        C: int,
        num_frame_for_scale: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Prepare camera, register, anchor, and/or scale tokens.

        Args:
            B: Batch size
            S_local: Local sequence length
            S_global: Global sequence length
            C: Embedding dimension
            num_frame_for_scale: Number of frames for scale estimation

        Returns:
            Special tokens [B*S_global, N_special, C]
        """
        # Get effective num_frame_for_scale
        scale_frames = self.num_frame_for_scale if num_frame_for_scale is None else num_frame_for_scale

        # Check cache state for both backends
        has_flashinfer_cache = self.kv_cache_manager is not None and self.kv_cache_manager.num_frames > 0
        has_sdpa_cache = self.kv_cache is not None and self.kv_cache.get("k_0") is not None

        # Determine if we're in causal inference mode based on KV cache state
        causal_inference = True

        if causal_inference and has_flashinfer_cache:
            S_cached = self.kv_cache_manager.num_frames
            S_true = S_cached + S_global
        elif causal_inference and has_sdpa_cache:
            _, _, S_cached, _, _ = self.kv_cache["k_0"].shape
            S_true = S_cached + S_global
        else:
            S_true = S_global

        # Expand tokens based on mode
        if causal_inference and S_true > S_global: #存在cache且因果推理
            # Streaming mode: expand with S_true, then slice to get current frames
            effective_scale_frames = min(scale_frames, S_true)

            special_tokens = [
                slice_expand_and_flatten(self.camera_token, B, S_true)[-S_global:, :, :]
            ]
            if self.num_register_tokens > 0:
                special_tokens.append(
                    slice_expand_and_flatten(self.register_token, B, S_true)[-S_global:, :, :]
                )
            if self.use_anchor_token:
                special_tokens.append(
                    slice_expand_and_flatten(
                        self.anchor_token,
                        B,
                        S_true,
                        first_num_frame=effective_scale_frames,
                    )[-S_global:, :, :]
                )
            if self.use_scale_token:
                special_tokens.append(
                    slice_expand_and_flatten(
                        self.scale_token,
                        B,
                        S_true,
                        first_num_frame=effective_scale_frames,
                    )[-S_global:, :, :]
                )
        else:
            # Batch mode or first inference: expand directly
            effective_scale_frames = min(scale_frames, S_global)

            special_tokens = [slice_expand_and_flatten(self.camera_token, B, S_global)]
            if self.num_register_tokens > 0:
                special_tokens.append(slice_expand_and_flatten(self.register_token, B, S_global))
            if self.use_anchor_token:
                special_tokens.append(
                    slice_expand_and_flatten(
                        self.anchor_token,
                        B,
                        S_global,
                        first_num_frame=effective_scale_frames,
                    )
                )
            if self.use_scale_token:
                special_tokens.append(
                    slice_expand_and_flatten(
                        self.scale_token,
                        B,
                        S_global,
                        first_num_frame=effective_scale_frames,
                    )
                )

        special_tokens = torch.cat(special_tokens, dim=1)

        # Verify shape
        expected_shape = (B * S_global, self.num_special_tokens, C)
        assert special_tokens.shape == expected_shape, \
            f"Expected {expected_shape}, got {special_tokens.shape}"

        return special_tokens

    def _process_global_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S_local: int,
        S_global: int,
        P: int,
        C: int,
        global_idx: int,
        pos: Optional[torch.Tensor] = None,
        # Mode-specific parameters (模式特定参数)
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
        **kwargs,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        【跨帧因果注意力处理入口】

        本方法是 _process_global_attention 抽象方法的具体实现，
        调用 _process_causal_stream 执行实际的因果注意力计算。

        处理流程:
        ┌────────────────────────────────────────────────────────────────────┐
        │                                                                     │
        │  输入:                                                               │
        │  │   tokens: [B*S_local, P, C]                                      │
        │  │   各种参数                                                        │
        │                                                                     │
        │  提取图像尺寸:                                                        │
        │  │   image_height = kwargs.get('image_height', self.img_size)      │
        │  │   image_width = kwargs.get('image_width', self.img_size)        │
        │                                                                     │
        │  调用核心函数:                                                        │
        │  │   _process_causal_stream(                                        │
        │  │       tokens, B, S_local, S_global, P, C, global_idx, pos,      │
        │  │       num_frame_per_block, sliding_window_size, num_frame_for_  │
        │  │       scale, image_height, image_width                           │
        │  │   )                                                              │
        │                                                                     │
        │  返回:                                                               │
        │  │   (tokens, global_idx, intermediates)                           │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            tokens: 输入 tokens [B*S_local, P, C]
            B: Batch size
            S_local: 本地序列长度
            S_global: 全局序列长度
            P: 每帧 token 数
            C: 嵌入维度
            global_idx: 当前 global block 索引
            pos: 位置编码 [B*S_global, P, 2]
            num_frame_for_scale: 尺度估计帧数
            sliding_window_size: 滑动窗口大小（blocks）
            num_frame_per_block: 每个block处理的帧数
            **kwargs: 其他参数
                - image_height: 图像高度
                - image_width: 图像宽度

        Returns:
            tuple: (tokens, global_idx, intermediates)
                - tokens: 处理后的 tokens
                - global_idx: 更新后的 block 索引
                - intermediates: 各 block 输出的列表
        """
        # 从 kwargs 提取图像尺寸（用于 3D RoPE）
        image_height = kwargs.get('image_height', self.img_size)
        image_width = kwargs.get('image_width', self.img_size)

        # 调用核心函数执行因果注意力
        return self._process_causal_stream(
            tokens, B, S_local, S_global, P, C, global_idx, pos,
            num_frame_per_block, sliding_window_size, num_frame_for_scale,
            image_height=image_height, image_width=image_width
        )

    def _process_causal_stream(
        self,
        tokens: torch.Tensor,
        B: int,
        S_local: int,
        S_global: int,
        P: int,
        C: int,
        global_idx: int,
        pos: Optional[torch.Tensor] = None,
        num_frame_per_block: int = 1,
        sliding_window_size: Optional[int] = None,
        num_frame_for_scale: Optional[int] = None,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
    ):
        """
        【核心函数】因果注意力流式推理 - 使用 FlashInfer KV Cache

        本方法是 AggregatorStream 的核心，实现了:
        1. 跨帧因果注意力计算
        2. KV Cache 的读取、存储和驱逐
        3. Scale Token 的双向注意力控制
        4. 3D RoPE 的位置编码（可选）

        ╔════════════════════════════════════════════════════════════════════╗
        ║             _process_causal_stream 完整处理流程                     ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【输入】                                                          ║
        ║                                                                    ║
        ║  tokens: [B*S_local, P, C]                                         ║
        ║  │   - B*S_local: 扁平化的 batch * local序列                      ║
        ║  │   - P: 每帧 token 数 (≈1005)                                   ║
        ║  │   - C: 嵌入维度 (1024)                                          ║
        ║                                                                    ║
        ║  其他参数:                                                          ║
        ║  │   B: batch大小                                                   ║
        ║  │   S_local: 本地序列长度                                          ║
        ║  │   S_global: 全局序列长度                                         ║
        ║  │   P: 每帧token数                                                 ║
        ║  │   C: 嵌入维度                                                    ║
        ║  │   global_idx: 当前 global block 索引                            ║
        ║  │   pos: 位置编码 [B*S_global, P, 2]                              ║
        ║  │   num_frame_per_block: 每block处理的帧数                        ║
        ║  │   num_frame_for_scale: 尺度估计帧数                             ║
        ║  │   image_height/width: 图像尺寸（用于3D RoPE）                  ║
        ║                                                                    ║
        ║  【处理流程】                                                      ║
        ║                                                                    ║
        ║  步骤1: 获取有效参数                                                ║
        ║  │   scale_frames = num_frame_for_scale or self.num_frame_for_scale║
        ║  │   确定哪些帧是 scale frames                                     ║
        ║                                                                    ║
        ║  步骤2: Reshape tokens                                              ║
        ║  │   [B*S_local, P, C] -> [B, S_local*P, C]                        ║
        ║  │   将帧合并为跨帧形式，方便全局注意力计算                         ║
        ║  │                                                                  ║
        ║  │   reshape过程:                                                   ║
        ║  │   │   tokens.view(B, S_local, P, C)  # 分离帧                   ║
        ║  │   │       .view(B, S_local*P, C)    # 合并为序列                 ║
        ║  │                                                                  │
        ║  │   示例 (B=1, S=8, P=1005, C=1024):                               ║
        ║  │   │   输入: [8, 1005, 1024]                                      ║
        ║  │   │   输出: [1, 8*1005, 1024] = [1, 8040, 1024]                 ║
        ║                                                                    ║
        ║  步骤3: 计算帧数和patch数                                           ║
        ║  │   num_frames = S_global                                          ║
        ║  │   num_patches = P - num_special_tokens                          ║
        ║  │   用于注意力掩码的计算                                           ║
        ║                                                                    ║
        ║  步骤4: 判断是否为第一个 block group                                ║
        ║  │   is_first_block_group = (global_idx < aa_block_size)          │
        ║  │   第一个 block group 需要初始化 3D RoPE                         ║
        ║                                                                    ║
        ║  步骤5: 处理位置编码                                                ║
        ║  │                                                                  ║
        ║  │   【情况A: 启用 3D RoPE】                                        ║
        ║  │   │   if enable_3d_rope and rope3d is not None:                │
        ║  │   │       if is_first_block_group:                             │
        ║  │   │           # 计算全局帧索引                                   │
        ║  │   │           f_start = total_frames_processed                 │
        ║  │   │           f_end = total_frames_processed + S_global        │
        ║  │   │           # 生成 3D 位置                                    │
        ║  │   │           pos3d = _get_3d_positions_streaming(             │
        ║  │   │               S_global, H, W, device, f_start, f_end        │
        ║  │   │           )                                                 │
        ║  │   │           cached_pos3d = pos3d                             │
        ║  │   │       else:                                                 │
        ║  │   │           # 使用缓存的位置                                  │
        ║  │   │           pos3d = cached_pos3d                             │
        ║  │   │       pos = pos3d                                           │
        ║  │                                                                  │
        ║  │   【情况B: 使用 2D RoPE】                                        │
        ║  │   │   else:                                                      │
        ║  │   │       # Reshape pos: [B*S_global, P, 2] -> [B, S_global*P, 2]║
        ║  │   │       pos = pos.view(B, S_global, P, 2)                     │
        ║  │   │            .view(B, S_global*P, 2)                          │
        ║  │                                                                  │
        ║  │   【3D RoPE vs 2D RoPE】                                        ║
        ║  │   ┌─────────────────────────────────────────────────────────────┐║
        ║  │   │ 2D RoPE:                                                    │║
        ║  │   │ - 只编码空间位置 (row, col)                                  │║
        ║  │   │ - 每帧的 patch 有相同的位置编码                              │║
        ║  │   │ - 不同帧的相同位置 patch 有相同的 RoPE                      │║
        ║  │   │                                                              │║
        ║  │   │ 3D RoPE:                                                    │║
        ║  │   │ - 编码时间 + 空间位置 (frame, row, col)                     │║
        ║  │   │ - 不同帧的 patch 有不同的位置编码                           │║
        ║  │   │ - 提高流式推理的时间一致性                                   │║
        ║  │   │ - 适合长序列推理                                              │║
        ║  │   └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  步骤6: 处理 global_blocks (带 KV Cache)                           ║
        ║  │                                                                  ║
        ║  │   intermediates = []                                             │║
        ║  │                                                                  ║
        ║  │   for _ in range(aa_block_size):  # 通常为1                     │║
        ║  │       num_patches = P - num_special_tokens                      │║
        ║  │                                                                  │║
        ║  │       【Backend A: SDPA (dict-based KV cache)】                 │║
        ║  │       │   if use_sdpa:                                           │║
        ║  │       │       tokens = global_blocks[global_idx](               │║
        ║  │       │           tokens,                                        │║
        ║  │       │           pos=pos,                                       │║
        ║  │       │           kv_cache=self.kv_cache,  # dict               │║
        ║  │       │           num_frames=num_frames,                        │║
        ║  │       │           num_frame_for_scale=scale_frames,             │║
        ║  │       │           ...                                             │║
        ║  │       │       )                                                   │║
        ║  │                                                                  │║
        ║  │       【Backend B: FlashInfer (paged KV cache)】                │║
        ║  │       │   else:                                                  │║
        ║  │       │       # 获取 FlashInfer manager (懒初始化)              │║
        ║  │       │       manager = _get_flashinfer_manager(                │║
        ║  │       │           device, dtype, tokens_per_frame=P             │║
        ║  │       │       )                                                   │║
        ║  │       │       tokens = global_blocks[global_idx](               │║
        ║  │       │           tokens,                                        │║
        ║  │       │           pos=pos,                                       │║
        ║  │       │           kv_cache=manager,  # FlashInferKVCacheManager │║
        ║  │       │           num_frames=num_frames,                        │║
        ║  │       │           num_frame_for_scale=scale_frames,             │║
        ║  │       │           ...                                             │║
        ║  │       │       )                                                   │║
        ║  │                                                                  │║
        ║  │       【global_block 内部处理】                                 │║
        ║  │       ┌─────────────────────────────────────────────────────────┐║║
        ║  │       │ FlashInferBlock/SDPABlock 处理流程:                      │║║
        ║  │       │                                                           │║║
        ║  │       │ 1. LayerNorm                                               │║║
        ║  │       │    tokens_norm = LayerNorm(tokens)                        │║║
        ║  │       │                                                           │║║
        ║  │       │ 2. 计算 Q, K, V                                            │║║
        ║  │       │    Q, K, V = Linear(tokens_norm)                          │║║
        ║  │       │    Q: [B, S_local*P, num_heads, head_dim]                 │║║
        ║  │       │    K, V: 同样形状                                          │║║
        ║  │       │                                                           │║║
        ║  │       │ 3. 应用 RoPE                                               │║║
        ║  │       │    Q = apply_rotary_emb(Q, pos)                           │║║
        ║  │       │    K = apply_rotary_emb(K, pos)                           │║║
        ║  │       │                                                           │║║
        ║  │       │ 4. 构建注意力掩码                                          │║║
        ║  │       │    causal_mask = 构建因果掩码                             │║║
        ║  │       │                                                           │║║
        ║  │       │    【因果掩码示例】                                        │║║
        ║  │       │    ┌─────────────────────────────────────────────────────┐║║║
        ║  │       │    │ 对于帧 0,1,2,3,4,5,6,7 (scale frames):              │║║║
        ║  │       │    │      全双向: 所有帧互相可见                          │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │ 对于帧 8,9,10,... (后续帧):                             │║║║
        ║  │       │    │      因果: Frame 8 只看 0-8                            │║║║
        ║  │       │    │           Frame 9 只看 0-9                            │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │ 掩码矩阵 (假设 scale_frames=3, 当前处理帧4):        │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │ Frame:    0    1    2    3    4                       │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │ Patch0(f0): ✓   ✓   ✓   ✓   ✓   <- 可见所有          │║║║
        ║  │       │    │ Patch0(f1): ✓   ✓   ✓   ✓   ✓                         │║║║
        ║  │       │    │ Patch0(f2): ✓   ✓   ✓   ✓   ✓   <- scale frames     │║║║
        ║  │       │    │ Patch0(f3): ✓   ✓   ✓   ✓   ✓   <- 因果，看0-3      │║║║
        ║  │       │    │ Patch0(f4): ✓   ✓   ✓   ✓   ✓   <- 当前帧           │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │ Scale frames 间的注意力是双向的                        │║║║
        ║  │       │    │ 非当前帧的 patches 之间也是因果的                      │║║║
        ║  │       │    └─────────────────────────────────────────────────────┘║║║
        ║  │       │                                                           │║║
        ║  │       │ 5. 获取 KV Cache                                           │║║
        ║  │       │    K_cached, V_cached = kv_cache.get(global_idx)          │║║
        ║  │       │                                                           │║║
        ║  │       │    【KV Cache 结构】                                      │║║
        ║  │       │    ┌─────────────────────────────────────────────────────┐║║║
        ║  │       │    │ FlashInfer (paged cache):                           │║║║
        ║  │       │    │   manager: FlashInferKVCacheManager                 │║║║
        ║  │       │    │   - 分页管理 KV                                       │║║║
        ║  │       │    │   - 高效的内存分配                                    │║║║
        ║  │       │    │   - 支持变长序列                                      │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │ SDPA (dict cache):                                     │║║║
        ║  │       │    │   kv_cache = {                                          │║║║
        ║  │       │    │     "k_0": [B, 1, S_cached, P, C],                   │║║║
        ║  │       │    │     "v_0": [B, 1, S_cached, P, C],                   │║║║
        ║  │       │    │     "k_0_special": [B, 1, S_cached, num_spec, C],   │║║║
        ║  │       │    │     "v_0_special": ...,                                │║║║
        ║  │       │    │     ...                                                   │║║║
        ║  │       │    │   }                                                       │║║║
        ║  │       │    │                                                           │║║║
        ║  │       │    │   S_cached = 已缓存的帧数                               │║║║
        ║  │       │    │   包含 scale frames + 滑动窗口内的帧                   │║║║
        ║  │       │    └─────────────────────────────────────────────────────┘║║║
        ║  │       │                                                           │║║
        ║  │       │ 6. 拼接当前 K/V 和缓存 K/V                                 │║║
        ║  │       │    if has_cache:                                           │║║
        ║  │       │        K_full = concat([K_cached, K_cur])                  │║║
        ║  │       │        V_full = concat([V_cached, V_cur])                  │║║
        ║  │       │    else:                                                   │║║
        ║  │       │        K_full, V_full = K_cur, V_cur                       │║║
        ║  │       │                                                           │║║
        ║  │       │ 7. 计算注意力                                               │║║
        ║  │       │    # 对于当前帧的 patches                                  │║║
        ║  │       │    Q_cur = Q[:, -S_local*P:]  # 当前帧的 Query             │║║
        ║  │       │                                                           │║║
        ║  │       │    Attention = softmax(                                   │║║
        ║  │       │        Q_cur @ K_full^T / sqrt(head_dim)                   │║║
        ║  │       │    ) @ V_full                                              │║║
        ║  │       │                                                           │║║
        ║  │       │    Attention shape:                                        │║║
        ║  │       │    [B, S_local*P, num_heads, head_dim]                     │║║
        ║  │       │                                                           │║║
        ║  │       │ 8. 输出投影                                                 │║║
        ║  │       │    Attn_out = Linear(Attention) + tokens  # 残差          │║║
        ║  │       │                                                           │║║
        ║  │       │ 9. FFN                                                      │║║
        ║  │       │    FFN_out = MLP(LayerNorm(Attn_out)) + Attn_out          │║║
        ║  │       │                                                           │║║
        ║  │       │ 10. 存储当前帧 K/V 到 cache                                 │║║
        ║  │       │    if not skip_append:                                     │║║
        ║  │       │        kv_cache.append(global_idx, K_cur, V_cur)          │║║
        ║  │       │                                                           │║║
        ║  │       │ 11. 驱逐旧帧 (如果超过 sliding_window)                    │║║
        ║  │       │    if num_cached > sliding_window:                        │║║
        ║  │       │        # 驱逐最旧帧                                         │║║
        ║  │       │        # 但保留:                                            │║║
        ║  │       │        #   - scale frames                                   │║║
        ║  │       │        #   - special tokens from all frames                │║║
        ║  │       │        kv_cache.evict_oldest(                             │║║
        ║  │       │            preserve_scale=True,                            │║║
        ║  │       │            preserve_special=True                           │║║
        ║  │       │        )                                                    │║║
        ║  │       └─────────────────────────────────────────────────────────┘║║
        ║  │                                                                  │║
        ║  │       global_idx += 1                                             │║
        ║  │       intermediates.append(tokens.view(B, S_local, P, C))         │║
        ║                                                                    ║
        ║  步骤7: 更新 total_frames_processed                                 ║
        ║  │   if is_first_block_group and not skip_append:                  │
        ║  │       total_frames_processed += S_global                        │
        ║  │   记录已处理的帧数，用于 3D RoPE 的全局帧索引                   │
        ║                                                                    ║
        ║  【输出】                                                          ║
        ║                                                                    ║
        ║  tokens: [B, S_local*P, C]                                          ║
        ║  │   处理后的 tokens                                                ║
        ║                                                                    ║
        ║  global_idx: int                                                    ║
        ║  │   更新后的 block 索引                                            ║
        ║                                                                    ║
        ║  intermediates: List[[B, S_local, P, C]]                            ║
        ║  │   各 block 的输出                                                ║
        ║                                                                    ║
        ║  【关键帧模式 (keyframe_interval > 1)】                             ║
        ║                                                                    ║
        ║  当启用关键帧模式时:                                                ║
        ║  │   - _skip_append 标志控制是否存储 KV                            │
        ║  │   - 关键帧: 存储 KV 到 cache                                      │
        ║  │   - 非关键帧:                                                    │
        ║  │   │     1. 设置 _skip_append = True                              │
        ║  │   │     2. 执行 forward (参与注意力计算)                         │
        ║  │   │     3. 重置 _skip_append = False                             │
        ║  │   │     4. KV 不被存储                                           │
        ║  │   │                                                              │
        ║  │   效果:                                                           │
        ║  │   │   - 内存占用减少约 1/keyframe_interval                       │
        ║  │   │   - 所有帧都产生完整预测                                     │
        ║  │   │   - 适合长序列推理                                            │
        ║                                                                    ║
        ║  【Flow-based 关键帧模式】                                         ║
        ║                                                                    ║
        ║  当 flow_threshold > 0 时:                                         │
        ║  │   - 使用 defer_eviction + rollback 机制                         │
        ║  │   - 延迟驱逐: 处理时不驱逐旧帧                                   │
        ║  │   - 计算光流: 估计当前帧与上次关键帧的运动                       │
        ║  │   - 决策:                                                        │
        ║  │   │     如果 flow > threshold 或 gap > max_gap:                 │
        ║  │   │       执行驱逐，确认当前帧为关键帧                           │
        ║  │   │     否则:                                                    │
        ║  │   │       rollback，撤销当前帧的 KV 存储                         │
        ║  │   │                                                              │
        ║  │   自适应关键帧选择，更好的运动估计                               │
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            tokens: 输入 tokens [B*S_local, P, C]
            B: Batch size
            S_local: 本地序列长度（当前处理的帧数）
            S_global: 全局序列长度
            P: 每帧 token 数（包含特殊 tokens）
            C: 嵌入维度
            global_idx: 当前 global block 索引
            pos: 位置编码 [B*S_global, P, 2]（可选）
            num_frame_per_block: 每个block处理的帧数
                - 1: 逐帧处理（流式）
                - N: batch处理（scale frames）
            sliding_window_size: 滑动窗口大小（blocks）
            num_frame_for_scale: 尺度估计帧数
                - 前 N 帧使用双向注意力
            image_height: 图像高度（用于 3D RoPE）
            image_width: 图像宽度（用于 3D RoPE）

        Returns:
            tuple: (tokens, global_idx, intermediates)
                - tokens: 处理后的 tokens [B, S_local*P, C]
                - global_idx: 更新后的 global block 索引
                - intermediates: 各 block 输出的列表 [B, S_local, P, C]
        """
        # ════════════════════════════════════════════════════════════════════
        # 步骤1: 获取有效的尺度帧数
        # ════════════════════════════════════════════════════════════════════
        scale_frames = num_frame_for_scale if num_frame_for_scale is not None else self.num_frame_for_scale

        # ════════════════════════════════════════════════════════════════════
        # 步骤2: Reshape tokens 为跨帧形式
        # ════════════════════════════════════════════════════════════════════
        # 从 [B*S_local, P, C] 转换为 [B, S_local*P, C]
        # 这是跨帧注意力的标准输入形式
        if tokens.shape != (B, S_local * P, C):
            tokens = tokens.view(B, S_local, P, C).view(B, S_local * P, C)

        # ════════════════════════════════════════════════════════════════════
        # 步骤3: 计算帧数和patch数
        # ════════════════════════════════════════════════════════════════════
        num_frames = S_global  # 总帧数
        num_patches = P - self.num_special_tokens  # patch tokens 数量

        # ════════════════════════════════════════════════════════════════════
        # 步骤4: 判断是否为第一个 block group
        # ════════════════════════════════════════════════════════════════════
        # 第一个 block group 需要初始化 3D RoPE 的位置编码
        is_first_block_group = (global_idx < self.aa_block_size)

        # ════════════════════════════════════════════════════════════════════
        # 步骤5: 处理位置编码
        # ════════════════════════════════════════════════════════════════════
        if self.enable_3d_rope and self.rope3d is not None:
            # 【使用 3D RoPE】编码时间 + 空间位置
            if is_first_block_group:
                # 计算全局帧索引范围
                f_start = self.total_frames_processed
                f_end = self.total_frames_processed + S_global

                # 获取图像尺寸
                H = image_height if image_height is not None else self.img_size
                W = image_width if image_width is not None else self.img_size

                # 生成 3D 位置编码
                pos3d = self._get_3d_positions_streaming(
                    S_global, H, W, tokens.device, f_start, f_end
                )
                # 缓存位置编码，后续 block group 使用
                self._cached_pos3d = pos3d
            else:
                # 使用缓存的位置编码
                pos3d = self._cached_pos3d
            pos = pos3d
        else:
            # 【使用 2D RoPE】只编码空间位置
            # Reshape pos: [B*S_global, P, 2] -> [B, S_global*P, 2]
            if pos is not None and pos.shape != (B, S_global * P, 2):
                pos = pos.view(B, S_global, P, 2).view(B, S_global * P, 2)

        # ════════════════════════════════════════════════════════════════════
        # 步骤6: 处理 global_blocks
        # ════════════════════════════════════════════════════════════════════
        effective_sliding_window_size = (
            self.sliding_window_size if sliding_window_size is None else sliding_window_size
        )
        # Stage1/global-attention mode processes the whole view set at once.
        # In that case SDPA does not need the streaming KV cache; avoiding it
        # saves the extra K/V graph references that make backbone training OOM.
        use_full_batch_sdpa = (
            self.use_sdpa
            and effective_sliding_window_size == -1
            and num_frame_per_block == S_global
            and S_local == S_global
        )
        intermediates = []

        for _ in range(self.aa_block_size):
            num_patches = P - self.num_special_tokens

            if self.use_sdpa:
                # 【Backend A: SDPA】global full-batch 用普通 SDPA；streaming 才使用 dict cache
                kv_cache = None if use_full_batch_sdpa else self.kv_cache
                block = self.global_blocks[global_idx]
                block_global_idx = global_idx

                def _run_sdpa_global_block(x):
                    return block(
                        x,
                        pos=pos,
                        enable_ulysses_cp=False,  # Context Parallelism 已禁用
                        num_patches=num_patches,
                        num_special=self.num_special_tokens,
                        num_frames=num_frames,
                        enable_3d_rope=self.enable_3d_rope,
                        kv_cache=kv_cache,
                        global_idx=block_global_idx,
                        num_frame_per_block=num_frame_per_block,
                        num_frame_for_scale=scale_frames,
                        num_register_tokens=self.num_register_tokens,
                        num_anchor_tokens=self.num_anchor_tokens,
                        num_scale_tokens=self.num_scale_tokens,
                        sliding_window_size=effective_sliding_window_size,
                    )

                if (
                    use_full_batch_sdpa
                    and self.training
                    and self.use_gradient_checkpoint
                    and tokens.requires_grad
                ):
                    from torch.utils.checkpoint import checkpoint
                    tokens = checkpoint(
                        _run_sdpa_global_block,
                        tokens,
                        use_reentrant=self.use_reentrant,
                    )
                else:
                    tokens = _run_sdpa_global_block(tokens)
            else:
                # 【Backend B: FlashInfer】使用分页 KV cache
                # 懒初始化 FlashInfer manager（首次使用时创建）
                manager = self._get_flashinfer_manager(tokens.device, tokens.dtype, tokens_per_frame=P)
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    enable_ulysses_cp=False,
                    num_patches=num_patches,
                    num_special=self.num_special_tokens,
                    num_frames=num_frames,
                    enable_3d_rope=self.enable_3d_rope,
                    kv_cache=manager,  # FlashInferKVCacheManager
                    global_idx=global_idx,
                    num_frame_per_block=num_frame_per_block,
                    num_frame_for_scale=scale_frames,
                    num_register_tokens=self.num_register_tokens,
                    num_anchor_tokens=self.num_anchor_tokens,
                    num_scale_tokens=self.num_scale_tokens,
                    sliding_window_size=effective_sliding_window_size,
                )

            # 更新 block 索引
            global_idx += 1
            # 收集中间输出（reshape 回 [B, S_local, P, C]）
            intermediates.append(tokens.view(B, S_local, P, C))

        # ════════════════════════════════════════════════════════════════════
        # 步骤7: 更新已处理帧数计数器
        # ════════════════════════════════════════════════════════════════════
        # 只在第一个 block group 且非 skip_append 模式时更新
        if is_first_block_group and not (isinstance(self.kv_cache, dict) and self.kv_cache.get("_skip_append", False)):
            self.total_frames_processed += S_global

        return tokens, global_idx, intermediates
