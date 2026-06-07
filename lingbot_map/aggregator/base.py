"""
AggregatorBase - 所有 Aggregator 实现的基类

提供共享功能:
- Patch Embedding (DINOv2)
- 特殊 Tokens (camera, register, scale)
- Block 构建
- 通用的前向传播结构

子类实现模式特定的注意力逻辑。

【架构概览】

Aggregator 是 GCT 模型的核心组件，负责:
1. 图像特征提取: 使用 DINOv2 ViT-L/14 进行 Patch Embedding
2. 特殊 Token 管理: 添加 camera、register、scale tokens
3. 注意力处理:
   - frame_blocks: 帧内自注意力（每帧独立处理）
   - global_blocks: 跨帧注意力（子类实现具体逻辑）

【forward() 执行流程】

┌─────────────────────────────────────────────────────────────────────┐
│                      Aggregator.forward() 流程                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  images [B, S, 3, H, W]                                             │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ _embed_images()                                                │ │
│  │                                                                 │ │
│  │  1. 图像归一化: ResNet mean/std                                 │ │
│  │  2. Reshape: [B,S,3,H,W] -> [B*S,3,H,W]                        │ │
│  │  3. Patch Embedding: DINOv2 ViT-L/14                           │ │
│  │     - patch_size=14                                             │ │
│  │     - 输出: patch_tokens [B*S, P_patch, C]                     │ │
│  │  4. 添加 Special Tokens:                                        │ │
│  │     [camera][register(4)][scale] + patch_tokens                │ │
│  │     -> tokens [B*S, P, C]                                       │ │
│  │                                                                 │ │
│  │  返回: tokens, B, S_local, S_global, P, C                       │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ _get_positions() - 生成2D位置编码                              │ │
│  │                                                                 │ │
│  │  为每个 token 生成位置信息（用于 RoPE）                         │ │
│  │  - pos_local: [B*S, P, 2]                                       │ │
│  │  - pos_global: [B*S, P, 2]                                      │ │
│  │                                                                 │ │
│  │  RoPE (Rotary Position Embedding):                             │ │
│  │  - 将位置信息编码到注意力计算中                                 │ │
│  │  - 支持相对位置信息，提高长序列建模效果                         │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ 交替执行 frame_blocks 和 global_blocks                         │ │
│  │                                                                 │ │
│  │  aa_order = ["frame", "global"]                                 │ │
│  │  aa_block_num = depth / aa_block_size = 24                     │ │
│  │                                                                 │ │
│  │  for block_group_idx in range(24):                             │ │
│  │      ┌───────────────────────────────────────────────────────┐ │ │
│  │      │ _process_frame_attention()                             │ │ │
│  │      │                                                         │ │ │
│  │      │  帧内自注意力：每帧独立处理                              │ │ │
│  │      │  - tokens: [B*S, P, C] -> [B*S, P, C]                  │ │ │
│  │      │  - 使用 RoPE 位置编码                                   │ │ │
│  │      │  - 无跨帧通信                                           │ │ │
│  │      │                                                         │ │ │
│  │      │  计算过程:                                               │ │ │
│  │      │  Q = tokens @ W_q                                       │ │ │
│  │      │  K = tokens @ W_k                                       │ │ │
│  │      │  V = tokens @ W_v                                       │ │ │
│  │      │  Attention(Q, K, V) with RoPE                          │ │ │
│  │      │  -> LayerNorm -> FFN -> LayerNorm                       │ │ │
│  │      │                                                         │ │ │
│  │      │  训练时可选 gradient checkpoint 节省内存                 │ │ │
│  │      └───────────────────────────────────────────────────────┘ │ │
│  │      ┌───────────────────────────────────────────────────────┐ │ │
│  │      │ _process_global_attention()                            │ │ │
│  │      │                                                         │ │ │
│  │      │  跨帧注意力：子类实现具体逻辑                            │ │ │
│  │      │  - GCTStream 使用 AggregatorStream._process_causal_   │ │ │
│  │      │    stream() 实现因果注意力                              │ │ │
│  │      │  - 支持 KV cache 的流式推理                             │ │ │
│  │      │  - scale token 控制前N帧双向注意力                      │ │ │
│  │      │                                                         │ │ │
│  │      │  tokens reshape: [B*S, P, C] -> [B, S*P, C]            │ │ │
│  │      │                                                         │ │ │
│  │      │  因果注意力（流式推理）:                                  │ │ │
│  │      │  - 帧 t 只能看到帧 0~t                                  │ │ │
│  │      │  - 使用 KV cache 存储历史帧的 Key/Value                 │ │ │
│  │      │  - 新帧只需与 cache 计算注意力，无需重新编码历史         │ │ │
│  │      └───────────────────────────────────────────────────────┘ │ │
│  │                                                                 │ │
│  │      收集输出:                                                   │ │
│  │      if block_group_idx in selected_idx:                       │ │
│  │          concat = cat([frame_out, global_out], dim=-1)         │ │
│  │          output_list.append(concat)  # [B, S, P, 2C]           │ │
│  │                                                                 │ │
│  └───────────────────────────────────────────────────────────────┘ │
│       │                                                             │
│       ↓                                                             │
│  返回: (output_list, patch_start_idx)                               │
│  - output_list: List[[B, S, P, 2C]]                                 │
│  - patch_start_idx: 6                                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

【特殊 Token 结构】

每帧的 token 序列:
┌────────────────────────────────────────────────────────────────────┐
│ Index:    0      1-4      5      6...end                           │
│ Token: camera register scale  patches                              │
│          ↓       ↓       ↓       ↓                                  │
│ 用途: 位姿预测 特征增强 尺度控制 图像特征                           │
│                                                                    │
│ patch_start_idx = 1 + num_register_tokens + 1 = 6                 │
│ num_special_tokens = 1 + 4 + 1 = 6                                │
└────────────────────────────────────────────────────────────────────┘

【DINOv2 预训练权重初始化】

frame_blocks 和 global_blocks 可以从 DINOv2 权重初始化:
- 提高特征质量
- 加速训练收敛
- 使用 DINO 的 register token 机制
"""

import logging
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List

from lingbot_map.layers import PatchEmbed
from lingbot_map.layers.block import Block
from lingbot_map.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from lingbot_map.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

# ResNet 图像归一化参数（用于 DINOv2 输入）
_RESNET_MEAN = [0.485, 0.456, 0.406]  # RGB 通道均值
_RESNET_STD = [0.229, 0.224, 0.225]   # RGB 通道标准差


def slice_expand_and_flatten(token, B, S, first_num_frame=1):
    """
    【辅助函数】切片、展开并扁平化特殊 Token

    用于处理 camera_token 和 scale_token 的不同激活状态:
    - camera_token/scale_token 有两个状态 [1, 2, N, C]
      - index 0: 用于第一帧（或 scale frames）
      - index 1: 用于后续帧

    处理逻辑:
    ┌────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  输入 token: [1, 2, N, C]                                          │
    │  - N: token 数量（camera=1, register=num_reg, scale=1）           │
    │  - C: 嵌入维度                                                     │
    │                                                                     │
    │  如果 first_num_frame > 1:                                         │
    │  │   token_first = token[:, :1]  -> 用于前 first_num_frame 帧    │
    │  │   token_rest = token[:, 1:]   -> 用于剩余帧                   │
    │  │   token_expanded = cat([token_first, token_rest])             │
    │  │   -> [B, S, N, C]                                              │
    │                                                                     │
    │  如果 first_num_frame == 1:                                        │
    │  │   token_first = token[:, :1]  -> 用于第1帧                    │
    │  │   token_rest = token[:, 1:]   -> 用于第2~S帧                  │
    │  │   token_expanded = cat([token_first, token_rest])             │
    │  │   -> [B, S, N, C]                                              │
    │                                                                     │
    │  输出: [B*S, N, C]  扁平化后用于注意力计算                         │
    │                                                                     │
    └────────────────────────────────────────────────────────────────────┘

    【Scale Token 应用示例】

    scale_token 的 first_num_frame = num_frame_for_scale (默认8):
    - 前8帧使用激活的 scale_token (index 0)
    - 后续帧使用非激活的 scale_token (index 1)
    - 这使得前8帧之间可以双向通信（建立尺度基准）

    Args:
        token: 特殊 token [1, 2, N, C]
        B: Batch size
        S: 序列长度（帧数）
        first_num_frame: 使用第一个状态的前N帧数量

    Returns:
        扁平化的 token [B*S, N, C]
    """
    # token shape: [1, 2, N, C]
    # 展开到 [B, S, N, C]
    if first_num_frame > 1:
        # 使用第一个 token 状态用于前 first_num_frame 帧
        token_first = token[:, :1].expand(B, first_num_frame, -1, -1)  # [B, first_num_frame, N, C]
        # 使用第二个 token 状态用于剩余帧
        token_rest = token[:, 1:].expand(B, S - first_num_frame, -1, -1)  # [B, S-first_num_frame, N, C]
        # 拼接
        token_expanded = torch.cat([token_first, token_rest], dim=1)  # [B, S, N, C]
    else:
        # 使用第一个 token 状态用于第1帧
        token_first = token[:, :1].expand(B, 1, -1, -1)  # [B, 1, N, C]
        # 使用第二个 token 状态用于剩余帧
        token_rest = token[:, 1:].expand(B, S - 1, -1, -1)  # [B, S-1, N, C]
        # 拼接
        token_expanded = torch.cat([token_first, token_rest], dim=1)  # [B, S, N, C]

    # 扁平化到 [B*S, N, C] 用于注意力计算
    return token_expanded.reshape(B * S, -1, token.shape[-1])


class AggregatorBase(nn.Module, ABC):
    """
    所有 Aggregator 实现的基类

    处理共享组件:
    - Patch Embedding (DINOv2 或 conv)
    - 特殊 Tokens (camera, register, scale)
    - Block 创建 (frame + global)
    - RoPE (2D 旋转位置嵌入)
    - 通用的前向传播框架

    子类必须实现:
    - _process_global_attention(): 模式特定的跨帧注意力逻辑

    【设计模式】

    本类使用抽象基类模式，定义了 Aggregator 的核心架构:
    - frame_blocks: 帧内自注意力（基类实现）
    - global_blocks: 跨帧注意力（子类实现）
    - 特殊 token 管理（子类实现）

    【参数说明】

    Architecture parameters:
    - img_size: 输入图像尺寸，默认518
    - patch_size: Patch大小，默认14（DINOv2标准）
    - embed_dim: 嵌入维度，默认1024（ViT-L）
    - depth: Transformer深度，默认24层
    - num_heads: 注意力头数，默认16
    - mlp_ratio: MLP扩展比例，默认4.0
    - num_register_tokens: Register token数量，默认4

    Block configuration:
    - block_fn: Block类型，默认 Block
    - qkv_bias: QKV是否有bias
    - proj_bias: 投影是否有bias
    - ffn_bias: FFN是否有bias
    - qk_norm: 是否使用QK归一化
    - init_values: LayerScale初始值

    Patch embedding:
    - patch_embed: Patch embedding类型
      - "dinov2_vitl14_reg": DINOv2 ViT-L/14 with registers
      - "conv": 简单卷积
    - pretrained_path: DINOv2预训练权重路径

    Attention pattern:
    - aa_order: 注意力顺序 ["frame", "global"]
    - aa_block_size: 每组block数量

    RoPE:
    - rope_freq: RoPE频率，默认100
    - disable_global_rope: 是否禁用全局注意力中的RoPE

    Gradient checkpointing:
    - use_reentrant: 是否使用reentrant模式
    - use_gradient_checkpoint: 是否启用梯度检查点
    """

    def __init__(
        self,
        # Architecture parameters (架构参数)
        img_size=518,        # 输入图像尺寸
        patch_size=14,       # Patch大小
        embed_dim=1024,      # 嵌入维度
        depth=24,            # Transformer深度
        num_heads=16,        # 注意力头数
        mlp_ratio=4.0,       # MLP扩展比例
        num_register_tokens=4,  # Register token数量
        # Block configuration (Block配置)
        block_fn=Block,      # Block函数
        qkv_bias=True,       # QKV bias
        proj_bias=True,      # 投影 bias
        ffn_bias=True,       # FFN bias
        qk_norm=True,        # QK归一化
        init_values=0.01,    # LayerScale初始值
        # Patch embedding (Patch嵌入)
        patch_embed="dinov2_vitl14_reg",  # Patch embedding类型
        pretrained_path=None,  # 预训练权重路径
        # Attention pattern (注意力模式)
        aa_order=["frame", "global"],  # 注意力顺序
        aa_block_size=1,     # 每组block数
        # RoPE (旋转位置编码)
        rope_freq=100,       # RoPE频率
        disable_global_rope=False,  # 是否禁用全局RoPE
        # Gradient checkpointing (梯度检查点)
        use_reentrant: bool = False,
        use_gradient_checkpoint: bool = True,
    ):
        super().__init__()

        # ════════════════════════════════════════════════════════════════════
        # 存储配置参数
        # ════════════════════════════════════════════════════════════════════
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_register_tokens = num_register_tokens
        self.aa_order = aa_order  # 注意力交替顺序
        self.aa_block_size = aa_block_size  # 每组block数
        self.disable_global_rope = disable_global_rope
        self.use_reentrant = use_reentrant
        self.use_gradient_checkpoint = use_gradient_checkpoint
        self.pretrained_path = pretrained_path
        self.enable_ulysses_cp = False  # Context Parallelism 已禁用

        print("pretrained_path:", self.pretrained_path)

        # 验证 depth 必须能被 aa_block_size 整除
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")
        self.aa_block_num = self.depth // self.aa_block_size  # block组数

        # ════════════════════════════════════════════════════════════════════
        # 构建 Patch Embedding (DINOv2 或 Conv)
        # ════════════════════════════════════════════════════════════════════
        self._build_patch_embed(
            patch_embed=patch_embed,
            img_size=img_size,
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            embed_dim=embed_dim,
            pretrained_path=pretrained_path
        )

        # ════════════════════════════════════════════════════════════════════
        # 初始化 RoPE (2D 旋转位置编码)
        # ════════════════════════════════════════════════════════════════════
        # RoPE 用于在注意力计算中编码相对位置信息
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        # ════════════════════════════════════════════════════════════════════
        # 构建 Blocks (frame_blocks + global_blocks)
        # ════════════════════════════════════════════════════════════════════
        self._build_blocks(
            block_fn=block_fn,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            init_values=init_values,
            qk_norm=qk_norm,
        )

        # ════════════════════════════════════════════════════════════════════
        # 设置特殊 Tokens (camera, register, scale)
        # ════════════════════════════════════════════════════════════════════
        self._setup_special_tokens()

        # ════════════════════════════════════════════════════════════════════
        # 注册归一化常数为 buffer
        # ════════════════════════════════════════════════════════════════════
        # 这些是固定的常数，不参与训练
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        # ════════════════════════════════════════════════════════════════════
        # 从 DINO 预训练权重初始化 blocks（如果可用）
        # ════════════════════════════════════════════════════════════════════
        if hasattr(self, '_dino_checkpoint') and self._dino_checkpoint is not None:
            self._init_blocks_from_dino(self._dino_checkpoint)
            del self._dino_checkpoint  # 释放内存

    def _build_patch_embed(
        self,
        patch_embed: str,
        img_size: int,
        patch_size: int,
        num_register_tokens: int,
        embed_dim: int,
        pretrained_path: str,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
    ):
        """
        Build patch embedding layer.

        Supports:
        - "conv": Simple convolutional patch embedding
        - "dinov2_*": DINOv2 ViT variants (vitl14, vitb14, vits14, vitg2)
        """
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim
            )
            self._dino_checkpoint = None

        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            if patch_embed not in vit_models:
                raise NotImplementedError(f"Unknown patch_embed type: {patch_embed}")

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Load optional standalone DINOv2 weights. Full GCT checkpoints are
            # loaded later by callers and intentionally leave this path empty.
            if pretrained_path:
                try:
                    ckpt = torch.load(pretrained_path)
                    del ckpt['pos_embed']
                    logger.info("Loading pretrained weights for DINOv2")
                    missing, unexpected = self.patch_embed.load_state_dict(ckpt, strict=False)
                    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

                    # Store checkpoint for block initialization
                    self._dino_checkpoint = ckpt
                except Exception as e:
                    logger.warning(f"Failed to load pretrained weights: {e}")
                    self._dino_checkpoint = None
            else:
                self._dino_checkpoint = None

            # Disable gradients for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    @abstractmethod
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
        """
        Build frame_blocks and global_blocks.

        Subclasses implement mode-specific block creation.

        Must create:
        - self.frame_blocks: nn.ModuleList of frame attention blocks
        - self.global_blocks: nn.ModuleList of global attention blocks
        """
        pass

    @abstractmethod
    def _setup_special_tokens(self):
        """
        Setup camera token, register tokens, and optionally scale token.

        Subclasses implement mode-specific token initialization.

        Must create:
        - self.camera_token
        - self.register_token
        - self.scale_token (optional, for causal mode)
        - self.patch_start_idx
        - self.num_special_tokens
        """
        pass

    def _init_blocks_from_dino(self, dino_ckpt: dict):
        """
        Initialize frame_blocks and global_blocks from DINOv2 pretrained weights.

        Args:
            dino_ckpt: Checkpoint dictionary from DINOv2 model
        """
        logger.info("Initializing blocks from DINOv2 pretrained weights")

        # Extract block keys
        dino_block_keys = [k for k in dino_ckpt.keys() if k.startswith('blocks.')]
        if not dino_block_keys:
            logger.warning("No 'blocks' found in DINO checkpoint")
            return

        # Get block indices
        block_indices = set()
        for key in dino_block_keys:
            parts = key.split('.')
            if len(parts) > 1 and parts[1].isdigit():
                block_indices.add(int(parts[1]))

        num_dino_blocks = len(block_indices)
        print(f"Found {num_dino_blocks} blocks in DINO checkpoint")

        # Initialize frame_blocks
        for i, block in enumerate(self.frame_blocks):
            dino_block_idx = i % num_dino_blocks
            block_state_dict = {}
            prefix = f'blocks.{dino_block_idx}.'
            for key, value in dino_ckpt.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    block_state_dict[new_key] = value

            if block_state_dict:
                missing, unexpected = block.load_state_dict(block_state_dict, strict=False)
                if i == 0:  # Only log for first block to avoid spam
                    print(f"Frame block 0: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        # Initialize global_blocks
        for i, block in enumerate(self.global_blocks):
            dino_block_idx = i % num_dino_blocks
            block_state_dict = {}
            prefix = f'blocks.{dino_block_idx}.'
            for key, value in dino_ckpt.items():
                if key.startswith(prefix):
                    new_key = key[len(prefix):]
                    block_state_dict[new_key] = value

            if block_state_dict:
                missing, unexpected = block.load_state_dict(block_state_dict, strict=False)
                if i == 0:  # Only log for first block to avoid spam
                    print(f"Global block 0: Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        logger.info("Successfully initialized blocks from DINOv2 weights")

    def _embed_images(
        self,
        images: torch.Tensor,
        num_frame_for_scale: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int, int, int, int, int]:
        """
        【图像嵌入】处理图像并准备 tokens

        本方法将输入图像转换为 tokens 序列，包括:
        1. 图像归一化 (ResNet标准)
        2. DINOv2 Patch Embedding
        3. 添加特殊 tokens (camera, register, scale)
        4. 拼接形成完整的 token 序列

        ╔════════════════════════════════════════════════════════════════════╗
        ║                   _embed_images 处理流程                           ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【输入】                                                          ║
        ║  images: [B, S, 3, H, W]                                           ║
        ║  │   - B: batch大小                                                ║
        ║  │   - S: 序列长度（帧数）                                         ║
        ║  │   - 3: RGB通道                                                  ║
        ║  │   - H: 图像高度（默认518）                                      ║
        ║  │   - W: 图像宽度（默认378）                                      ║
        ║  │   - 值域: [0, 1]                                                ║
        ║                                                                    ║
        ║  【处理流程】                                                      ║
        ║                                                                    ║
        ║  步骤1: 验证输入通道                                                ║
        ║  │   if C_in != 3:                                                 ║
        ║  │       raise ValueError                                          ║
        ║                                                                    ║
        ║  步骤2: 图像归一化                                                  ║
        ║  │   使用 ResNet 标准归一化（DINOv2 的标准预处理）                 ║
        ║  │   images_norm = (images - mean) / std                           ║
        ║  │   mean = [0.485, 0.456, 0.406] (RGB)                            ║
        ║  │   std = [0.229, 0.224, 0.225] (RGB)                             ║
        ║  │                                                                  ║
        ║  │   归一化后:                                                      ║
        ║  │   - RGB通道独立归一化                                            ║
        ║  │   - 使输入分布接近预训练数据的分布                               ║
        ║  │   - 提高特征提取质量                                             ║
        ║                                                                    ║
        ║  步骤3: 设置序列长度                                                ║
        ║  │   S_local = S_global = S                                        ║
        ║  │   - 无 Context Parallelism 切片                                 ║
        ║  │   - 所有帧一起处理                                               ║
        ║                                                                    ║
        ║  步骤4: Reshape 为 patch embedding 输入                            ║
        ║  │   images_norm = images_norm.view(B * S, 3, H, W)                ║
        ║  │   - 将 batch 和 序列维度合并                                    ║
        ║  │   - 方便一次性处理所有帧                                         ║
        ║                                                                    ║
        ║  步骤5: DINOv2 Patch Embedding                                     ║
        ║  │   patch_tokens = self.patch_embed(images_norm)                  ║
        ║  │                                                                  ║
        ║  │   DINOv2 ViT-L/14 处理:                                          ║
        ║  │   ┌─────────────────────────────────────────────────────────────┐║
        ║  │   │ Patch 切分:                                                   │║
        ║  │   │ - patch_size = 14 像素                                        │║
        ║  │   │ - 每个 patch 是 14x14 的图像区域                              │║
        ║  │   │ - patch 数量 = (H/14) * (W/14)                               │║
        ║  │   │   例如: (518/14) * (378/14) ≈ 37 * 27 = 999 个 patch       │║
        ║  │   │                                                               │║
        ║  │   │ Patch Embedding:                                              │║
        ║  │   │ - 每个 patch 通过线性投影变成 C 维向量                        │║
        ║  │   │ - C = 1024 (ViT-L 的隐藏层维度)                               │║
        ║  │   │ - 加上 position embedding (DINO 的绝对位置编码)              │║
        ║  │   │                                                               │║
        ║  │   │ DINOv2 特点:                                                   │║
        ║  │   │ - Register tokens: 4个额外的 tokens 提高特征质量             │║
        ║  │   │ - 无 mask token (DINO 不使用 masking)                        │║
        ║  │   │ - 输出包含 patch_tokens 和 register_tokens                   │║
        ║  │   │                                                               │║
        ║  │   │ 输出:                                                          │║
        ║  │   │ patch_tokens: [B*S, P_patch, C]                              │║
        ║  │   │ - P_patch ≈ 999                                               │║
        ║  │   │ - C = 1024                                                    │║
        ║  │   └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  步骤6: 准备特殊 Tokens                                             ║
        ║  │   special_tokens = self._prepare_special_tokens(                ║
        ║  │       B, S_local, S_global, C,                                   ║
        ║  │       num_frame_for_scale=num_frame_for_scale                    ║
        ║  │   )                                                              ║
        ║  │                                                                  ║
        ║  │   特殊 tokens 结构:                                               ║
        ║  │   ┌─────────────────────────────────────────────────────────────┐║
        ║  │   │ Index:    0      1-4      5                                   │║
        ║  │   │ Token: camera register scale                                  │║
        ║  │   │ Size:   [B*S,1,C] [B*S,4,C] [B*S,1,C]                        │║
        ║  │   │                                                               │║
        ║  │   │ camera_token:                                                  │║
        ║  │   │ - 用于相机位姿预测                                             │║
        ║  │   │ - 在 camera_head 中提取并处理                                 │║
        ║  │   │                                                               │║
        ║  │   │ register_tokens:                                               │║
        ║  │   │ - DINOv2 的 register token 机制                               │║
        ║  │   │ - 提高特征质量，减少信息丢失                                   │║
        ║  │   │                                                               │║
        ║  │   │ scale_token:                                                   │║
        ║  │   │ - 控制尺度估计帧的注意力模式                                   │║
        ║  │   │ - 激活状态: 前8帧双向注意力                                    │║
        ║  │   │ - 非激活: 后续帧因果注意力                                      │║
        ║  │   │                                                               │║
        ║  │   │ 总大小: 1 + 4 + 1 = 6 个特殊 tokens                           │║
        ║  │   └─────────────────────────────────────────────────────────────┘║
        ║  │                                                                  ║
        ║  │   special_tokens: [B*S, 6, C]                                     ║
        ║                                                                    ║
        ║  步骤7: 拼接所有 tokens                                              ║
        ║  │   tokens = cat([special_tokens, patch_tokens], dim=1)            ║
        ║  │                                                                  ║
        ║  │   Token 序列结构:                                                  ║
        ║  │   ┌─────────────────────────────────────────────────────────────┐║
        ║  │   │ [camera][reg][reg][reg][reg][scale][patch0][patch1]...       │║
        ║  │   │  ↓     ↓                      ↓      ↓                       │║
        ║  │   │ idx=0  idx=1-4              idx=5   idx=6...                 │║
        ║  │   │                                                               │║
        ║  │   │ 总 token 数: P = 6 + P_patch ≈ 6 + 999 = 1005               │║
        ║  │   │                                                               │║
        ║  │   │ patch_start_idx = 6 (patch 从 index 6 开始)                 │║
        ║  │   └─────────────────────────────────────────────────────────────┘║
        ║  │                                                                  ║
        ║  │   tokens: [B*S, P, C]                                             ║
        ║  │   - P = num_special + P_patch ≈ 1005                             ║
        ║  │   - C = 1024                                                      ║
        ║                                                                    ║
        ║  【输出】                                                          ║
        ║                                                                    ║
        ║  tokens: [B*S, P, C]                                                ║
        ║  │   - 完整的 token 序列                                            ║
        ║                                                                    ║
        ║  B: int                                                             ║
        ║  │   - batch 大小                                                   ║
        ║                                                                    ║
        ║  S_local: int                                                       ║
        ║  │   - 本地序列长度 = S                                             ║
        ║                                                                    ║
        ║  S_global: int                                                      ║
        ║  │   - 全局序列长度 = S                                             ║
        ║                                                                    ║
        ║  P: int                                                             ║
        ║  │   - 每帧 token 数 ≈ 1005                                        ║
        ║                                                                    ║
        ║  C: int                                                             ║
        ║  │   - 嵌入维度 = 1024                                              ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            images: 输入图像 [B, S, 3, H, W]
                - 值域: [0, 1]
            num_frame_for_scale: 尺度估计帧数
                - 传递给 _prepare_special_tokens 处理 scale_token

        Returns:
            tuple: (tokens, B, S_local, S_global, P, C)
                - tokens: [B*S, P, C] 完整token序列
                - B: batch大小
                - S_local: 本地序列长度
                - S_global: 全局序列长度
                - P: 每帧token数
                - C: 嵌入维度
        """
        B, S, C_in, H, W = images.shape

        # 验证输入通道数
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # ════════════════════════════════════════════════════════════════════
        # 步骤1: 图像归一化 (ResNet标准)
        # ════════════════════════════════════════════════════════════════════
        # 使用 ResNet 的均值和标准差进行归一化
        # 这是 DINOv2 预训练时使用的标准预处理
        images = (images - self._resnet_mean) / self._resnet_std
        # 归一化后分布: 大约 mean=0, std=1

        # ════════════════════════════════════════════════════════════════════
        # 步骤2: 设置序列长度
        # ════════════════════════════════════════════════════════════════════
        # 无 Context Parallelism 切片: S_local == S_global
        S_local = S
        S_global = S

        # ════════════════════════════════════════════════════════════════════
        # 步骤3: Reshape 为 patch embedding 输入
        # ════════════════════════════════════════════════════════════════════
        # 将 batch 和 序列维度合并，方便一次性处理所有帧
        images = images.view(B * S, C_in, H, W)

        # ════════════════════════════════════════════════════════════════════
        # 步骤4: DINOv2 Patch Embedding
        # ════════════════════════════════════════════════════════════════════
        patch_tokens = self.patch_embed(images)  #返回经过多层transformer结构处理的token 这里取的是patchtoken
        # 如果输出是字典（DINOv2 格式），提取 patch tokens
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P_patch, C = patch_tokens.shape
        # P_patch = patch 数量 ≈ (H/14) * (W/14)

        # ════════════════════════════════════════════════════════════════════
        # 步骤5: 准备特殊 Tokens
        # ════════════════════════════════════════════════════════════════════
        special_tokens = self._prepare_special_tokens(
            B, S_local, S_global, C,
            num_frame_for_scale=num_frame_for_scale
        )
        # special_tokens: [B*S, num_special_tokens, C]
        # num_special_tokens = 1 + num_register + 1 = 6

        # ════════════════════════════════════════════════════════════════════
        # 步骤6: 拼接特殊 tokens + patch tokens
        # ════════════════════════════════════════════════════════════════════
        tokens = torch.cat([special_tokens, patch_tokens], dim=1)
        # tokens: [B*S, P, C]
        # P = num_special + P_patch

        _, P, C = tokens.shape

        return tokens, B, S_local, S_global, P, C

    @abstractmethod
    def _prepare_special_tokens(self, B: int, S_local: int, S_global: int, C: int, **kwargs) -> torch.Tensor:
        """
        【抽象方法】准备特殊 Tokens - 子类实现具体逻辑

        本方法创建 camera、register 和 scale 特殊 tokens，
        这些 tokens 是 GCT 模型的关键组成部分。

        ╔════════════════════════════════════════════════════════════════════╗
        ║               _prepare_special_tokens 设计框架                     ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【特殊 Tokens 的作用】                                            ║
        ║                                                                    ║
        ║  ┌─────────────────────────────────────────────────────────────┐   ║
        ║  │ Token 类型      │ 位置索引 │ 主要功能                          │   ║
        ║  │────────────────────────────────────────────────────────────│   ║
        ║  │ camera_token    │    0     │ 相机位姿预测的专用特征           │   ║
        ║  │ register_tokens │  1-4     │ 提高特征质量，减少信息丢失       │   ║
        ║  │ scale_token     │    5     │ 控制尺度帧的注意力模式           │   ║
        ║  └─────────────────────────────────────────────────────────────┘   ║
        ║                                                                    ║
        ║  【子类实现说明】                                                  ║
        ║                                                                    ║
        ║  AggregatorStream 实现:                                            ║
        ║  │   - camera_token 有两个状态                                     ║
        ║  │     [1, 2, 1, C]:                                               ║
        ║  │     - state 0: 用于 scale frames                                ║
        ║  │     - state 1: 用于后续帧                                        ║
        ║  │   - scale_token 也有两个状态                                     ║
        ║  │     - state 0: 激活，实现双向注意力                             ║
        ║  │     - state 1: 非激活，使用因果注意力                           ║
        ║                                                                    ║
        ║  【处理逻辑框架】                                                  ║
        ║                                                                    ║
        ║  输入:                                                              ║
        ║  │   B: batch大小                                                  ║
        ║  │   S_local: 本地序列长度                                         ║
        ║  │   S_global: 全局序列长度                                        ║
        ║  │   C: 嵌入维度                                                   ║
        ║  │   **kwargs:                                                      ║
        ║  │     num_frame_for_scale: 尺度帧数（用于scale_token）           ║
        ║                                                                    ║
        ║  步骤1: 确定有效尺度帧数                                            ║
        ║  │   effective_scale_frames = min(num_frame_for_scale, S_global)   ║
        ║  │                                                                  ║
        ║  步骤2: 使用 slice_expand_and_flatten 处理各 token                 ║
        ║  │                                                                  ║
        ║  │   camera_token:                                                   ║
        ║  │   │   self.camera_token: [1, 2, 1, C]                            ║
        ║  │   │   扩展到 [B, S, 1, C]                                         ║
        ║  │   │   扁平化为 [B*S, 1, C]                                        ║
        ║  │   │   first_num_frame 决定哪个状态用于哪些帧                     ║
        ║  │                                                                  ║
        ║  │   register_tokens:                                                ║
        ║  │   │   self.register_token: [1, 2, num_reg, C]                    ║
        ║  │   │   扩展到 [B, S, num_reg, C]                                   ║
        ║  │   │   扁平化为 [B*S, num_reg, C]                                  ║
        ║  │                                                                  ║
        ║  │   scale_token:                                                     ║
        ║  │   │   self.scale_token: [1, 2, 1, C]                              ║
        ║  │   │   扩展时 first_num_frame = num_frame_for_scale               ║
        ║  │   │   前 N 帧使用 state 0（激活）                                  ║
        ║  │   │   后续帧使用 state 1（非激活）                                 ║
        ║  │   │   扁平化为 [B*S, 1, C]                                         ║
        ║  │                                                                  ║
        ║  步骤3: 拼接所有特殊 tokens                                          ║
        ║  │   special_tokens = cat([camera, register, scale], dim=1)         ║
        ║  │   输出: [B*S, num_special_tokens, C]                              ║
        ║  │   num_special_tokens = 1 + num_register + 1 = 6                 ║
        ║                                                                    ║
        ║  【Scale Token 的激活机制】                                        ║
        ║                                                                    ║
        ║  scale_token 的两个状态决定了注意力模式:                            ║
        ║                                                                    ║
        ║  State 0 (激活) - 用于 scale frames:                               ║
        ║  │   ┌─────────────────────────────────────────────────────────────┐║
        ║  │   │ Scale Frames (前8帧):                                         │║
        ║  │   │                                                               │║
        ║  │   │   Frame 0  1  2  3  4  5  6  7                               │║
        ║  │   │     ↓    ↓  ↓  ↓  ↓  ↓  ↓  ↓                                 │║
        ║  │   │   scale_token 激活                                            │║
        ║  │   │                                                               │║
        ║  │   │   这些帧之间使用双向注意力:                                    │║
        ║  │   │   ┌───────────────────────────────────────────────────────┐ │║
        ║  │   │   │      Frame 0  1  2  3  4  5  6  7                      │ │║
        ║  │   │   │  0    ✓    ✓   ✓   ✓   ✓   ✓   ✓   ✓   <- 可见所有  │ │║
        ║  │   │   │  1    ✓    ✓   ✓   ✓   ✓   ✓   ✓   ✓   <- 可见所有  │ │║
        ║  │   │   │  ...                                                    │ │║
        ║  │   │   │  7    ✓    ✓   ✓   ✓   ✓   ✓   ✓   ✓   <- 可见所有  │ │║
        ║  │   │   │                                                          │ │║
        ║  │   │   │  目的: 建立全局坐标系和尺度基准                          │ │║
        ║  │   │   │  类似 SLAM 的初始化阶段                                   │ │║
        ║  │   │   └───────────────────────────────────────────────────────┘ │║
        ║  │   └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  State 1 (非激活) - 用于后续帧:                                     ║
        ║  │   ┌─────────────────────────────────────────────────────────────┐║
        ║  │   │ 后续帧 (Frame 8+):                                             │║
        ║  │   │                                                               │║
        ║  │   │   Frame 8  9  10  11  ...                                    │║
        ║  │   │     ↓    ↓   ↓    ↓                                          │║
        ║  │   │   scale_token 非激活                                          │║
        ║  │   │                                                               │║
        ║  │   │   使用因果注意力:                                              │║
        ║  │   │   ┌───────────────────────────────────────────────────────┐ │║
        ║  │   │   │      Frame 8  9  10  11  12  ...                        │ │║
        ║  │   │   │  8    ✓    ✗   ✗    ✗    ✗    <- 只看0-8            │ │║
        ║  │   │   │  9    ✓    ✓   ✗    ✗    ✗    <- 只看0-9            │ │║
        ║  │   │   │  10   ✓    ✓   ✓    ✗    ✗    <- 只看0-10           │ │║
        ║  │   │   │  ...                                                    │ │║
        ║  │   │   │                                                          │ │║
        ║  │   │   │  因果注意力使推理可以在线进行                             │ │║
        ║  │   │   │  不需要等待未来帧                                          │ │║
        ║  │   │   └───────────────────────────────────────────────────────┘ │║
        ║  │   └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  输出:                                                              ║
        ║  │   special_tokens: [B*S, num_special_tokens, C]                  ║
        ║  │   - num_special_tokens = 1 + num_register_tokens + 1 = 6        ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            B: Batch size
            S_local: 本地序列长度
            S_global: 全局序列长度
            C: 嵌入维度
            **kwargs: 模式特定参数
                - num_frame_for_scale: 尺度估计帧数

        Returns:
            special_tokens: [B*S, N_special, C]
                - N_special = 1 + num_register_tokens + 1 = 6
        """
        pass

    def _get_positions(self, B: int, S: int, H: int, W: int, device) -> Optional[torch.Tensor]:
        """
        【2D位置编码生成】为 RoPE 生成位置信息

        本方法为每个 token 生成 2D 位置编码，用于 Rotary Position Embedding (RoPE)。
        RoPE 是一种相对位置编码方法，将位置信息融入注意力计算中。

        ╔════════════════════════════════════════════════════════════════════╗
        ║                   _get_positions 处理流程                          ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【RoPE 原理】                                                      ║
        ║                                                                    ║
        ║  传统位置编码（绝对位置）:                                          ║
        ║  │   pos_emb = lookup_table[position]                             ║
        ║  │   token = token + pos_emb                                      ║
        ║                                                                    ║
        ║  RoPE（旋转位置编码）:                                              ║
        ║  │   将位置信息编码为旋转角度                                       ║
        ║  │   在注意力计算中应用旋转                                        ║
        ║  │   使得相对位置信息自然融入                                      ║
        ║                                                                    ║
        ║  优势:                                                              ║
        ║  │   1. 相对位置信息更自然（token间的相对距离更重要）             ║
        ║  │   2. 支持长序列，不易出现位置编码饱和                          ║
        ║  │   3. 2D RoPE 可以编码空间位置（x, y坐标）                      ║
        ║                                                                    ║
        ║  【处理流程】                                                      ║
        ║                                                                    ║
        ║  输入:                                                              ║
        ║  │   B: batch大小                                                  ║
        ║  │   S: 序列长度（帧数）                                           ║
        ║  │   H: 图像高度                                                   ║
        ║  │   W: 图像宽度                                                   ║
        ║  │   device: 计算设备                                              ║
        ║                                                                    ║
        ║  步骤1: 计算patch网格尺寸                                          ║
        ║  │   patches_h = H // patch_size  例如: 518/14 ≈ 37              ║
        ║  │   patches_w = W // patch_size  例如: 378/14 ≈ 27              ║
        ║  │   total_patches = patches_h * patches_w ≈ 999                 ║
        ║                                                                    ║
        ║  步骤2: 生成patch位置坐标                                          ║
        ║  │   PositionGetter 生成每个patch的 (row, col) 坐标              ║
        ║  │   对于每个帧:                                                    ║
        ║  │   ┌───────────────────────────────────────────────────────┐    ║
        ║  │   │  Patch网格 (patches_h × patches_w):                    │    ║
        ║  │   │                                                          │    ║
        ║  │   │  (0,0) (0,1) (0,2) ... (0,patches_w-1)                  │    ║
        ║  │   │  (1,0) (1,1) (1,2) ... (1,patches_w-1)                  │    ║
        ║  │   │  ...                                                      │    ║
        ║  │   │  (patches_h-1,0) ... (patches_h-1,patches_w-1)         │    ║
        ║  │   │                                                          │    ║
        ║  │   │  每个patch的位置 = (row, col)                            │    ║
        ║  │   │  pos = position_getter(B*S, patches_h, patches_w)      │    ║
        ║  │   │  输出: [B*S, total_patches, 2]                          │    ║
        ║  │   │    其中 [:,:,0] = row坐标                               │    ║
        ║  │   │    其中 [:,:,1] = col坐标                               │    ║
        ║  │   └───────────────────────────────────────────────────────┘    ║
        ║                                                                    ║
        ║  步骤3: 添加特殊token的位置                                        ║
        ║  │   特殊tokens位于位置索引 0~patch_start_idx-1                 ║
        ║  │   patch_start_idx = 1 + num_register + 1 = 6                 ║
        ║  │                                                                  ║
        ║  │   patch的位置需要偏移:                                          ║
        ║  │   │   pos = pos + 1  # 偏移，避开special tokens的位置0       ║
        ║  │                                                                  ║
        ║  │   为special tokens创建位置0:                                    ║
        ║  │   │   pos_special = zeros([B*S, patch_start_idx, 2])          ║
        ║  │   │   # camera, register, scale tokens都在位置0              ║
        ║  │                                                                  ║
        ║  │   拼接:                                                          ║
        ║  │   │   pos = cat([pos_special, pos_patch], dim=1)              ║
        ║  │   │   输出: [B*S, P, 2]  其中 P = special + patches           ║
        ║                                                                    ║
        ║  输出:                                                              ║
        ║  │   pos: [B*S, P, 2]                                              ║
        ║  │   - 用于 RoPE 计算                                               ║
        ║  │   - [:, :, 0] = row/高度坐标                                    ║
        ║  │   - [:, :, 1] = col/宽度坐标                                    ║
        ║                                                                    ║
        ║  【RoPE 在注意力中的应用】                                        ║
        ║                                                                    ║
        ║  在 Block 的注意力计算中:                                          ║
        ║  │   # 标准注意力                                                   ║
        ║  │   Q, K, V = linear(tokens)                                     ║
        ║  │   Attention = softmax(Q @ K^T) @ V                             ║
        ║                                                                    ║
        ║  │   # RoPE注意力                                                  ║
        ║  │   Q_rot = apply_rotary_emb(Q, pos)                             ║
        ║  │   K_rot = apply_rotary_emb(K, pos)                             ║
        ║  │   Attention = softmax(Q_rot @ K_rot^T) @ V                    ║
        ║                                                                    ║
        ║  │   旋转角度由位置决定:                                            ║
        ║  │   │   θ = pos / frequency                                      ║
        ║  │   │   rotate(Q, θ) 使得相对位置的token有正确的相对关系         ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            B: Batch size
            S: 序列长度（帧数）
            H: 图像高度（像素）
            W: 图像宽度（像素）
            device: 计算设备 (cuda/cpu)

        Returns:
            pos: 位置编码 [B*S, P, 2]
                - P: 每帧的token数（special + patches）
                - [:, :, 0]: row坐标（高度方向）
                - [:, :, 1]: col坐标（宽度方向）
                - 如果 rope=None 则返回 None
        """
        # 如果 RoPE 未启用，返回 None
        if self.rope is None:
            return None

        # ══════════════════════════════════════════════════════════════════
        # 步骤1: 获取 patch 位置坐标
        # ══════════════════════════════════════════════════════════════════
        # PositionGetter 为每个 patch 生成 坐标
        # 输出: [B*S, patches_h * patches_w, 2]
        pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=device)

        # ══════════════════════════════════════════════════════════════════
        # 步骤2: 添加偏移并为特殊 tokens 创建位置
        # ══════════════════════════════════════════════════════════════════
        # patch tokens 从 index = patch_start_idx 开始
        # 需要 +1 偏移，避免与 special tokens 的位置0冲突
        if self.patch_start_idx > 0:
            # patch位置偏移 +1
            pos = pos + 1
            # 为 special tokens (camera, register, scale) 创建位置0
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2, dtype=pos.dtype, device=device)
            # 拼接: special tokens 在前面，patch tokens 在后面
            pos = torch.cat([pos_special, pos], dim=1)

        return pos

    def _process_frame_attention(
        self,
        tokens: torch.Tensor,
        B: int,
        S: int,
        P: int,
        C: int,
        frame_idx: int,
        pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        【帧内自注意力处理】每帧独立进行自注意力计算

        Frame Attention 是帧内的注意力，每个帧独立处理，帧间不通信。
        这是与 Global Attention 的关键区别。

        ╔════════════════════════════════════════════════════════════════════╗
        ║              _process_frame_attention 处理流程                     ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【Frame vs Global Attention 区别】                                ║
        ║                                                                    ║
        ║  Frame Attention:                                                  ║
        ║  │   - 每帧独立处理                                                 ║
        ║  │   - 帧间完全不通信                                               ║
        ║  │   - tokens shape: [B*S, P, C] 保持不变                          ║
        ║  │   - 注意力范围: 单帧内的所有 tokens                             ║
        ║                                                                    ║
        ║  Global Attention:                                                 ║
        ║  │   - 跨帧注意力                                                   ║
        ║  │   - 帧间可以通信                                                 ║
        ║  │   - tokens reshape: [B*S, P, C] -> [B, S*P, C]                 ║
        ║  │   - 注意力范围: 多帧的 tokens                                   ║
        ║                                                                    ║
        ║  【处理流程】                                                      ║
        ║                                                                    ║
        ║  输入:                                                              ║
        ║  │   tokens: [B*S, P, C]                                           ║
        ║  │   - B*S: 扁平化的 batch * 序列                                  ║
        ║  │   - P: 每帧 token 数 (special + patches)                       ║
        ║  │   - C: 嵌入维度 (1024)                                          ║
        ║  │                                                                  ║
        ║  │   pos: [B*S, P, 2]                                               ║
        ║  │   - RoPE 位置编码                                               ║
        ║  │                                                                  ║
        ║  │   frame_idx: 当前处理的 block 索引                              ║
        ║                                                                    ║
        ║  步骤1: 确保形状正确                                                ║
        ║  │   tokens = tokens.view(B*S, P, C)  # 确保是正确的形状          ║
        ║  │   pos = pos.view(B*S, P, 2)      # 确保位置编码正确             ║
        ║                                                                    ║
        ║  步骤2: 处理 aa_block_size 个 frame_blocks                        ║
        ║  │   intermediates = []                                            ║
        ║  │                                                                  ║
        ║  │   for i in range(aa_block_size):                               ║
        ║  │       ┌───────────────────────────────────────────────────────┐║
        ║  │       │ frame_blocks[frame_idx]: Transformer Block            │║
        ║  │       │                                                         │║
        ║  │       │ Block 内部结构:                                          │║
        ║  │       │ ┌─────────────────────────────────────────────────────┐║║
        ║  │       │ │ 1. LayerNorm                                          │║║
        ║  │       │ │    tokens_norm = LayerNorm(tokens)                   │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 2. Self-Attention                                       │║║
        ║  │       │ │    Q, K, V = Linear(tokens_norm)                      │║║
        ║  │       │ │    Q_rot = RoPE(Q, pos)  # 应用旋转位置编码           │║║
        ║  │       │ │    K_rot = RoPE(K, pos)                               │║║
        ║  │       │ │    Attn = softmax(Q_rot @ K_rot^T / sqrt(d)) @ V     │║║
        ║  │       │ │    Attn_out = Linear(Attn) + tokens  # 残差连接      │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 3. LayerNorm                                            │║║
        ║  │       │ │    Attn_norm = LayerNorm(Attn_out)                    │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 4. FFN (Feed-Forward Network)                          │║║
        ║  │       │ │    FFN_out = MLP(Attn_norm) + Attn_out  # 残差连接    │║║
        ║  │       │ │    MLP: Linear(C -> 4C) -> GELU -> Linear(4C -> C)   │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 5. LayerScale                                           │║║
        ║  │       │ │    output = gamma * FFN_out  # gamma初始化为0.01      │║║
        ║  │       │ └─────────────────────────────────────────────────────┘║║
        ║  │       │                                                         │║
        ║  │       │ 输入/输出:                                               │║
        ║  │       │   tokens_in:  [B*S, P, C]                               │║
        ║  │       │   tokens_out: [B*S, P, C]                               │║
        ║  │       │                                                         │║
        ║  │       │ 训练模式:                                                │║
        ║  │       │   if training and gradient_checkpoint:                 │║
        ║  │       │     tokens = checkpoint(block, tokens, pos)             │║
        ║  │       │   # 节省内存，但增加计算时间                             │║
        ║  │       │                                                         │║
        ║  │       │ 推理模式:                                                │║
        ║  │       │   tokens = block(tokens, pos=pos)                       │║
        ║  │       └───────────────────────────────────────────────────────┘║
        ║  │                                                                  ║
        ║  │       frame_idx += 1  # 移动到下一个 block                      ║
        ║  │                                                                  ║
        ║  │       # 收集中间输出                                              ║
        ║  │       intermediate = tokens.view(B, S, P, C)  # reshape        ║
        ║  │       intermediates.append(intermediate)                       ║
        ║                                                                    ║
        ║  输出:                                                              ║
        ║  │   tokens: [B*S, P, C]  处理后的 tokens                          ║
        ║  │   frame_idx: 更新后的 block 索引                                ║
        ║  │   intermediates: List[[B, S, P, C]]  各 block 的输出            ║
        ║                                                                    ║
        ║  【为什么使用 Frame + Global 交替？】                               ║
        ║                                                                    ║
        ║  交替执行 frame 和 global attention:                               ║
        ║  │   1. Frame attention: 提取帧内细节特征                         ║
        ║  │      - 理解单帧的视觉内容                                        ║
        ║  │      - 处理空间关系                                               ║
        ║  │                                                                  ║
        ║  │   2. Global attention: 建立跨帧关系                             ║
        ║  │      - 时间一致性                                                 ║
        ║  │      - 相机运动估计                                               ║
        ║  │      - 3D结构推理                                                 ║
        ║                                                                    ║
        ║  │   交替执行使得模型同时学习空间和时间特征                         ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            tokens: 输入 tokens [B*S, P, C]
            B: Batch size
            S: 序列长度（帧数）
            P: 每帧 token 数
            C: 嵌入维度
            frame_idx: 当前 frame block 索引
            pos: 位置编码 [B*S, P, 2]（可选）

        Returns:
            tuple: (tokens, frame_idx, intermediates)
                - tokens: 处理后的 tokens [B*S, P, C]
                - frame_idx: 更新后的 block 索引
                - intermediates: 各 block 输出的列表 [B, S, P, C]
        """
        # ══════════════════════════════════════════════════════════════════
        # 步骤1: 确保输入形状正确
        # ══════════════════════════════════════════════════════════════════
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B * S, P, 2)

        intermediates = []

        # ══════════════════════════════════════════════════════════════════
        # 步骤2: 处理 aa_block_size 个 frame_blocks
        # ══════════════════════════════════════════════════════════════════
        for i in range(self.aa_block_size):
            # 训练时使用 gradient checkpoint 节省内存
            if self.training and self.use_gradient_checkpoint:
                from torch.utils.checkpoint import checkpoint
                # checkpoint 会重新计算前向传播，但节省中间激活的内存
                tokens = checkpoint(
                    self.frame_blocks[frame_idx],
                    tokens,
                    pos,
                    False,  # enable_ulysses_cp (始终为False)
                    use_reentrant=self.use_reentrant
                )
            else:
                # 推理时直接执行 block
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos, enable_ulysses_cp=False)

            # 更新 block 索引
            frame_idx += 1

            # 收集中间输出（reshape 为 [B, S, P, C]）
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    @abstractmethod
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
        **kwargs
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """
        【抽象方法】跨帧全局注意力处理 - 子类实现具体逻辑

        Global Attention 是跨帧的注意力，允许帧间通信。
        这是实现流式推理和因果注意力的核心。

        ╔════════════════════════════════════════════════════════════════════╗
        ║           _process_global_attention 设计框架                       ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【子类实现说明】                                                  ║
        ║                                                                    ║
        ║  AggregatorStream 实现了本方法:                                    ║
        ║  │   _process_global_attention() -> _process_causal_stream()      ║
        ║                                                                    ║
        ║  不同子类的实现策略:                                                ║
        ║                                                                    ║
        ║  1. AggregatorStream (流式推理):                                   ║
        ║  │   - 使用因果注意力                                               ║
        ║  │   - 帧 t 只能看到帧 0~t                                        ║
        ║  │   - 使用 FlashInferBlock 或 SDPABlock                          ║
        ║  │   - KV cache 存储历史帧的 Key/Value                             ║
        ║  │   - scale token 控制前N帧双向注意力                             ║
        ║                                                                    ║
        ║  2. AggregatorBatch (批处理，如果存在):                            ║
        ║  │   - 使用双向注意力                                               ║
        ║  │   - 所有帧可以互相看到                                          ║
        ║  │   - 适合离线处理                                                 ║
        ║                                                                    ║
        ║  【通用处理流程】                                                  ║
        ║                                                                    ║
        ║  输入:                                                              ║
        ║  │   tokens: [B*S_local, P, C]                                     ║
        ║  │   - B*S_local: 扁平化的 batch * local序列                      ║
        ║  │   - P: 每帧 token 数                                            ║
        ║  │   - C: 嵌入维度                                                  ║
        ║  │                                                                  ║
        ║  │   S_local: 本地序列长度                                          ║
        ║  │   S_global: 全局序列长度                                         ║
        ║  │   - 在流式推理中可能不同                                         ║
        ║  │                                                                  ║
        ║  │   global_idx: 当前 global block 索引                            ║
        ║  │                                                                  ║
        ║  │   pos: 位置编码 [B*S_global, P, 2]                              ║
        ║  │                                                                  ║
        ║  │   **kwargs: 模式特定参数                                         ║
        ║  │     - num_frame_for_scale: 尺度帧数                             ║
        ║  │     - sliding_window_size: 滑动窗口大小                         ║
        ║  │     - num_frame_per_block: 每block帧数                          ║
        ║                                                                    ║
        ║  步骤1: Reshape tokens                                             ║
        ║  │   将 tokens 从帧独立形式转为跨帧形式                            ║
        ║  │   [B*S_local, P, C] -> [B, S_local*P, C]                       ║
        ║  │   或 [B, S_global*P, C]                                         ║
        ║                                                                    ║
        ║  步骤2: 处理 global_blocks                                         ║
        ║  │   for i in range(aa_block_size):                               ║
        ║  │       ┌───────────────────────────────────────────────────────┐║
        ║  │       │ global_blocks[global_idx]                              │║
        ║  │       │                                                         │║
        ║  │       │ 子类特定实现:                                            │║
        ║  │       │                                                         │║
        ║  │       │ AggregatorStream._process_causal_stream():            │║
        ║  │       │                                                         │║
        ║  │       │ ┌─────────────────────────────────────────────────────┐║║
        ║  │       │ │ KV Cache 机制:                                        │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ cache 结构:                                             │║║
        ║  │       │ │   FlashInfer: FlashInferKVCacheManager               │║║
        ║  │       │ │   SDPA: dict {k_0:[B,1,S_cache,P,C], ...}            │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 计算流程:                                                │║║
        ║  │       │ │   1. 获取 KV cache manager                             │║║
        ║  │       │ │   2. 当前帧计算 Q, K, V                                 │║║
        ║  │       │ │   3. 应用 RoPE (可选)                                   │║║
        ║  │       │ │   4. 构建注意力掩码 (因果)                               │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 因果掩码示例 (S=5):                                      │║║
        ║  │       │ │ ┌─────────────────────────────────────────────────────┐║║║
        ║  │       │ │ │     Frame 0  1  2  3  4                              │║║║
        ║  │       │ │ │  0    ✓   ✓   ✓   ✓   ✓   <- Frame 0 看到所有      │║║║
        ║  │       │ │ │  1    ✗   ✓   ✓   ✓   ✓   <- Frame 1 不看Frame 0   │║║║
        ║  │       │ │ │  2    ✗   ✗   ✓   ✓   ✓                            │║║║
        ║  │       │ │ │  3    ✗   ✗   ✗   ✓   ✓                            │║║║
        ║  │       │ │ │  4    ✗   ✗   ✗   ✗   ✓                            │║║║
        ║  │       │ │ │                                                         │║║║
        ║  │       │ │ │ 但对于 scale frames (前8帧):                          │║║║
        ║  │       │ │ │   使用双向注意力，所有帧互相可见                       │║║║
        ║  │       │ │ └─────────────────────────────────────────────────────┘║║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 5. Attention 计算                                        │║║
        ║  │       │ │    Attn(Q_cur, [K_cache + K_cur], [V_cache + V_cur])   │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 6. 存储当前帧 KV 到 cache (如果不是 skip_append)        │║║
        ║  │       │ │                                                         │║║
        ║  │       │ │ 7. 如果超过 sliding_window，驱逐最旧帧                  │║║
        ║  │       │ │    但保留 scale frames 和 special tokens               │║║
        ║  │       │ └─────────────────────────────────────────────────────┘║║
        ║  │       │                                                         │║
        ║  │       │ global_idx += 1                                          │║
        ║  │       │ intermediates.append(tokens.reshape)                    │║
        ║  │       └───────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  输出:                                                              ║
        ║  │   tokens: 处理后的 tokens                                       ║
        ║  │   global_idx: 更新后的 block 索引                               ║
        ║  │   intermediates: 各 block 输出的列表                            ║
        ║                                                                    ║
        ║  【Scale Token 工作原理】                                          ║
        ║                                                                    ║
        ║  scale_token 的激活状态决定注意力模式:                              ║
        ║                                                                    ║
        ║  前 num_frame_for_scale 帧 (scale frames):                         ║
        ║  │   - scale_token 激活 (使用第一个状态)                           ║
        ║  │   - 这些帧之间使用双向注意力                                     ║
        ║  │   - 目的: 建立全局坐标系的尺度基准                               ║
        ║  │   - 类似 SLAM 的初始化阶段                                       ║
        ║                                                                    ║
        ║  后续帧:                                                            ║
        ║  │   - scale_token 非激活 (使用第二个状态)                         ║
        ║  │   - 使用因果注意力                                               ║
        ║  │   - 帧 t 只能看到帧 0~t                                        ║
        ║                                                                    ║
        ║  【3D RoPE (可选)】                                                ║
        ║                                                                    ║
        ║  如果 enable_3d_rope=True:                                         ║
        ║  │   - 使用 3D 旋转位置编码                                         ║
        ║  │   - 编码时间维度 + 空间维度                                      ║
        ║  │   - 提高流式推理的时间一致性                                     ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            tokens: 输入 tokens
            B: Batch size
            S_local: 本地序列长度
            S_global: 全局序列长度
            P: 每帧 token 数
            C: 嵌入维度
            global_idx: 当前 global block 索引
            pos: 位置编码（可选）
            **kwargs: 模式特定参数
                - num_frame_for_scale: 尺度帧数
                - sliding_window_size: 滑动窗口大小
                - num_frame_per_block: 每block帧数

        Returns:
            tuple: (tokens, global_idx, intermediates)
                - tokens: 处理后的 tokens
                - global_idx: 更新后的 block 索引
                - intermediates: 各 block 输出的列表
        """
        pass

    def forward(
        self,
        images: torch.Tensor,
        selected_idx: Optional[List[int]] = None,
        # Mode-specific parameters (模式特定参数)
        num_frame_for_scale: Optional[int] = None,
        sliding_window_size: Optional[int] = None,
        num_frame_per_block: int = 1,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        【前向传播】Aggregator 的核心推理入口

        本方法串联了图像嵌入、位置编码、帧内注意力和跨帧注意力的完整处理流程。
        交替执行 frame_blocks 和 global_blocks 实现空间+时间特征学习。

        ╔════════════════════════════════════════════════════════════════════╗
        ║                    forward() 执行流程详解                          ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【输入】                                                          ║
        ║  images: [B, S, 3, H, W]                                           ║
        ║  │   - B: batch大小（通常为1）                                     ║
        ║  │   - S: 序列长度（帧数）                                         ║
        ║  │   - H, W: 图像尺寸（默认518x378）                              ║
        ║  │   - 值域: [0, 1]                                                ║
        ║                                                                    ║
        ║  selected_idx: 选择输出哪些block的特征                            ║
        ║  │   - None: 输出所有block                                        ║
        ║  │   - [4, 11, 17, 23]: 输出这4个block（用于多尺度预测）          ║
        ║                                                                    ║
        ║  【处理流程】                                                      ║
        ║                                                                    ║
        ║  步骤1: _embed_images() - 图像嵌入                                ║
        ║  ┌───────────────────────────────────────────────────────────────┐║
        ║  │                                                                 │║
        ║  │  输入: images [B, S, 3, H, W]                                   │║
        ║  │                                                                 │║
        ║  │  1.1 图像归一化:                                                 │║
        ║  │      images_norm = (images - ResNet_mean) / ResNet_std         │║
        ║  │      - ResNet_mean = [0.485, 0.456, 0.406]                     │║
        ║  │      - ResNet_std = [0.229, 0.224, 0.225]                      │║
        ║  │      - 这是 DINOv2 的标准预处理                                 │║
        ║  │                                                                 │║
        ║  │  1.2 Reshape:                                                    │║
        ║  │      images_norm = images_norm.view(B*S, 3, H, W)              │║
        ║  │      - 将所有帧合并，方便 batch 处理                            │║
        ║  │                                                                 │║
        ║  │  1.3 DINOv2 Patch Embedding:                                     │║
        ║  │      patch_tokens = self.patch_embed(images_norm)              │║
        ║  │      - DINOv2 ViT-L/14:                                         │║
        ║  │        - patch_size = 14                                        │║
        ║  │        - 每个patch 14x14像素 -> 1024维向量                      │║
        ║  │        - patch数量 = (H/14)*(W/14) ≈ 37*27 ≈ 999               │║
        ║  │      - 输出: [B*S, P_patch, C]                                  │║
        ║  │        - P_patch = patch数量                                    │║
        ║  │        - C = 1024                                               │║
        ║  │                                                                 │║
        ║  │  1.4 准备 Special Tokens:                                        │║
        ║  │      special_tokens = _prepare_special_tokens(B, S, C)         │║
        ║  │      - camera_token [B*S, 1, C]: 相机位姿预测                   │║
        ║  │      - register_tokens [B*S, 4, C]: 特征增强                   │║
        ║  │      - scale_token [B*S, 1, C]: 尺度控制                        │║
        ║  │                                                                 │║
        ║  │  1.5 拼接:                                                       │║
        ║  │      tokens = cat([special_tokens, patch_tokens], dim=1)       │║
        ║  │      - 输出: [B*S, P, C]                                         │║
        ║  │        - P = 1 + 4 + 1 + P_patch ≈ 1005                        │║
        ║  │                                                                 │║
        ║  │  返回:                                                            │║
        ║  │    tokens: [B*S, P, C]                                           │║
        ║  │    B, S_local, S_global, P, C                                    │║
        ║    │   - S_local = S_global = S (无CP切片)                         │║
        ║  │                                                                 │║
        ║  └───────────────────────────────────────────────────────────────┘║
        ║       │                                                             ║
        ║       ↓                                                             ║
        ║  步骤2: _get_positions() - 生成位置编码                            ║
        ║  ┌───────────────────────────────────────────────────────────────┐║
        ║  │                                                                 │║
        ║  │  为每个 token 生成 2D 位置坐标，用于 RoPE                       │║
        ║  │                                                                 │║
        ║  │  pos_local = _get_positions(B, S_local, H, W, device)          │║
        ║  │  pos_global = _get_positions(B, S_global, H, W, device)        │║
        ║  │                                                                 │║
        ║  │  输出: [B*S, P, 2]                                               │║
        ║  │    [:, :, 0] = row坐标                                           │║
        ║  │    [:, :, 1] = col坐标                                           │║
        ║  │                                                                 │║
        ║  └───────────────────────────────────────────────────────────────┘║
        ║       │                                                             ║
        ║       ↓                                                             ║
        ║  步骤3: 交替执行 frame_blocks 和 global_blocks                     ║
        ║  ┌───────────────────────────────────────────────────────────────┐║
        ║  │                                                                 │║
        ║  │  aa_order = ["frame", "global"]                                 │║
        ║  │  aa_block_num = depth / aa_block_size = 24                     │║
        ║  │                                                                 │║
        ║  │  frame_idx = 0                                                   │║
        ║  │  global_idx = 0                                                  │║
        ║  │  output_list = []                                                │║
        ║  │                                                                 │║
        ║  │  for block_group_idx in range(24):                             │║
        ║  │      ┌─────────────────────────────────────────────────────────┐║║
        ║  │      │ 循环结构 (aa_order = ["frame", "global"]):                │║║
        ║  │      │                                                           │║║
        ║  │      │   第1组: frame_block[0] -> global_block[0]               │║║
        ║  │      │   第2组: frame_block[1] -> global_block[1]               │║║
        ║  │      │   ...                                                      │║║
        ║  │      │   第24组: frame_block[23] -> global_block[23]            │║║
        ║  │      │                                                           │║║
        ║  │      │ 每个 block_group 内的执行顺序:                             │║║
        ║  │      │   1. frame_blocks: 帧内自注意力                           │║║
        ║  │      │   2. global_blocks: 跨帧注意力                           │║║
        ║  │      └─────────────────────────────────────────────────────────┘║║
        ║  │                                                                 │║
        ║  │      for attn_type in ["frame", "global"]:                      │║
        ║  │                                                                 │║
        ║  │          if attn_type == "frame":                               │║
        ║  │              ┌─────────────────────────────────────────────────┐║║
        ║  │              │ _process_frame_attention()                      │║║
        ║  │              │                                                   │║║
        ║  │              │  帧内自注意力:                                     │║║
        ║  │              │  - tokens: [B*S, P, C] (保持形状)                │║║
        ║  │              │  - 使用 RoPE 位置编码                             │║║
        ║  │              │  - 每帧独立，帧间不通信                           │║║
        ║  │              │                                                   │║║
        ║  │              │  计算过程:                                         │║║
        ║  │              │    Q, K, V = linear(tokens)                      │║║
        ║  │              │    Q_rot, K_rot = RoPE(Q, K, pos)                │║║
        ║  │              │    Attn = softmax(Q_rot @ K_rot^T) @ V           │║║
        ║  │              │    tokens = FFN(LayerNorm(Attn))                 │║║
        ║  │              │                                                   │║║
        ║  │              │  返回:                                             │║║
        ║  │              │    tokens, frame_idx, frame_intermediates        │║║
        ║  │              │    frame_intermediates: [B, S, P, C]             │║║
        ║  │              └─────────────────────────────────────────────────┘║║
        ║  │                                                                 │║
        ║  │          elif attn_type == "global":                            │║
        ║  │              ┌─────────────────────────────────────────────────┐║║
        ║  │              │ _process_global_attention()                     │║║
        ║  │              │                                                   │║║
        ║  │              │  跨帧注意力 (子类实现):                            │║║
        ║  │              │  - AggregatorStream 使用因果注意力               │║║
        ║  │              │  - KV cache 存储历史帧                            │║║
        ║  │              │  - scale token 控制前N帧双向                     │║║
        ║  │              │                                                   │║║
        ║  │              │  tokens reshape: [B*S,P,C] -> [B,S*P,C]          │║║
        ║  │              │                                                   │║║
        ║  │              │  返回:                                             │║║
        ║  │              │    tokens, global_idx, global_intermediates      │║║
        ║  │              │    global_intermediates: [B, S, P, C]            │║║
        ║  │              └─────────────────────────────────────────────────┘║║
        ║  │                                                                 │║
        ║  │      # 收集输出 (如果 block_group_idx 在 selected_idx 中)        │║
        ║  │      if selected_idx is None or block_group_idx in selected_idx:│║
        ║  │          for i in range(len(frame_intermediates)):              │║
        ║  │              # 拼接 frame 和 global 特征                         │║
        ║  │              concat = cat([                                     │║
        ║  │                  frame_intermediates[i],  # [B, S, P, C]        │║
        ║  │                  global_intermediates[i]   # [B, S, P, C]       │║
        ║  │              ], dim=-1)                                          │║
        ║  │              # 输出: [B, S, P, 2C]                                │║
        ║  │              output_list.append(concat)                          │║
        ║  │                                                                 │║
        ║  └───────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  【输出】                                                          ║
        ║                                                                    ║
        ║  output_list: List[[B, S, P, 2C]]                                 ║
        ║  │   - 长度取决于 selected_idx                                    ║
        ║  │   - 如果 selected_idx=[4,11,17,23]: 输出4个特征               ║
        ║  │   - 每个特征: frame特征(C) + global特征(C) 拼接                ║
        ║  │   - 用于 DPTHead 的多尺度预测                                   ║
        ║                                                                    ║
        ║  patch_start_idx: int                                              ║
        ║  │   - patch tokens 的起始索引 = 6                                ║
        ║  │   - 用于 depth_head 和 point_head 提取 patch 特征              ║
        ║                                                                    ║
        ║  【为什么使用 Frame + Global 交替？】                               ║
        ║                                                                    ║
        ║  Frame Attention:                                                  ║
        ║  │   - 学习每帧的空间特征                                          ║
        ║  │   - 理解图像的视觉内容                                          ║
        ║  │   - 提取局部细节                                                 ║
        ║                                                                    ║
        ║  Global Attention:                                                 ║
        ║  │   - 学习跨帧的时间特征                                          ║
        ║  │   - 建立帧间关系                                                 ║
        ║  │   - 推理相机运动和3D结构                                        ║
        ║                                                                    ║
        ║  交替执行的优势:                                                    ║
        ║  │   1. 同时学习空间和时间特征                                     ║
        ║  │   2. 逐层加深特征抽象                                           ║
        ║  │   3. 避免 frame 和 global 独立处理时的信息断层                  ║
        ║                                                                    ║
        ║  【selected_idx 的作用】                                           ║
        ║                                                                    ║
        ║  DPTHead 要多尺度特征:                                            ║
        ║  │   - 早期block (idx=4): 低级视觉特征                            ║
        ║  │   - 中期block (idx=11): 中级语义特征                           ║
        ║  │   - 后期block (idx=17,23): 高级抽象特征                        ║
        ║                                                                    ║
        ║  多尺度特征用于密集预测:                                            ║
        ║  │   - 逐步上采样解码                                              ║
        ║  │   - 不同尺度提供不同粒度的信息                                  ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            images: 输入图像 [B, S, 3, H, W]
                - 值域: [0, 1]
                - H, W: 默认518x378
            selected_idx: 选择输出哪些block的特征
                - None: 输出所有block (长度=24)
                - [4, 11, 17, 23]: 输出4个指定block
                - 用于多尺度预测
            num_frame_for_scale: 尺度估计帧数
                - 前 N 帧使用双向注意力
                - 用于流式推理
            sliding_window_size: 滑动窗口大小
                - KV cache 最大帧数
                - 用于流式推理
            num_frame_per_block: 每个block处理的帧数
                - 1: 逐帧处理
                - num_frame_for_scale: scale frames一起处理

        Returns:
            tuple: (output_list, patch_start_idx)
                - output_list: 多尺度特征列表
                  - 每个元素: [B, S, P, 2C]
                  - P: 每帧token数
                  - 2C: frame + global 特征拼接
                - patch_start_idx: patch tokens起始索引
                  - = 6 (跳过 camera + register(4) + scale)
        """
        B, S_input, _, H, W = images.shape

        # ════════════════════════════════════════════════════════════════════
        # 步骤1: 图像嵌入 - DINOv2 Patch Embedding + Special Tokens
        # ════════════════════════════════════════════════════════════════════
        tokens, B, S_local, S_global, P, C = self._embed_images(
            images,
            num_frame_for_scale=num_frame_for_scale,
        )
        # tokens: [B*S, P, C]
        # S_local = S_global = S (无Context Parallel切片)

        # ════════════════════════════════════════════════════════════════════
        # 步骤2: 生成位置编码 - RoPE 位置信息
        # ════════════════════════════════════════════════════════════════════
        pos_local = self._get_positions(B, S_local, H, W, device=images.device)
        pos_global = self._get_positions(B, S_global, H, W, device=images.device)

        # ════════════════════════════════════════════════════════════════════
        # 步骤3: 交替执行 frame_blocks 和 global_blocks
        # ════════════════════════════════════════════════════════════════════
        frame_idx = 0    # frame block 累计索引
        global_idx = 0   # global block 累计索引
        output_list = []  # 输出特征列表

        # 循环执行 24 个 block_group (aa_block_num = depth / aa_block_size)
        for block_group_idx in range(self.aa_block_num):
            # 按 aa_order = ["frame", "global"] 的顺序执行
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    # 【帧内自注意力】
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S_local, P, C, frame_idx, pos=pos_local
                    )
                    # tokens 保持形状 [B*S, P, C]
                    # frame_intermediates: List[[B, S, P, C]]

                elif attn_type == "global":
                    # 【跨帧注意力】(子类实现具体逻辑)
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S_local, S_global, P, C, global_idx,
                        pos=pos_global,
                        # 流式推理参数
                        num_frame_for_scale=num_frame_for_scale,
                        sliding_window_size=sliding_window_size,
                        num_frame_per_block=num_frame_per_block,
                        image_height=H,
                        image_width=W,
                    )
                    # global_intermediates: List[[B, S, P, C]]

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # ══════════════════════════════════════════════════════════════════
            # 步骤4: 收集输出特征
            # ══════════════════════════════════════════════════════════════════
            # 只收集 selected_idx 指定的 block_group 输出
            if selected_idx is None or block_group_idx in selected_idx:
                for i in range(len(frame_intermediates)):
                    # 拼接 frame 和 global 特征
                    # frame特征包含空间信息，global特征包含时间信息
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    # 输出: [B, S, P, 2C]
                    output_list.append(concat_inter)

        # 返回多尺度特征列表和 patch 起始索引
        return output_list, self.patch_start_idx
