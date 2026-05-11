# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
CameraHead - 相机位姿预测头

本模块实现了 GCT 模型的相机位姿预测功能，采用迭代优化策略
逐步精化位姿估计。

【架构概览】

提供两种 CameraHead 实现:
1. CameraHead: 基础版本，使用标准 Transformer Block
2. CameraCausalHead: 因果版本，支持 KV Cache 的流式推理

【位姿编码格式】

pose_encoding_type = "absT_quaR_FoV":
- 9维位姿编码:
  - [:3]: 相机中心位置 (x, y, z) - 绝对平移
  - [3:7]: 四元数旋转 (qw, qx, qy, qz) - 四元数表示
  - [7:9]: 焦距和偏移 (focal_length, offset) - 视场角参数

【迭代优化策略】

┌─────────────────────────────────────────────────────────────────────┐
│                    CameraCausalHead 迭代优化流程                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入: pose_tokens [B, S, C] - camera token 特征                   │
│                                                                     │
│  for iteration in range(4):                                        │
│      ┌───────────────────────────────────────────────────────────┐ │
│      │ 步骤1: 初始化位姿编码                                        │ │
│      │   if iteration == 0:                                        │ │
│      │       module_input = embed_pose(empty_pose_tokens)         │ │
│      │   else:                                                       │ │
│      │       pred_pose_enc = pred_pose_enc.detach()               │ │
│      │       module_input = embed_pose(pred_pose_enc)             │ │
│      │                                                                  │ │
│      │   empty_pose_tokens:                                          │ │
│      │   - 可学习的初始位姿                                           │ │
│      │   - 用于第一次迭代的起点                                       │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   步骤2: 生成调制参数                                         │ │
│      │   │   modulation = poseLN_modulation(module_input)           │ │
│      │   │   shift_msa, scale_msa, gate_msa = modulation.chunk(3)  │ │
│      │   │                                                            │ │
│      │   │   调制参数的作用:                                          │ │
│      │   │   - shift_msa: 平移调制                                   │ │
│      │   │   - scale_msa: 缩放调制                                   │ │
│      │   │   - gate_msa: 门控调制（控制信息流动）                     │ │
│      │   │                                                            │ │
│      │   │   这类似于 DiT (Diffusion Transformer) 的 AdaLN         │ │
│      │   │   根据当前位姿估计动态调整特征                             │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   步骤3: 自适应 LayerNorm + 调制                              │ │
│      │   │   pose_tokens_norm = adaln_norm(pose_tokens)             │ │
│      │   │   pose_tokens_modulated = gate_msa *                     │ │
│      │   │       modulate(pose_tokens_norm, shift_msa, scale_msa)  │ │
│      │   │   pose_tokens_modulated = pose_tokens_modulated +        │ │
│      │   │       pose_tokens  # 残差连接                             │ │
│      │   │                                                            │ │
│      │   │   modulate 函数:                                          │ │
│      │   │   │   output = x * (1 + scale) + shift                   │ │
│      │   │                                                            │ │
│      │   │   残差连接保留原始 pose_tokens 信息                       │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   步骤4: CameraBlock 处理                                     │ │
│      │   │   for block in trunk:                                     │ │
│      │   │       pose_tokens_modulated = block(                     │ │
│      │   │           pose_tokens_modulated,                          │ │
│      │   │           pos=pos3d,          # 3D RoPE 位置             │ │
│      │   │           kv_cache=kv_cache,  # KV cache                 │ │
│      │   │           num_frames=S,       # 帧数                     │ │
│      │   │           ...                                                 │ │
│      │   │       )                                                        │ │
│      │   │                                                            │ │
│      │   │   CameraBlock 内部:                                        │ │
│      │   │   ┌─────────────────────────────────────────────────────┐ │ │
│      │   │   │ 1. LayerNorm                                           │ │ │
│      │   │   │ 2. 计算 Q, K, V                                         │ │ │
│      │   │   │ 3. 应用 3D RoPE (可选)                                  │ │ │
│      │   │   │ 4. 因果注意力:                                          │ │ │
│      │   │   │    - 与 KV cache 中历史 camera tokens 计算注意力     │ │ │
│      │   │   │    - 帧 t 只能看到帧 0~t                              │ │ │
│      │   │   │ 5. 输出投影 + 残差                                     │ │ │
│      │   │   │ 6. FFN + 残差                                          │ │ │
│      │   │   │ 7. 存储 KV 到 cache                                    │ │ │
│      │   │   └─────────────────────────────────────────────────────┘ │ │
│      │   │                                                            │ │
│      │   │   trunk_depth = 4 (4个 CameraBlock)                      │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   步骤5: 计算位姿增量                                         │ │
│      │   │   pose_tokens_modulated_norm = trunk_norm(...)           │ │
│      │   │   pred_pose_enc_delta = pose_branch(...)                 │ │
│      │   │                                                            │ │
│      │   │   pose_branch: MLP                                         │ │
│      │   │   │   Linear(C -> C//2) -> Linear(C//2 -> 9)            │ │
│      │   │   输出 9维位姿增量                                          │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   步骤6: 更新位姿编码                                         │ │
│      │   │   if pred_pose_enc is None:                               │ │
│      │   │       pred_pose_enc = pred_pose_enc_delta                 │ │
│      │   │   else:                                                       │ │
│      │   │       pred_pose_enc = pred_pose_enc + pred_pose_enc_delta │ │
│      │   │                                                            │ │
│      │   │   累积增量更新:                                             │ │
│      │   │   - iteration 0: pose = delta_0                           │ │
│      │   │   - iteration 1: pose = delta_0 + delta_1                 │ │
│      │   │   - iteration 2: pose = delta_0 + delta_1 + delta_2       │ │
│      │   │   - iteration 3: pose = delta_0 + delta_1 + delta_2 + delta_3 │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   步骤7: 激活函数                                              │ │
│      │   │   activated_pose = activate_pose(                        │ │
│      │   │       pred_pose_enc,                                      │ │
│      │   │       trans_act="linear",   # 平移激活                    │ │
│      │   │       quat_act="linear",   # 四元数激活                   │ │
│      │   │       fl_act="relu"         # 焦距激活(确保正值)          │ │
│      │   │   )                                                        │ │
│      │   │                                                            │ │
│      │   │   activate_pose:                                           │ │
│      │   │   │   trans [:3]: linear (无激活)                          │ │
│      │   │   │   quat [3:7]: linear + 四元数归一化                   │ │
│      │   │   │   fl   [7:9]: relu (确保焦距为正)                     │ │
│      │   │                                                            │ │
│      │   │   四元数归一化: quat = quat / ||quat||                   │ │
│      │   │   确保旋转表示的有效性                                       │ │
│      └───────────────────────────────────────────────────────────┘ │
│      │                                                              │ │
│      │   pred_pose_enc_list.append(activated_pose)                  │ │
│                                                                    │
│  输出: pred_pose_enc_list - 4次迭代的位姿预测列表                  │
│  │   - pred_pose_enc_list[-1] 是最终预测（最精确）                │
│  │   - 每次迭代逐步精化位姿估计                                    │
│                                                                    │
│  【迭代优化的优势】                                                 │
│                                                                    │
│  1. 逐步精化:                                                       │
│  │   - 从粗略估计开始                                              │
│  │   - 每次迭代增加细节                                            │
│  │   - 最终达到高精度                                              │
│                                                                    │
│  2. 信息流动:                                                       │
│  │   - 当前估计调制下一轮的特征处理                                │
│  │   - 实现自适应的特征关注                                        │
│                                                                    │
│  3. 残差更新:                                                       │
│  │   - 只预测增量 delta                                            │
│  │   - 累积更新避免大幅跳变                                        │
│  │   - 数值稳定性更好                                              │
│                                                                    │
│  【KV Cache 支持】                                                  │
│                                                                    │
│  CameraCausalHead 支持 KV Cache:                                   │
│  │   - kv_cache: 存储 4 次迭代各自的 KV                            │
│  │   - 每次迭代有独立的 KV cache                                    │
│  │   - frame_idx: 跟踪全局帧索引                                   │
│                                                                    │
│  cache 结构:                                                        │
│  │   kv_cache = [                                                   │
│  │       {  # iteration 0                                            │
│  │           "_skip_append": False,                                 │
│  │           "k_0": [B, 1, S_cached, 1, head_dim],                  │
│  │           "v_0": ...                                               │
│  │           ...                                                       │
│  │       },                                                            │
│  │       {  # iteration 1                                            │
│  │           ...                                                       │
│  │       },                                                            │
│  │       ...                                                           │
│  │   ]                                                               │
│                                                                    │
│  【3D RoPE 支持】                                                  │
│                                                                    │
│  CameraCausalHead 支持 3D RoPE:                                    │
│  │   - 编码时间位置 (frame index)                                  │
│  │   - WanRotaryPosEmbed                                            │
│  │   - patch_size=(max_frames, 1, 1)                                │
│  │   - 每帧只有1个 camera token                                     │
│                                                                    │
└─────────────────────────────────────────────────────────────────────┘

【位姿解码】

使用 pose_encoding_to_extri_intri() 将 9维编码转换为:
- extrinsics: [B, S, 3, 4] 外参矩阵 (world-to-camera)
- intrinsics: [B, S, 3, 3] 内参矩阵

extrinsics 结构:
│   [R | t]  其中 R 是旋转矩阵 (3x3), t 是平移向量 (3x1)
│   旋转由四元数转换得到

intrinsics 结构:
│   [f   0   cx]
│   [0   f   cy]
│   [0   0   1 ]
│   f: 焦距
│   cx, cy: 图像中心偏移

camera-to-world 变换:
│   c2w = closed_form_inverse_se3(extrinsics_4x4)
│   用于将相机坐标系点转换到世界坐标系
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from lingbot_map.layers import Mlp
from lingbot_map.layers.block import Block
from lingbot_map.layers.block import CameraBlock
from lingbot_map.heads.head_act import activate_pose
from lingbot_map.layers.rope import WanRotaryPosEmbed
from functools import partial
from torch.utils.checkpoint import checkpoint


class CameraHead(nn.Module):
    """
    【基础版相机预测头】不带 KV Cache 的批处理模式

    CameraHead 预测相机参数，使用迭代优化策略。
    它将一系列 Transformer Block ("trunk") 应用于专用的 camera tokens。

    【与 CameraCausalHead 的区别】

    | 特性 | CameraHead | CameraCausalHead |
    |------|------------|-------------------|
    | Block类型 | Block | CameraBlock |
    | KV Cache | 不支持 | 支持 |
    | 推理模式 | 批处理 | 流式推理 |
    | 3D RoPE | 不支持 | 支持 |
    | 滑动窗口 | 不支持 | 支持 |

    【适用场景】

    CameraHead 适合:
    - 离线批处理推理
    - 所有帧同时处理
    - 不需要流式推理

    【参数说明】

    dim_in: 输入维度 (2048 = 2*embed_dim)
    trunk_depth: trunk 深度 (4个 Block)
    pose_encoding_type: 位姿编码类型 ("absT_quaR_FoV")
    num_heads: 注意力头数 (16)
    mlp_ratio: MLP 扩展比例 (4)
    init_values: LayerScale 初始值 (0.01)
    trans_act: 平移激活 ("linear")
    quat_act: 四元数激活 ("linear")
    fl_act: 焦距激活 ("relu")
    """

    def __init__(
        self,
        dim_in: int = 2048,  # 输入维度 (frame + global 特征拼接)
        trunk_depth: int = 4,  # trunk 的 Block 数量
        pose_encoding_type: str = "absT_quaR_FoV",  # 位姿编码类型
        num_heads: int = 16,  # 注意力头数
        mlp_ratio: int = 4,  # MLP 扩展比例
        init_values: float = 0.01,  # LayerScale 初始值
        trans_act: str = "linear",  # 平移激活函数
        quat_act: str = "linear",  # 四元数激活函数
        fl_act: str = "relu",  # 焦距激活函数（确保正值）
        enable_ulysses_cp=False,  # Context Parallelism（已弃用）
    ):
        super().__init__()

        # 设置位姿编码类型
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9  # 9维位姿编码
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth

        self.enable_ulysses_cp = enable_ulysses_cp

        # ════════════════════════════════════════════════════════════════════
        # 构建 trunk: 4个标准 Transformer Block
        # ════════════════════════════════════════════════════════════════════
        self.trunk = nn.Sequential(
            *[
                Block(dim=dim_in, num_heads=num_heads, mlp_ratio=mlp_ratio, init_values=init_values)
                for _ in range(trunk_depth)
            ]
        )

        # ════════════════════════════════════════════════════════════════════
        # LayerNorm: 用于 camera token 和 trunk 输出
        # ════════════════════════════════════════════════════════════════════
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # ════════════════════════════════════════════════════════════════════
        # 可学习的空位姿 token
        # ════════════════════════════════════════════════════════════════════
        # 用于第一次迭代的初始化
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))

        # 将 9维位姿编码嵌入到 dim_in 维度
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # ════════════════════════════════════════════════════════════════════
        # 调制参数生成模块
        # ════════════════════════════════════════════════════════════════════
        # 生成 shift, scale, gate 三个调制参数
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # ════════════════════════════════════════════════════════════════════
        # 自适应 LayerNorm (无 affine 参数)
        # ════════════════════════════════════════════════════════════════════
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        # 位姿输出分支: MLP 输出 9维位姿增量
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

    def forward(self, aggregated_tokens_list: list, num_iterations: int = 4, **kwargs) -> list:
        """
        【前向传播】预测相机位姿参数

        处理流程:
        ┌────────────────────────────────────────────────────────────────────┐
        │                                                                     │
        │  输入: aggregated_tokens_list - 多尺度特征列表                     │
        │       │                                                              │
        │       │   使用最后一个 block 的特征                                  │
        │       ↓                                                              │
        │  提取 camera token                                                  │
        │  │   tokens = aggregated_tokens_list[-1]                           │
        │  │   pose_tokens = tokens[:, :, 0]  # 第一个token                 │
        │  │   pose_tokens = token_norm(pose_tokens)                        │
        │       │                                                              │
        │       ↓                                                              │
        │  迭代优化 (trunk_fn)                                                │
        │  │   for 4 iterations:                                              │
        │  │       - 初始化/更新位姿编码                                       │
        │  │       - 生成调制参数                                              │
        │  │       - AdaLN + 调制                                             │
        │  │       - trunk Block 处理                                         │
        │  │       - 计算位姿增量                                              │
        │  │       - 激活函数                                                  │
        │       │                                                              │
        │       ↓                                                              │
        │  输出: pred_pose_enc_list - 4次迭代的位姿列表                      │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            aggregated_tokens_list: Token tensors 列表
                - 每个元素: [B, S, P, 2C]
                - 使用最后一个 tensor 进行预测
            num_iterations: 迭代次数 (默认4)

        Returns:
            list: 各迭代步骤的位姿编码列表
                - 每个元素: [B, S, 9]
                - 最后一个是最终预测
        """
        # 使用最后一个 block 的 tokens 进行相机预测
        tokens = aggregated_tokens_list[-1]

        # 提取 camera token (第一个 token)
        pose_tokens = tokens[:, :, 0]
        # LayerNorm 标准化
        pose_tokens = self.token_norm(pose_tokens)

        # 迭代优化预测位姿
        pred_pose_enc_list = self.trunk_fn(pose_tokens, num_iterations)
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list:
        """
        【迭代优化核心】逐步精化相机位姿预测

        实现迭代优化策略，每次迭代:
        1. 使用当前位姿估计调制特征
        2. 通过 trunk Block 处理
        3. 计算位姿增量
        4. 累积更新位姿估计

        ╔════════════════════════════════════════════════════════════════════╗
        ║                    trunk_fn 迭代优化详解                            ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【初始化】                                                        ║
        ║                                                                    ║
        ║  pred_pose_enc = None                                              ║
        ║  pred_pose_enc_list = []                                           ║
        ║  B, S, C = pose_tokens.shape                                       ║
        ║                                                                    ║
        ║  【迭代循环】                                                      ║
        ║                                                                    ║
        ║  for iteration in range(4):                                        ║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤1: 准备输入                                               │║
        ║      │                                                               │║
        ║      │   第一次迭代:                                                   │║
        ║      │     module_input = embed_pose(empty_pose_tokens.expand(B,S,-1))│║
        ║      │     - 使用可学习的初始位姿                                     │║
        ║      │     - 作为优化的起点                                           │║
        ║      │                                                               │║
        ║      │   后续迭代:                                                     │║
        ║      │     pred_pose_enc = pred_pose_enc.detach()                    │║
        ║      │     module_input = embed_pose(pred_pose_enc)                  │║
        ║      │     - 使用上一轮的预测                                         │║
        ║      │     - detach 避免梯度回传                                     │║
        ║      │     - 根据当前估计调制特征                                     │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤2: 生成调制参数                                           │║
        ║      │                                                               │║
        ║      │   modulation = SiLU(module_input)                            │║
        ║      │   modulation = Linear(modulation)  # [B, S, 3*C]            │║
        ║      │                                                               │║
        ║      │   shift_msa, scale_msa, gate_msa = modulation.chunk(3)      │║
        ║      │   # 每个 [B, S, C]                                            │║
        ║      │                                                               │║
        ║      │   【调制参数的作用】                                          │║
        ║      │                                                               │║
        ║      │   shift_msa: 平移调制                                         │║
        ║      │   │   - 调整特征的基准值                                      │║
        ║      │   - 类似于条件生成中的条件偏移                                 │║
        ║      │                                                               │║
        ║      │   scale_msa: 缩放调制                                         │║
        ║      │   │   - 调整特征的幅度                                        │║
        ║      │   - 控制特征的敏感程度                                         │║
        ║      │                                                               │║
        ║      │   gate_msa: 门控调制                                          │║
        ║      │   │   - 控制调制的影响程度                                    │║
        ║      │   - gate_msa * modulated 表示加权                            │║
        ║      │   - 动态调整信息流动                                          │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤3: 自适应 LayerNorm                                       │║
        ║      │                                                               │║
        ║      │   # adaln_norm: LayerNorm without affine params              │║
        ║      │   pose_tokens_norm = adaln_norm(pose_tokens)                 │║
        ║      │                                                               │║
        ║      │   # modulate 函数                                             │║
        ║      │   def modulate(x, shift, scale):                              │║
        ║      │       return x * (1 + scale) + shift                          │║
        ║      │                                                               │║
        ║      │   pose_tokens_modulated = gate_msa *                          │║
        ║      │       modulate(pose_tokens_norm, shift_msa, scale_msa)        │║
        ║      │                                                               │║
        ║      │   # 残差连接                                                   │║
        ║      │   pose_tokens_modulated = pose_tokens_modulated + pose_tokens │║
        ║      │                                                               │║
        ║      │   【残差连接的作用】                                          │║
        ║      │   │   - 保留原始 pose_tokens 信息                             │║
        ║      │   - 调制只是增强，不是替换                                     │║
        ║      │   - 稳定训练过程                                              │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤4: Trunk Block 处理                                      │║
        ║      │                                                               │║
        ║      │   for block in trunk:                                        │║
        ║      │       pose_tokens_modulated = block(pose_tokens_modulated)   │║
        ║      │                                                               │║
        ║      │   【Block 内部结构】                                          │║
        ║      │                                                               │║
        ║      │   Block (标准 ViT Block):                                     │║
        ║      │   ┌───────────────────────────────────────────────────────┐  │║
        ║      │   │ 1. LayerNorm                                            │  │║
        ║      │   │ 2. Self-Attention                                       │  │║
        ║      │   │    Q, K, V = Linear(tokens)                              │  │║
        ║      │   │    Attn = softmax(Q @ K^T) @ V                           │  │║
        ║      │   │    注意力在 [B*S, C] 维度进行                             │  │║
        ║      │   │    每帧的 camera token 独立处理                          │  │║
        ║      │   │ 3. 输出投影 + 残差                                        │  │║
        ║      │   │ 4. LayerNorm                                              │  │║
        ║      │   │ 5. FFN                                                    │  │║
        ║      │   │    MLP: Linear(C -> 4C) -> GELU -> Linear(4C -> C)     │  │║
        ║      │   │ 6. LayerScale                                             │  │║
        ║      │   │    output = gamma * output                               │  │║
        ║      │   │    gamma 初始化为 0.01                                    │  │║
        ║      │   └───────────────────────────────────────────────────────┘  │║
        ║      │                                                               │║
        ║      │   trunk 有 4 个 Block                                          │║
        ║      │   逐步提取更高级的特征                                         │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤5: 计算位姿增量                                           │║
        ║      │                                                               │║
        ║      │   pose_tokens_norm = trunk_norm(pose_tokens_modulated)       │║
        ║      │                                                               │║
        ║      │   pred_pose_enc_delta = pose_branch(pose_tokens_norm)        │║
        ║      │                                                               │║
        ║      │   【pose_branch 结构】                                        │║
        ║      │   │   Mlp:                                                     │║
        ║      │   │     Linear(C -> C//2)                                      │║
        ║      │   │     Linear(C//2 -> 9)                                      │║
        ║      │   │                                                               │║
        ║      │   │   输出 9维位姿增量:                                        │║
        ║      │   │     delta[:3]: 平移增量                                    │║
        ║      │   │     delta[3:7]: 四元数增量                                 │║
        ║      │   │     delta[7:9]: 焦距增量                                   │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤6: 累积更新                                               │║
        ║      │                                                               │║
        ║      │   if pred_pose_enc is None:                                   │║
        ║      │       pred_pose_enc = pred_pose_enc_delta                     │║
        ║      │   else:                                                        │║
        ║      │       pred_pose_enc = pred_pose_enc + pred_pose_enc_delta     │║
        ║      │                                                               │║
        ║      │   【累积更新的优势】                                          │║
        ║      │                                                               │║
        ║      │   iteration 0: pose = delta_0                                 │║
        ║      │   iteration 1: pose = delta_0 + delta_1                       │║
        ║      │   iteration 2: pose = delta_0 + delta_1 + delta_2             │║
        ║      │   iteration 3: pose = delta_0 + delta_1 + delta_2 + delta_3   │║
        ║      │                                                               │║
        ║      │   每次迭代只预测增量:                                          │║
        ║      │   │   - 避免大幅跳变                                          │║
        ║      │   - 数值稳定性更好                                            │║
        ║      │   - 类似优化算法的迭代步                                       │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤7: 激活函数                                               │║
        ║      │                                                               │║
        ║      │   activated_pose = activate_pose(                            │║
        ║      │       pred_pose_enc,                                          │║
        ║      │       trans_act="linear",                                     │║
        ║      │       quat_act="linear",                                      │║
        ║      │       fl_act="relu"                                           │║
        ║      │   )                                                            │║
        ║      │                                                               │║
        ║      │   【activate_pose 分解】                                      │║
        ║      │                                                               │║
        ║      │   # 平移 [:3]                                                  │║
        ║      │   trans = pred_pose_enc[:, :, :3]                             │║
        ║      │   if trans_act == "linear":                                   │║
        ║      │       trans = trans  # 无激活                                  │║
        ║      │                                                               │║
        ║      │   # 四元数 [3:7]                                               │║
        ║      │   quat = pred_pose_enc[:, :, 3:7]                             │║
        ║      │   if quat_act == "linear":                                    │║
        ║      │       quat = quat                                              │║
        ║      │   # 四元数归一化                                               │║
        ║      │   quat = quat / torch.norm(quat, dim=-1, keepdim=True)        │║
        ║      │   # 确保旋转表示的有效性                                       │║
        ║      │                                                               │║
        ║      │   # 焦距 [7:9]                                                 │║
        ║      │   fl = pred_pose_enc[:, :, 7:9]                               │║
        ║      │   if fl_act == "relu":                                        │║
        ║      │       fl = F.relu(fl)  # 确保焦距为正                         │║
        ║      │                                                               │║
        ║      │   activated_pose = cat([trans, quat, fl], dim=-1)             │║
        ║      │                                                               │║
        ║      │   【为什么使用不同激活】                                       │║
        ║      │                                                               │║
        ║      │   trans: 无激活                                                │║
        ║      │   │   - 平移可以是任意值                                      │║
        ║      │   - 无约束                                                    │║
        ║      │                                                               │║
        ║      │   quat: 归一化                                                 │║
        ║      │   │   - 四元数必须满足 ||quat|| = 1                           │║
        ║      │   - 保证旋转矩阵的有效性                                       │║
        ║      │                                                               │║
        ║      │   fl: relu                                                     │║
        ║      │   │   - 焦距必须是正数                                         │║
        ║      │   - 物理约束                                                  │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      pred_pose_enc_list.append(activated_pose)                      │║
        ║                                                                    ║
        ║  【输出】                                                          ║
        ║                                                                    ║
        ║  pred_pose_enc_list: 4个迭代步骤的位姿预测                         │║
        ║  │   - [0]: 第一次迭代结果 (粗略估计)                               │║
        ║  │   - [1]: 第二次迭代结果                                          │║
        ║  │   - [2]: 第三次迭代结果                                          │║
        ║  │   - [3]: 第四次迭代结果 (最终精确预测)                           │║
        ║                                                                    ║
        ║  通常使用 pred_pose_enc_list[-1] 作为最终输出                     │║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            pose_tokens: 标准化的 camera tokens [B, S, C]
            num_iterations: 迭代次数 (默认4)

        Returns:
            list: 各迭代的激活位姿编码列表
                - 每个元素: [B, S, 9]
        """
        B, S, C = pose_tokens.shape  # S 预期为 1 或帧数
        pred_pose_enc = None
        pred_pose_enc_list = []

        for _ in range(num_iterations):
            # ══════════════════════════════════════════════════════════════════
            # 步骤1: 准备输入 - 初始化或使用上一轮预测
            # ══════════════════════════════════════════════════════════════════
            if pred_pose_enc is None:
                # 第一次迭代：使用可学习的空位姿 token
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # 后续迭代：使用上一轮预测（detach 避免梯度回传）
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # ══════════════════════════════════════════════════════════════════
            # 步骤2: 生成调制参数 - shift, scale, gate
            # ══════════════════════════════════════════════════════════════════
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # ══════════════════════════════════════════════════════════════════
            # 步骤3: 自适应 LayerNorm + 调制 + 残差
            # ══════════════════════════════════════════════════════════════════
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens  # 残差连接

            # ══════════════════════════════════════════════════════════════════
            # 步骤4: Trunk Block 处理 - 4个标准 Block
            # ══════════════════════════════════════════════════════════════════
            for block in self.trunk:
                pose_tokens_modulated = block(pose_tokens_modulated, enable_ulysses_cp=self.enable_ulysses_cp)

            # ══════════════════════════════════════════════════════════════════
            # 步骤5: 计算位姿增量 - MLP 输出 9维
            # ══════════════════════════════════════════════════════════════════
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            # ══════════════════════════════════════════════════════════════════
            # 步骤6: 累积更新位姿编码
            # ══════════════════════════════════════════════════════════════════
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # ══════════════════════════════════════════════════════════════════
            # 步骤7: 激活函数 - 平移、四元数归一化、焦距 relu
            # ══════════════════════════════════════════════════════════════════
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift


class CameraCausalHead(nn.Module):
    """
    【因果版相机预测头】支持 KV Cache 的流式推理

    CameraCausalHead 是 CameraHead 的因果版本，支持:
    - KV Cache: 流式推理的核心，避免重复计算
    - 3D RoPE: 时间位置编码
    - 滑动窗口: 控制注意力范围
    - Scale Frame: 初始帧的批量处理

    ╔════════════════════════════════════════════════════════════════════╗
    ║               CameraCausalHead vs CameraHead 对比                  ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  【核心区别】                                                      ║
    ║                                                                    ║
    ║  | 特性              | CameraHead        | CameraCausalHead      |║
    ║  |--------------------|-------------------|----------------------|║
    ║  | Block类型          | Block             | CameraBlock          |║
    ║  | 注意力机制         | 批量 Self-Attn    | 因果 FlashAttn       |║
    ║  | KV Cache           | 不支持            | 支持 (4个迭代独立)   |║
    ║  | 3D RoPE            | 不支持            | 支持 (时间位置)      |║
    ║  | Scale Frame        | 批量处理          | 批量处理 (双向注意力)|║
    ║  | 滑动窗口           | 不支持            | 支持                 |║
    ║  | 推理模式           | 离线批处理        | 流式在线推理         |║
    ║                                                                    ║
    ║  【CameraBlock 特性】                                              ║
    ║                                                                    ║
    ║  CameraBlock 相比标准 Block 有以下增强:                           ║
    ║                                                                    ║
    ║  1. KV Cache 支持:                                                 ║
    ║     │   - 存储 K, V 历史值                                         ║
    ║     │   - 新帧只需计算当前 Q                                       ║
    ║     │   - 与历史 KV 计算注意力                                     ║
    ║     │   - 避免重复计算历史帧                                       ║
    ║                                                                    ║
    ║  2. 因果注意力:                                                    ║
    ║     │   - 帧 t 只能看到帧 0~t                                     ║
    ║     │   - 不能看到未来帧                                           ║
    ║     │   - 保证流式推理的一致性                                     ║
    ║                                                                    ║
    ║  3. Scale Frame 处理:                                              ║
    ║     │   - 前 N 帧 (num_scale_frames=8)                            ║
    ║     │   - 使用双向注意力建立坐标系                                 ║
    ║     │   - 后续帧使用因果注意力                                     ║
    ║                                                                    ║
    ║  4. 滑动窗口:                                                      ║
    ║     │   - kv_cache_sliding_window=64                              ║
    ║     │   - 只保留最近64帧的 KV                                      ║
    ║     │   - 超出窗口的帧被驱逐                                       ║
    ║                                                                    ║
    ║  【KV Cache 结构】                                                 ║
    ║                                                                    ║
    ║  self.kv_cache = [                                                 ║
    ║      {  # iteration 0                                              ║
    ║          "_skip_append": False,  # 是否跳过追加                    ║
    ║          "k_0": [B, 1, S_cached, 1, head_dim],  # Block 0 的 K     ║
    ║          "v_0": [B, 1, S_cached, 1, head_dim],  # Block 0 的 V     ║
    ║          "k_1": ...,  # Block 1 的 K                               ║
    ║          "v_1": ...,  # Block 1 的 V                               ║
    ║          ...                                                        ║
    ║          "k_3": ...,  # Block 3 的 K (trunk_depth=4)              ║
    ║          "v_3": ...,  # Block 3 的 V                               ║
    ║      },                                                             ║
    ║      {  # iteration 1                                              ║
    ║          ...                                                        ║
    ║      },                                                             ║
    ║      {  # iteration 2                                              ║
    ║          ...                                                        ║
    ║      },                                                             ║
    ║      {  # iteration 3                                              ║
    ║          ...                                                        ║
    ║      },                                                             ║
    ║  ]                                                                 ║
    ║                                                                    ║
    ║  【为什么每个迭代有独立 KV Cache】                                 ║
    ║                                                                    ║
    ║  每次迭代:                                                         ║
    ║  │   1. pose_tokens_modulated = gate * modulate(pose_tokens)      ║
    ║  │   2. pose_tokens_modulated 通过 trunk                          ║
    ║                                                                    ║
    ║  由于调制参数 (shift, scale, gate) 每次迭代不同:                   ║
    ║  │   - pose_tokens_modulated 的值每轮都不同                       ║
    ║  │   - 导致每轮 K, V 的值也不同                                    ║
    ║  │   - 所以需要独立的 KV cache                                     ║
    ║                                                                    ║
    ║  【3D RoPE 位置编码】                                              ║
    ║                                                                    ║
    ║  WanRotaryPosEmbed 参数:                                          ║
    ║  │   - patch_size: (max_frame_num, 1, 1)                         ║
    ║  │   - 每帧只有 1 个 camera token                                  ║
    ║  │   - 时间维度变化，空间维度固定                                  ║
    ║                                                                    ║
    ║  pos3d 生成:                                                       ║
    ║  │   - 在流式模式下使用 frame_idx 跟踪全局位置                    ║
    ║  │   - f_start = frame_idx, f_end = frame_idx + S                 ║
    ║  │   - 保证连续帧的位置编码正确                                    ║
    ║                                                                    ║
    ║  【帧索引管理】                                                    ║
    ║                                                                    ║
    ║  self.frame_idx:                                                   ║
    ║  │   - 跟踪当前处理到的全局帧索引                                  ║
    ║  │   - 每次 forward 后更新: frame_idx += S                        ║
    ║  │   - 用于生成正确的 3D RoPE 位置                                 ║
    ║  │   - clean_kv_cache() 时重置为 0                                 ║
    ║                                                                    ║
    ╚══════════════════════════════════════════════════════════════════╝

    【参数说明】

    dim_in: 输入维度 (2048)
    trunk_depth: trunk 深度 (4个 CameraBlock)
    pose_encoding_type: 位姿编码类型 ("absT_quaR_FoV")
    num_heads: 注意力头数 (16)
    mlp_ratio: MLP 扩展比例 (4)
    init_values: LayerScale 初始值 (0.01)
    trans_act: 平移激活 ("linear")
    quat_act: 四元数激活 ("linear")
    fl_act: 焦距激活 ("relu")
    num_iterations: 迭代次数 (4)
    sliding_window_size: 注意力滑动窗口 (-1 表示不限制)
    kv_cache_sliding_window: KV Cache 滑动窗口 (64)
    kv_cache_scale_frames: Scale 帧数 (8)
    enable_3d_rope: 是否启用 3D RoPE (False)
    max_frame_num: 最大帧数 (1024)
    rope_theta: RoPE theta (10000.0)
    """

    def __init__(
        self,
        dim_in: int = 2048,  # 输入维度
        trunk_depth: int = 4,  # trunk 的 CameraBlock 数量
        pose_encoding_type: str = "absT_quaR_FoV",  # 位姿编码类型
        num_heads: int = 16,  # 注意力头数
        mlp_ratio: int = 4,  # MLP 扩展比例
        init_values: float = 0.01,  # LayerScale 初始值
        trans_act: str = "linear",  # 平移激活函数
        quat_act: str = "linear",  # 四元数激活函数
        fl_act: str = "relu",  # 焦距激活函数（确保正值）
        num_iterations = 4,  # 迭代次数
        elementwise_attn_output_gate: bool = False,  # 注意力输出门控
        sliding_window_size: int = -1,  # 注意力滑动窗口 (-1 不限制)
        attend_to_scale_frames: bool = False,  # 是否关注 scale frames
        num_random_frames: int = 0,  # 随机帧数
        enable_ulysses_cp: bool = False,  # Ulysses Context Parallelism
        attn_class: str = "flexflashattn_varlen",  # 注意力实现类
        # KV cache parameters
        kv_cache_sliding_window: int = 64,  # KV Cache 滑动窗口大小
        kv_cache_scale_frames: int = 8,  # Scale 帧数（前8帧双向注意力）
        kv_cache_cross_frame_special: bool = True,  # 跨帧特殊 token 处理
        kv_cache_include_scale_frames: bool = True,  # Scale 帧是否包含在 KV cache
        kv_cache_camera_only: bool = False,  # 是否只处理 camera token
        # 3D RoPE parameters
        enable_3d_rope: bool = False,  # 是否启用 3D RoPE
        max_frame_num: int = 1024,  # 最大帧数（用于 RoPE）
        rope_theta: float = 10000.0,  # RoPE theta 参数
    ):
        super().__init__()

        # ════════════════════════════════════════════════════════════════════
        # 设置位姿编码类型和目标维度
        # ════════════════════════════════════════════════════════════════════
        if pose_encoding_type == "absT_quaR_FoV":
            self.target_dim = 9  # 9维位姿编码
        else:
            raise ValueError(f"Unsupported camera encoding type: {pose_encoding_type}")

        self.trans_act = trans_act
        self.quat_act = quat_act
        self.fl_act = fl_act
        self.trunk_depth = trunk_depth
        self.sliding_window_size = sliding_window_size
        self.enable_ulysses_cp = enable_ulysses_cp
        self.num_heads = num_heads

        # ════════════════════════════════════════════════════════════════════
        # 3D RoPE 时间位置编码
        # ════════════════════════════════════════════════════════════════════
        # 用于编码时间位置，确保流式推理的一致性
        # Camera token: 每帧只有 1 个 token，时间维度变化
        self.enable_3d_rope = enable_3d_rope
        if enable_3d_rope:
            head_dim = dim_in // num_heads
            # Camera head 的 3D RoPE:
            # - patch_size=(max_frames, 1, 1): 时间维度最大帧数，空间维度为 1
            # - fhw_dim=[40, 44, 44]: 时间/高度/宽度维度分配
            #   - 40 用于时间维度
            #   - 44+44=88 用于空间维度（但空间只有1个token）
            self.rope3d = WanRotaryPosEmbed(
                attention_head_dim=head_dim,
                patch_size=(max_frame_num, 1, 1),
                theta=rope_theta,
                fhw_dim=[40, 44, 44],  # 维度分配
            )
        else:
            self.rope3d = None

        # ════════════════════════════════════════════════════════════════════
        # 构建 trunk: 4个 CameraBlock
        # ════════════════════════════════════════════════════════════════════
        # CameraBlock 相比标准 Block 增加了:
        # - KV Cache 支持
        # - 因果注意力
        # - Scale Frame 处理
        # - 滑动窗口
        self.trunk = nn.Sequential(
            *[
                CameraBlock(
                    dim=dim_in,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                    elementwise_attn_output_gate=elementwise_attn_output_gate,
                    sliding_window_size=sliding_window_size,
                    attend_to_scale_frames=attend_to_scale_frames,
                    num_random_frames=num_random_frames,
                    kv_cache_sliding_window=kv_cache_sliding_window,
                    kv_cache_scale_frames=kv_cache_scale_frames,
                    kv_cache_cross_frame_special=kv_cache_cross_frame_special,
                    kv_cache_include_scale_frames=kv_cache_include_scale_frames,
                    kv_cache_camera_only=kv_cache_camera_only,
                )
                for _ in range(trunk_depth)
            ]
        )

        # ════════════════════════════════════════════════════════════════════
        # LayerNorm: 用于 camera token 和 trunk 输出
        # ════════════════════════════════════════════════════════════════════
        self.token_norm = nn.LayerNorm(dim_in)
        self.trunk_norm = nn.LayerNorm(dim_in)

        # ════════════════════════════════════════════════════════════════════
        # 可学习的空位姿 token
        # ════════════════════════════════════════════════════════════════════
        # 用于第一次迭代的初始化
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
        # 将 9维位姿编码嵌入到 dim_in 维度
        self.embed_pose = nn.Linear(self.target_dim, dim_in)

        # ════════════════════════════════════════════════════════════════════
        # 调制参数生成模块
        # ════════════════════════════════════════════════════════════════════
        # SiLU -> Linear(dim_in -> 3*dim_in)
        # 输出 split 为: shift_msa, scale_msa, gate_msa
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True))

        # ════════════════════════════════════════════════════════════════════
        # 自适应 LayerNorm (无 affine 参数)
        # ════════════════════════════════════════════════════════════════════
        # 调制参数单独提供 shift 和 scale
        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        # 位姿输出分支: MLP 输出 9维位姿增量
        self.pose_branch = Mlp(in_features=dim_in, hidden_features=dim_in // 2, out_features=self.target_dim, drop=0)

        # ════════════════════════════════════════════════════════════════════
        # 流式推理状态变量
        # ════════════════════════════════════════════════════════════════════
        self.num_iterations = num_iterations
        self.kv_cache = None  # KV Cache (每个迭代独立)
        self.pos_cache = None  # 位置缓存（已弃用）
        self.frame_idx = 0  # 当前帧索引（用于 3D RoPE）
        self.cp_size = 1  # Context Parallelism 大小

        # ════════════════════════════════════════════════════════════════════
        # Ulysses Context Parallelism 设置
        # ════════════════════════════════════════════════════════════════════
        if self.enable_ulysses_cp:
            from torchtitan.distributed.sequence_parallel import (
            init_sequence_parallel,
            get_ulysses_sequence_parallel_rank,
            get_ulysses_sequence_parallel_world_size,
        )

            self.cp_size = get_ulysses_sequence_parallel_world_size()



    def clean_kv_cache(self):
        """
        【清理 KV Cache】

        清理 KV Cache 并重置帧索引，用于:
        - 开始新的推理序列
        - 释放 GPU 内存
        - 重置流式推理状态

        Args:
            无

        Effects:
            - 删除 self.kv_cache
            - 重置 self.kv_cache = None
            - 重置 self.frame_idx = 0
        """
        del self.kv_cache
        self.kv_cache = None
        self.frame_idx = 0

    def forward(self, aggregated_tokens_list: list, mask=None, num_iterations: int = 4, causal_inference=False, num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=None, **kwargs) -> list:
        """
        【前向传播】支持因果推理的相机位姿预测

        处理流程:
        ┌────────────────────────────────────────────────────────────────────┐
        │                                                                     │
        │  输入: aggregated_tokens_list - 多尺度特征列表                     │
        │       │                                                              │
        │       │   使用最后一个 block 的特征                                  │
        │       ↓                                                              │
        │  提取 camera token                                                  │
        │  │   tokens = aggregated_tokens_list[-1]                           │
        │  │   pose_tokens = tokens[:, :, 0]  # 第一个token                 │
        │  │   pose_tokens = token_norm(pose_tokens)                        │
        │       │                                                              │
        │       ↓                                                              │
        │  【因果推理模式判断】                                                │
        │  │   if causal_inference:                                          │
        │  │       if kv_cache is None:                                       │
        │  │           初始化 KV Cache                                        │
        │  │           │   kv_cache = [{k_0, v_0, ...}, ...]  # 4个迭代      │
        │  │           │   每个迭代存储 trunk_depth=4 个 Block 的 KV         │
        │       │                                                              │
        │       ↓                                                              │
        │  迭代优化 (trunk_fn)                                                │
        │  │   包含:                                                           │
        │  │   - 生成 3D RoPE 位置 (如果启用)                                 │
        │  │   - 4次迭代优化                                                   │
        │  │   - KV Cache 更新                                                 │
        │  │   - frame_idx 更新                                                │
        │       │                                                              │
        │       ↓                                                              │
        │  输出: pred_pose_enc_list - 4次迭代的位姿列表                      │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        【KV Cache 初始化详解】

        causal_inference=True 时，首次调用会初始化 KV Cache:

        kv_cache 结构:
        │   kv_cache = [                                                     │
        │       {  # iteration 0                                              │
        │           "_skip_append": False,                                    │
        │           "k_0": None,  # Block 0 的 K                              │
        │           "v_0": None,  # Block 0 的 V                              │
        │           "k_1": None,                                              │
        │           "v_1": None,                                              │
        │           "k_2": None,                                              │
        │           "v_2": None,                                              │
        │           "k_3": None,  # Block 3 (最后一个 Block)                 │
        │           "v_3": None,                                              │
        │       },                                                             │
        │       {  # iteration 1 - 相同结构                                   │
        │           ...                                                        │
        │       },                                                             │
        │       {  # iteration 2                                              │
        │           ...                                                        │
        │       },                                                             │
        │       {  # iteration 3                                              │
        │           ...                                                        │
        │       },                                                             │
        │   ]                                                                 │

        _skip_append 标志:
        │   - False: 正常追加 KV
        │   - True: 跳过追加（用于特定场景）

        Args:
            aggregated_tokens_list: Token tensors 列表
                - 每个元素: [B, S, P, 2C]
                - 使用最后一个 tensor 进行预测
            mask: 视频掩码 (可选)
            num_iterations: 迭代次数 (默认4)
            causal_inference: 是否启用因果推理 (流式推理时为 True)
            num_frame_per_block: 每个 Block 处理的帧数 (默认1)
            num_frame_for_scale: Scale 帧数 (-1 表示自动)
            sliding_window_size: 滑动窗口大小 (覆盖默认值)

        Returns:
            list: 各迭代步骤的位姿编码列表
                - 每个元素: [B, S, 9]
                - 最后一个是最终预测
        """
        # 使用传入的滑动窗口大小，或默认值
        effective_sliding_window_size = sliding_window_size if sliding_window_size is not None else self.sliding_window_size

        # 使用最后一个 block 的 tokens 进行相机预测
        tokens = aggregated_tokens_list[-1]

        # 提取 camera token (第一个 token)
        pose_tokens = tokens[:, :, 0]
        # LayerNorm 标准化
        pose_tokens = self.token_norm(pose_tokens)

        # ════════════════════════════════════════════════════════════════════
        # 因果推理模式: 初始化 KV Cache
        # ════════════════════════════════════════════════════════════════════
        if causal_inference:
            if self.kv_cache is None:
                # 首次调用，初始化 KV Cache
                self.kv_cache = []
                for i in range(self.num_iterations):
                    # 每个迭代创建一个独立的 cache dict
                    self.kv_cache.append({"_skip_append": False})
                    for j in range(self.trunk_depth):
                        # 初始化每个 Block 的 K, V 为 None
                        self.kv_cache[i][f"k_{j}"] = None
                        self.kv_cache[i][f"v_{j}"] = None

        # 迭代优化预测位姿
        pred_pose_enc_list = self.trunk_fn(
            pose_tokens,
            mask,
            num_iterations,
            num_frame_per_block=num_frame_per_block,
            num_frame_for_scale=num_frame_for_scale,
            sliding_window_size=effective_sliding_window_size
        )
        return pred_pose_enc_list

    def trunk_fn(self, pose_tokens: torch.Tensor, mask=None, num_iterations: int=4, num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=None) -> list:
        """
        【迭代优化核心】逐步精化相机位姿预测（支持 KV Cache）

        与 CameraHead.trunk_fn 的核心区别:
        - 使用 CameraBlock 而非标准 Block
        - 支持 KV Cache 流式推理
        - 支持 3D RoPE 时间位置编码
        - 支持 Scale Frame 批量处理

        ╔════════════════════════════════════════════════════════════════════╗
        ║            CameraCausalHead.trunk_fn 流程详解                      ║
        ╠════════════════════════════════════════════════════════════════════╣
        ║                                                                    ║
        ║  【初始化】                                                        ║
        ║                                                                    ║
        ║  B, S, C = pose_tokens.shape                                       ║
        ║  pred_pose_enc = None                                              ║
        ║  pred_pose_enc_list = []                                           ║
        ║                                                                    ║
        ║  【Scale Frame 判断】                                              ║
        ║                                                                    ║
        ║  is_scale_frames = (kv_cache 存在 且 frame_idx == 0)              ║
        ║                                                                    ║
        ║  Scale Frame:                                                      ║
        ║  │   - 前 N 帧 (num_scale_frames=8)                                ║
        ║  │   - frame_idx == 0 表示处理第一批帧                             ║
        ║  │   - Scale Frame 使用批量模式注意力                              ║
        ║  │   - 不使用因果注意力，建立坐标系                                 ║
        ║                                                                    ║
        ║  【3D RoPE 位置生成】                                              ║
        ║                                                                    ║
        ║  if enable_3d_rope:                                                ║
        ║      if kv_cache 存在:                                             ║
        ║          # 流式模式: 使用 frame_idx 跟踪全局位置                   ║
        ║          f_start = frame_idx                                       ║
        ║          f_end = frame_idx + S                                     ║
        ║      else:                                                         ║
        ║          # 批处理模式: 从 0 开始                                   ║
        ║          f_start = 0                                               ║
        ║          f_end = None                                              ║
        ║                                                                    ║
        ║      pos3d = rope3d(ppf=S, pph=1, ppw=1, ...)                      ║
        ║      # 返回 [1, 1, S, head_dim//2] 复数                            ║
        ║                                                                    ║
        ║  【迭代循环】                                                      ║
        ║                                                                    ║
        ║  for i in range(4):                                                ║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤1: 准备输入                                               │║
        ║      │                                                               │║
        ║      │   第一次迭代:                                                   │║
        ║      │     module_input = embed_pose(empty_pose_tokens.expand(B,S,-1))│║
        ║      │                                                               │║
        ║      │   后续迭代:                                                     │║
        ║      │     pred_pose_enc = pred_pose_enc.detach()                    │║
        ║      │     module_input = embed_pose(pred_pose_enc)                  │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤2: 生成调制参数                                           │║
        ║      │                                                               │║
        ║      │   modulation = SiLU(module_input)                            │║
        ║      │   modulation = Linear(modulation)  # [B, S, 3*C]            │║
        ║      │                                                               │║
        ║      │   shift_msa, scale_msa, gate_msa = modulation.chunk(3)      │║
        ║      │   # 每个 [B, S, C]                                            │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤3: AdaLN + 调制 + 残差                                    │║
        ║      │                                                               │║
        ║      │   pose_tokens_modulated = gate_msa *                          │║
        ║      │       modulate(adaln_norm(pose_tokens), shift_msa, scale_msa) │║
        ║      │                                                               │║
        ║      │   pose_tokens_modulated = pose_tokens_modulated + pose_tokens │║
        ║      │   # 残差连接                                                   │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤4: CameraBlock 处理                                       │║
        ║      │                                                               │║
        ║      │   for idx in range(trunk_depth):                               │║
        ║      │       pose_tokens_modulated = trunk[idx](                     │║
        ║      │           pose_tokens_modulated,                               │║
        ║      │           pos=pos3d,               # 3D RoPE 位置             │║
        ║      │           video_mask=mask,         # 视频掩码                 │║
        ║      │           num_frames=S*cp_size,   # 帧数                     │║
        ║      │           frame_seqlen=1,         # 每帧 token 数            │║
        ║      │           kv_cache=kv_cache[i],   # 当前迭代的 KV Cache     │║
        ║      │           global_idx=idx,         # Block 索引               │║
        ║      │           ...                                                 │║
        ║      │       )                                                        │║
        ║      │                                                               │║
        ║      │   【CameraBlock 内部】                                        │║
        ║      │                                                               │║
        ║      │   CameraBlock 处理流程:                                       │║
        ║      │   ┌─────────────────────────────────────────────────────────┐║║
        ║      │   │ 1. LayerNorm                                               │║║
        ║      │   │ 2. 计算 Q, K, V                                             │║║
        ║      │   │ 3. 应用 3D RoPE (如果启用)                                  │║║
        ║      │   │ 4. 因果注意力计算:                                          │║║
        ║      │   │    │   - Scale Frame: 批量双向注意力                      │║║
        ║      │   │    │   - 非 Scale Frame: 与 KV cache 计算因果注意力       │║║
        ║      │   │    │   - 帧 t 只能看到帧 0~t                              │║║
        ║      │   │ 5. 输出投影 + 残差                                           │║║
        ║      │   │ 6. FFN + 残差                                                │║║
        ║      │   │ 7. 更新 KV Cache                                            │║║
        ║      │   │    │   - kv_cache[i]["k_{idx}"] = K                       │║║
        ║      │   │    │   - kv_cache[i]["v_{idx}"] = V                       │║║
        ║      │   └─────────────────────────────────────────────────────────┘║║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤5: 计算位姿增量                                           │║
        ║      │                                                               │║
        ║      │   pose_tokens_norm = trunk_norm(pose_tokens_modulated)       │║
        ║      │   pred_pose_enc_delta = pose_branch(pose_tokens_norm)        │║
        ║      │                                                               │║
        ║      │   pose_branch: MLP                                            │║
        ║      │   │   Linear(C -> C//2) -> Linear(C//2 -> 9)                  │║
        ║      │   输出 9维位姿增量                                              │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤6: 累积更新                                               │║
        ║      │                                                               │║
        ║      │   if pred_pose_enc is None:                                   │║
        ║      │       pred_pose_enc = pred_pose_enc_delta                     │║
        ║      │   else:                                                        │║
        ║      │       pred_pose_enc = pred_pose_enc + pred_pose_enc_delta     │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║      ┌─────────────────────────────────────────────────────────────┐║
        ║      │ 步骤7: 激活函数                                               │║
        ║      │                                                               │║
        ║      │   activated_pose = activate_pose(                            │║
        ║      │       pred_pose_enc,                                          │║
        ║      │       trans_act="linear",                                     │║
        ║      │       quat_act="linear",                                      │║
        ║      │       fl_act="relu"                                           │║
        ║      │   )                                                            │║
        ║      │                                                               │║
        ║      │   # 平移: 无激活                                               │║
        ║      │   # 四元数: 归一化                                             │║
        ║      │   # 焦距: relu (确保正值)                                      │║
        ║      │                                                               │║
        ║      │   pred_pose_enc_list.append(activated_pose)                  │║
        ║      └─────────────────────────────────────────────────────────────┘║
        ║                                                                    ║
        ║  【帧索引更新】                                                    ║
        ║                                                                    ║
        ║  if kv_cache 存在:                                                 ║
        ║      frame_idx += S                                                 ║
        ║      # 跟踪全局帧位置，用于下一次 3D RoPE 生成                      ║
        ║                                                                    ║
        ║  【输出】                                                          ║
        ║                                                                    ║
        ║  pred_pose_enc_list: 4个迭代的位姿预测                             ║
        ║  │   - [0]: 第一次迭代 (粗略估计)                                   ║
        ║  │   - [1]: 第二次迭代                                              ║
        ║  │   - [2]: 第三次迭代                                              ║
        ║  │   - [3]: 第四次迭代 (最终精确预测)                               ║
        ║                                                                    ║
        ╚══════════════════════════════════════════════════════════════════╝

        Args:
            pose_tokens: 标准化的 camera tokens [B, S, C]
            mask: 视频掩码 (可选)
            num_iterations: 迭代次数 (默认4)
            num_frame_per_block: 每个 Block 处理的帧数
            num_frame_for_scale: Scale 帧数 (-1 表示自动)
            sliding_window_size: 滑动窗口大小

        Returns:
            list: 各迭代的激活位姿编码列表
                - 每个元素: [B, S, 9]
        """
        B, S, C = pose_tokens.shape
        pred_pose_enc = None
        pred_pose_enc_list = []

        # ════════════════════════════════════════════════════════════════════
        # Scale Frame 判断
        # ════════════════════════════════════════════════════════════════════
        # Scale Frame 是前 N 帧 (num_scale_frames=8)
        # - frame_idx == 0 表示这是第一批帧
        # - Scale Frame 使用批量模式注意力（双向注意力）
        # - 后续帧使用因果注意力
        is_scale_frames = (self.kv_cache is not None and self.frame_idx == 0)

        # ════════════════════════════════════════════════════════════════════
        # 生成 3D RoPE 位置 (如果启用)
        # ════════════════════════════════════════════════════════════════════
        # 用于编码时间位置，确保流式推理的一致性
        pos3d = None
        if self.rope3d is not None:
            # Camera tokens: 每帧只有 1 个 token
            # 位置为 (f, 0, 0) - 时间维度变化，空间维度固定

            # 流式模式下使用 frame_idx 跟踪全局位置
            # 批处理模式下从 0 开始
            if self.kv_cache is not None:
                f_start = self.frame_idx
                f_end = self.frame_idx + S
            else:
                f_start = 0
                f_end = None  # 使用 ppf 作为帧数

            pos3d = self.rope3d(
                ppf=S * self.cp_size,  # 总帧数（考虑 Context Parallelism）
                pph=1,                  # height = 1 (camera token)
                ppw=1,                  # width = 1 (camera token)
                patch_start_idx=0,      # 无前置特殊 token
                device=pose_tokens.device,
                f_start=f_start,
                f_end=f_end,
            )
            # 返回 [1, 1, S*cp_size, head_dim//2] 复数张量

        for i in range(num_iterations):
            # ══════════════════════════════════════════════════════════════════
            # 步骤1: 准备输入 - 初始化或使用上一轮预测
            # ══════════════════════════════════════════════════════════════════
            if pred_pose_enc is None:
                # 第一次迭代：使用可学习的空位姿 token
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                # 后续迭代：使用上一轮预测（detach 避免梯度回传）
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            # ══════════════════════════════════════════════════════════════════
            # 步骤2: 生成调制参数 - shift, scale, gate
            # ══════════════════════════════════════════════════════════════════
            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)

            # ══════════════════════════════════════════════════════════════════
            # 步骤3: AdaLN + 调制 + 残差
            # ══════════════════════════════════════════════════════════════════
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens  # 残差连接

            # ══════════════════════════════════════════════════════════════════
            # 步骤4: CameraBlock 处理 - 支持 KV Cache 和 3D RoPE
            # ══════════════════════════════════════════════════════════════════
            for idx in range(self.trunk_depth):
                pose_tokens_modulated = self.trunk[idx](
                    pose_tokens_modulated,
                    pos=pos3d,                     # 3D RoPE 位置编码
                    video_mask=mask,               # 视频掩码
                    num_frames=S * self.cp_size,   # 帧数
                    frame_seqlen=1,                # 每帧 token 数 (camera: 1)
                    kv_cache=self.kv_cache[i] if self.kv_cache is not None else None,  # 当前迭代的 KV Cache
                    global_idx=idx,                # Block 索引
                    num_frame_per_block=num_frame_per_block,
                    num_frame_for_scale=num_frame_for_scale,
                    sliding_window_size=sliding_window_size,
                    enable_ulysses_cp=self.enable_ulysses_cp,
                    enable_3d_rope=self.enable_3d_rope,
                    is_scale_frames=is_scale_frames,  # 是否为 Scale Frame
                )

            # ══════════════════════════════════════════════════════════════════
            # 步骤5: 计算位姿增量 - MLP 输出 9维
            # ══════════════════════════════════════════════════════════════════
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            # ══════════════════════════════════════════════════════════════════
            # 步骤6: 累积更新位姿编码
            # ══════════════════════════════════════════════════════════════════
            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta

            # ══════════════════════════════════════════════════════════════════
            # 步骤7: 激活函数 - 平移、四元数归一化、焦距 relu
            # ══════════════════════════════════════════════════════════════════
            activated_pose = activate_pose(
                pred_pose_enc, trans_act=self.trans_act, quat_act=self.quat_act, fl_act=self.fl_act
            )
            pred_pose_enc_list.append(activated_pose)

        # ════════════════════════════════════════════════════════════════════
        # 帧索引更新（流式模式）
        # ════════════════════════════════════════════════════════════════════
        # 跟踪全局帧位置，用于下一次 3D RoPE 生成
        if self.kv_cache is not None:
            self.frame_idx += S

        return pred_pose_enc_list


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Modulate the input tensor using scaling and shifting parameters.
    """
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift




class CameraDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            Block(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        B, V, P, C = hidden.shape
        hidden = hidden.view(hidden.shape[0]*hidden.shape[1], hidden.shape[2], hidden.shape[3])
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, pos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, pos=xpos)
        out = self.linear_out(hidden).view(B, V, P, -1)
        
        return out
