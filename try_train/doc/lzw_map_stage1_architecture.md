# LZW-Map Stage1 模型结构说明与 LingBot-Map 对比

本文档说明当前仓库中新增的 `lzw-map-stage1` 轻量模型结构，并和原始 `lingbot-map` / `GCTStream` Stage1 结构做对比。

相关代码入口：

```text
lingbot_map/models/gct_stream.py
lingbot_map/models/lzw_map.py
try_train/print_lzw_map_stage1_params.py
try_train/train_replica_gct.py
```

## 1. 设计目标

`lzw-map` 的目标是在保留原始 LingBot-Map 几何建模框架的基础上，降低完整模型的参数量和训练开销，用于后续 Stage1 训练。

当前 Stage1 版本做了三处结构轻量化：

| 部分 | 原始 LingBot-Map | LZW-Map Stage1 | 说明 |
|---|---:|---:|---|
| Backbone | `dinov2_vitl14_reg` | `dinov2_vitb14_reg` | 从 ViT-L 换为 ViT-B |
| `embed_dim` | 1024 | 768 | token width 降低 |
| Aggregator depth | 24 | 12 | frame/global blocks 深度减半 |
| Camera head trunk | 4 | 2 | 位姿预测头 Transformer 深度减半 |
| Depth head | 原 DPTHead | 原 DPTHead | 不做通道减半，仅 `dim_in` 随 `embed_dim` 变小 |
| Point head | 默认启用 | Stage1 默认关闭 | Stage1 只训练 depth + pose 时不需要 world-point head |

注意：LZW-Map 的 `depth_head` 没有按 `features/out_channels` 减半，仍使用原始 DPT 解码结构。不过由于输入特征维度从 `2*1024=2048` 变为 `2*768=1536`，`LayerNorm` 和 1x1 projection 的参数量会自然减少一点。

## 2. LZW-Map Stage1 结构

默认构造入口：

```python
from lingbot_map.models.lzw_map import create_lzw_map_stage1

model = create_lzw_map_stage1(
    pretrained_path="pretrained/dinov2_vitb14_reg4_pretrain.pth",
    enable_point=False,
    use_sdpa=True,
)
```

模型结构：

```text
LZWMapStage1 / GCTStream
├── aggregator: AggregatorStream
│   ├── patch_embed: DINOv2 ViT-B/14 with 4 registers
│   │   ├── embed_dim = 768
│   │   ├── depth = 12
│   │   └── num_heads = 12
│   ├── special tokens
│   │   ├── camera_token
│   │   ├── register_token
│   │   └── scale_token
│   ├── frame_blocks: 12 个帧内 Transformer Block
│   └── global_blocks: 12 个跨帧 Transformer Block, 支持 SDPA / FlashInfer KV cache
├── camera_head: CameraCausalHead
│   ├── dim_in = 1536
│   ├── num_heads = 12
│   ├── trunk_depth = 2
│   ├── poseLN_modulation
│   ├── pose_branch
│   └── 默认 4 次 iterative pose refinement
├── depth_head: DPTHead
│   ├── dim_in = 1536
│   ├── features = 256
│   ├── out_channels = [256, 512, 1024, 1024]
│   ├── output_dim = 2
│   └── activation = exp
└── point_head: None by default
```

关键默认参数：

| 参数 | LZW-Map Stage1 默认值 |
|---|---:|
| `img_size` | 518 |
| `patch_size` | 14 |
| `patch_embed` | `dinov2_vitb14_reg` |
| `embed_dim` | 768 |
| `aggregator_depth` | 12 |
| `selected_idx` | `[2, 5, 8, 11]` |
| `camera_trunk_depth` | 2 |
| `camera_num_heads` | 12 |
| `enable_depth` | True |
| `enable_camera` | True |
| `enable_point` | False |
| `use_sdpa` | True |

`selected_idx=[2,5,8,11]` 是 12 层 aggregator 的四个 feature taps，对应浅层、中浅层、中深层、最后层。原始 24 层模型使用 `[4,11,17,23]`。DPTHead 仍需要四层特征输入，因此这里不能继续使用原始的 `[4,11,17,23]`。

## 3. Stage1 前向流程

当前 Stage1 训练逻辑仍沿用原始 GCTStream 的 Base Model Training 设置：

```text
images [B, S, 3, H, W]
  │
  ├── ImageNet / ResNet mean-std 归一化
  │
  ├── DINOv2 ViT-B patch_embed
  │
  ├── 添加 camera/register/scale special tokens
  │
  ├── 12 层 frame attention + global attention 交替聚合
  │     └── selected_idx = [2,5,8,11] 输出四层 token features
  │
  ├── CameraCausalHead
  │     └── 输出 pose_enc [B, S, 9]
  │
  └── DPTHead
        └── 输出 depth [B, S, H, W, 1], depth_conf [B, S, H, W]
```

在 `try_train/train_replica_gct.py` 的 Stage1 helper 中：

```text
num_frame_for_scale = S
sliding_window_size = -1
num_frame_per_block = S
causal_inference = False
```

也就是说，Stage1 把一个 sample 中的所有 views 作为一个 multi-view set 做全局注意力，而不是逐帧 streaming inference。

## 4. 参数量统计

统计命令：

```bash
/home/lizhengwu/miniconda3/envs/lingbot-map/bin/python try_train/print_lzw_map_stage1_params.py
```

如果只想看结构参数量，不加载 DINOv2 预训练权重：

```bash
/home/lizhengwu/miniconda3/envs/lingbot-map/bin/python try_train/print_lzw_map_stage1_params.py --no_pretrained
```

### 4.1 LZW-Map Stage1 默认参数量

默认 `enable_point=False`：

| 模块 | 参数量 | FP32 大小 |
|---|---:|---:|
| total | 352,916,980 | 1346.27 MiB |
| `aggregator.total` | 256,744,704 | 979.40 MiB |
| `aggregator.patch_embed_backbone` | 86,583,552 | 330.29 MiB |
| `aggregator.frame_blocks` | 85,075,968 | 324.54 MiB |
| `aggregator.global_blocks` | 85,075,968 | 324.54 MiB |
| `aggregator.special_tokens` | 9,216 | 0.04 MiB |
| `camera_head` | 64,960,530 | 247.80 MiB |
| `depth_head` | 31,211,746 | 119.06 MiB |
| `point_head` | 0 | 0 |

可训练参数默认几乎等于总参数，只有 DINOv2 `mask_token` 被禁用梯度：

```text
total params     = 352,916,980
trainable params = 352,916,212
frozen params    = 768
```

如果 `--enable_point` 打开 world-point head：

| 模块 | 参数量 |
|---|---:|
| total | 384,128,792 |
| `point_head` | 31,211,812 |

## 5. 与原始 LingBot-Map Stage1 结构对比

原始 LingBot-Map 默认结构：

```text
GCTStream
├── aggregator.patch_embed: DINOv2 ViT-L/14 reg, embed_dim=1024, depth=24
├── aggregator.frame_blocks: 24
├── aggregator.global_blocks: 24
├── camera_head: dim_in=2048, trunk_depth=4, num_heads=16
├── depth_head: DPTHead, dim_in=2048
└── point_head: DPTHead, dim_in=2048
```

### 5.1 结构差异

| 项 | 原始 LingBot-Map | LZW-Map Stage1 | 变化 |
|---|---:|---:|---:|
| DINOv2 backbone | ViT-L/14 reg | ViT-B/14 reg | backbone 缩小 |
| `embed_dim` | 1024 | 768 | 75% |
| DINO backbone depth | 24 | 12 | 50% |
| Aggregator frame blocks | 24 | 12 | 50% |
| Aggregator global blocks | 24 | 12 | 50% |
| Aggregator heads | 16 | 12 | 75% |
| Camera `dim_in` | 2048 | 1536 | 75% |
| Camera trunk depth | 4 | 2 | 50% |
| Camera heads | 16 | 12 | 75% |
| Depth DPT channels | `[256,512,1024,1024]` | `[256,512,1024,1024]` | 不减半 |
| Stage1 feature taps | `[4,11,17,23]` | `[2,5,8,11]` | 适配 12 层 aggregator |

### 5.2 参数量差异

这里按 Stage1 depth+pose 路径比较，即默认不启用 point head。

| 模块 | 原始 LingBot-Map | LZW-Map Stage1 | LZW / 原始 |
|---|---:|---:|---:|
| total, no point head | 1,157,943,540 | 352,916,980 | 30.48% |
| aggregator total | 909,114,368 | 256,744,704 | 28.24% |
| patch_embed backbone | 304,372,736 | 86,583,552 | 28.45% |
| frame_blocks | 302,364,672 | 85,075,968 | 28.14% |
| global_blocks | 302,364,672 | 85,075,968 | 28.14% |
| camera_head | 216,174,610 | 64,960,530 | 30.05% |
| depth_head | 32,654,562 | 31,211,746 | 95.58% |

结论：

1. LZW-Map Stage1 在不启用 point head 时，总参数约为原始 depth+pose Stage1 路径的 30.5%。
2. 参数减少主要来自 backbone width/depth、aggregator depth、camera trunk depth。
3. Depth head 基本保持原结构，因此参数量只小幅下降。
4. 如果和原始完整默认模型比较，也就是原始模型包含 point head 的 `1,190,598,168` 参数，LZW-Map Stage1 默认约为 29.64%。

### 5.3 Head-only / 冻结 aggregator 训练时的差异

如果继续沿用 `train_replica_gct.py` 的 head-only 策略，也就是冻结 aggregator，只训练 camera head + depth head：

| 训练参数 | 原始 LingBot-Map | LZW-Map Stage1 | LZW / 原始 |
|---|---:|---:|---:|
| camera + depth | 248,829,172 | 96,172,276 | 38.65% |
| camera_head | 216,174,610 | 64,960,530 | 30.05% |
| depth_head | 32,654,562 | 31,211,746 | 95.58% |

也就是说，即使只做 head-only Stage1，LZW-Map 也能把实际训练参数从约 248.8M 降到约 96.2M。

## 6. 预训练初始化

默认使用本地 DINOv2 ViT-B reg4 权重：

```text
pretrained/dinov2_vitb14_reg4_pretrain.pth
```

当 `pretrained_path` 非空时：

1. `aggregator.patch_embed` 会加载 ViT-B DINOv2 权重。
2. `AggregatorBase._init_blocks_from_dino()` 会把 DINOv2 的 12 个 block 权重复制初始化到：
   - `aggregator.frame_blocks`
   - `aggregator.global_blocks`
3. `camera_head` 和 `depth_head` 仍为新结构随机初始化，后续通过 Stage1 训练学习。

本地验证日志会出现：

```text
Found 12 blocks in DINO checkpoint
Frame block 0: Missing keys: 4, Unexpected keys: 0
Global block 0: Missing keys: 4, Unexpected keys: 0
```

这里的 missing keys 主要来自 GCT block 中额外的 LayerScale / qk norm 等结构差异，属于当前 `strict=False` 初始化路径的预期现象。

## 7. 当前实现状态

已完成：

| 项 | 状态 |
|---|---|
| `GCTStream` 暴露 `aggregator_depth` | 已完成 |
| `GCTStream` 暴露 `selected_idx` | 已完成 |
| `GCTStream` 暴露 `camera_trunk_depth` | 已完成 |
| `GCTStream` 暴露 `camera_num_heads` | 已完成 |
| `LZWMapStage1` 构造器 | 已完成 |
| 参数量打印脚本 | 已完成 |
| Stage1 helper 读取 `model.selected_idx` | 已完成 |
| 小尺寸 forward smoke test | 已通过 |

验证过的 smoke test：

```text
input images: [1, 2, 3, 28, 28]
pose_enc:    [1, 2, 9]
depth:       [1, 2, 28, 28, 1]
depth_conf:  [1, 2, 28, 28]
```

## 8. 训练接入建议

### 8.1 新建 LZW-Map Stage1 训练脚本

建议不要直接覆盖原始 `train_replica_gct.py`，而是新增独立脚本，例如：

```text
try_train/train_lzw_map_stage1.py
```

训练脚本中使用：

```python
from lingbot_map.models.lzw_map import create_lzw_map_stage1

model = create_lzw_map_stage1(
    pretrained_path="pretrained/dinov2_vitb14_reg4_pretrain.pth",
    enable_point=False,
    use_sdpa=True,
)
```

### 8.2 推荐的 Stage1 训练模式

如果目标是训练完整 student：

```text
trainable:
  aggregator.patch_embed
  aggregator.frame_blocks
  aggregator.global_blocks
  camera_head
  depth_head

frozen:
  DINOv2 mask_token
```

如果目标是先做较稳的 warmup：

```text
Phase A: 冻结 aggregator，只训 camera_head + depth_head
Phase B: 解冻 aggregator.global_blocks
Phase C: 解冻 aggregator.frame_blocks + patch_embed，做全模型微调
```

### 8.3 建议保存 checkpoint 时区分结构

建议 checkpoint 中记录：

```python
checkpoint["model_name"] = "lzw-map-stage1"
checkpoint["architecture"] = {
    "patch_embed": "dinov2_vitb14_reg",
    "embed_dim": 768,
    "aggregator_depth": 12,
    "selected_idx": [2, 5, 8, 11],
    "camera_trunk_depth": 2,
    "camera_num_heads": 12,
    "enable_point": False,
}
```

这样后续 validation / resume 不会误用原始 LingBot-Map 的 24 层配置。

## 9. 后续可选轻量化方向

当前版本没有压缩 depth head。若 LZW-Map Stage1 训练稳定，下一步可以考虑：

| 方向 | 建议 |
|---|---|
| Depth head 半通道 | `features=128`, `out_channels=[128,256,512,512]` |
| Camera iterations | 保持参数不变，但可测试 `num_iterations=2` 降低计算 |
| Aggregator FFN channel pruning | 优先剪 FFN hidden channel，而不是先剪 attention head |
| Teacher-student distillation | 用原始 LingBot-Map ViT-L checkpoint 蒸馏 depth / pose / feature |
| Point head 后置训练 | Stage1 depth+pose 稳定后再启用 `enable_point=True` |

当前建议先不要同步压缩 depth head，因为你已经验证原始训练策略有效，第一版 LZW-Map 应优先验证 backbone + aggregator + camera head 轻量化后的收敛性。
