# LingBot-Map 网络权重构成与轻量化重训建议

本文档面向当前 `lingbot-map` 仓库中的 `GCTStream` 训练路线，重点回答三个问题：

1. 当前 LingBot-Map 网络由哪些模块构成，权重主要在哪里。
2. 你现在的 head-only 训练到底训练了多少参数，瓶颈在哪里。
3. 如果要轻量化重新训练，可以优先改哪些结构，以及剪枝、蒸馏、量化应如何落地。

统计基于本地 checkpoint：

```text
try_train/checkpoints/gct_stage2_5k/checkpoint_final.pt
```

统计方式：使用 `torch._subclasses.fake_tensor.FakeTensorMode` 读取 checkpoint 元数据，不实际加载 6GB 权重到内存。该 checkpoint 中的 `full_model_state_dict` 全部为 `torch.float32`。

## 1. 当前 GCTStream 架构概览

源码主入口：

```text
lingbot_map/models/gct_stream.py
lingbot_map/models/gct_base.py
lingbot_map/aggregator/base.py
lingbot_map/aggregator/stream.py
lingbot_map/heads/camera_head.py
lingbot_map/heads/dpt_head.py
```

当前默认模型是：

```text
GCTStream
├── aggregator: AggregatorStream
│   ├── patch_embed: DINOv2 ViT-L/14 with registers
│   ├── special tokens: camera_token, register_token, scale_token
│   ├── frame_blocks: 24 个帧内 Transformer Block
│   └── global_blocks: 24 个跨帧/因果 Transformer Block, 支持 KV cache
├── camera_head: CameraCausalHead
│   ├── 4 个 CameraBlock
│   ├── poseLN_modulation
│   ├── pose_branch
│   └── 4 次迭代式 pose refinement
├── depth_head: DPTHead, output_dim=2
└── point_head: DPTHead, output_dim=4
```

关键默认参数：

| 项 | 当前值 | 说明 |
|---|---:|---|
| `img_size` | 518 | 输入分辨率配置 |
| `patch_size` | 14 | DINOv2 ViT-L/14 |
| `embed_dim` | 1024 | aggregator token 维度 |
| `patch_embed` | `dinov2_vitl14_reg` | DINOv2 ViT-L with registers |
| `AggregatorBase.depth` | 24 | frame/global block 数量 |
| `num_heads` | 16 | aggregator attention heads |
| `mlp_ratio` | 4.0 | Transformer FFN 扩展比例 |
| `CameraCausalHead.dim_in` | 2048 | `2 * embed_dim`，frame/global 特征拼接 |
| `CameraCausalHead.trunk_depth` | 4 | camera transformer block 数 |
| `DPTHead.features` | 256 | DPT 融合通道 |
| `DPTHead.out_channels` | `[256, 512, 1024, 1024]` | DPT 多尺度投影通道 |

注意：`GCTStream.__init__` 当前暴露了 `embed_dim` 和 `patch_embed`，但没有直接暴露 `AggregatorBase.depth`、`AggregatorBase.num_heads`、`CameraCausalHead.trunk_depth`、`DPTHead.features/out_channels`。如果要系统做轻量化，建议先把这些参数显式暴露出来。

## 2. Checkpoint 字段与文件大小

当前 checkpoint 文件：

```text
size = 6,753,748,634 bytes = 6.290 GiB
```

字段结构：

```text
iteration
head_state_dict
full_model_state_dict
optimizer_state_dict
scheduler_state_dict
loss_dict
args
```

对应保存逻辑位于 `try_train/train_replica_gct.py`：

```python
checkpoint = {
    "iteration": iteration,
    "head_state_dict": head_state_dict,
    "full_model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "loss_dict": loss_dict,
    "args": vars(args),
}
```

因此当前 6GB+ 文件不是 KV cache 或数据缓存造成的，而是保存了：

| 部分 | 参数量/状态量 | FP32 大小 |
|---|---:|---:|
| `full_model_state_dict` | 1,190,598,168 params | 4.435 GiB |
| `optimizer_state_dict` | 497,658,456 tensor elements | 1.854 GiB |
| `head_state_dict` | 248,829,172 params | 0.927 GiB |
| 其他元数据 | 很小 | 可忽略 |

`head_state_dict` 是 `depth_head + camera_head`，不包含 `point_head`。但 `full_model_state_dict` 仍包含完整 aggregator、camera、depth、point，所以 checkpoint 仍然很大。

一个实现细节：`freeze_aggregator()` 只冻结了 `aggregator`。如果 `GCTStream` 仍用默认 `enable_point=True` 构造，`point_head` 参数可能仍然是 `requires_grad=True`，只是当前 `forward_gct_heads_only()` 没有调用 point head，checkpoint 里的 `head_state_dict` 和 AdamW 实际状态也没有 point head。为了让轻量化实验更干净，建议构造时直接 `enable_point=False`，或者显式冻结 `model.point_head`。

## 3. 完整模型权重分布

`full_model_state_dict` 参数总量：

```text
1,190,598,168 params
FP32 size = 4.435 GiB
```

按大模块拆分：

| 模块 | 参数量 | FP32 大小 | 占比 |
|---|---:|---:|---:|
| `aggregator.patch_embed`，DINOv2 ViT-L/14 | 304,372,736 | 1.134 GiB | 25.56% |
| `aggregator.frame_blocks` | 302,364,672 | 1.126 GiB | 25.40% |
| `aggregator.global_blocks` | 302,364,672 | 1.126 GiB | 25.40% |
| `camera_head` | 216,174,610 | 0.805 GiB | 18.16% |
| `depth_head` | 32,654,562 | 0.122 GiB | 2.74% |
| `point_head` | 32,654,628 | 0.122 GiB | 2.74% |
| `aggregator` special tokens | 12,288 | 0.000046 GiB | 0.00% |
| 总计 | 1,190,598,168 | 4.435 GiB | 100% |

最关键的结论：

```text
aggregator = DINOv2 patch_embed + frame_blocks + global_blocks
           = 906,095,168 params
           = 76.1% of full model

heads = camera_head + depth_head + point_head
      = 281,483,800 params
      = 23.6% of full model

当前 checkpoint 中实际被保存/更新的 head 参数 = camera_head + depth_head
                                  = 248,829,172 params
                                  = 20.9% of full model
```

也就是说，虽然你冻结了 aggregator，但 head-only 训练依然不是很小，因为 `camera_head` 本身就有 2.16 亿参数。

## 4. Aggregator 权重构成

### 4.1 DINOv2 patch embedding

`aggregator.patch_embed` 总量：

| 子模块 | 参数量 | FP32 大小 |
|---|---:|---:|
| tokens + pos + final norm | 1,411,072 | 0.005 GiB |
| patch projection | 603,136 | 0.002 GiB |
| DINOv2 transformer blocks | 302,358,528 | 1.126 GiB |
| 总计 | 304,372,736 | 1.134 GiB |

DINOv2 这部分大头同样是 Transformer blocks，而不是 patch projection。

### 4.2 Frame/global Transformer blocks

`frame_blocks` 和 `global_blocks` 各有 24 个 block。

单个 block 参数构成：

| block 类型 | 单 block 参数量 | Attention | MLP/FFN | Norm | LayerScale |
|---|---:|---:|---:|---:|---:|
| DINO block | 12,598,272 | 33.33% | 66.63% | 0.03% | 0.02% |
| frame block | 12,598,528 | 33.33% | 66.62% | 0.03% | 0.02% |
| global block | 12,598,528 | 33.33% | 66.62% | 0.03% | 0.02% |

因此对 Transformer 主体做轻量化时，最有效的结构方向通常是：

1. 降低宽度 `embed_dim`。
2. 降低深度 `depth`。
3. 降低 FFN hidden 维度，也就是降低 `mlp_ratio` 或结构化剪掉 FFN channel。
4. attention head pruning 有用，但单独做通常不如降宽度/FFN 来得直接。

## 5. Head-only 训练的实际瓶颈

当前 head-only 保存的 `head_state_dict`：

| 模块 | 参数量 | FP32 大小 | 占 head-only 比例 |
|---|---:|---:|---:|
| `camera_head` | 216,174,610 | 0.805 GiB | 86.88% |
| `depth_head` | 32,654,562 | 0.122 GiB | 13.12% |
| 总计 | 248,829,172 | 0.927 GiB | 100% |

### 5.1 Camera head 内部拆分

| 子模块 | 参数量 | FP32 大小 |
|---|---:|---:|
| 4 个 `CameraBlock` trunk | 201,449,472 | 0.750 GiB |
| `poseLN_modulation` | 12,589,056 | 0.047 GiB |
| `pose_branch` | 2,107,401 | 0.008 GiB |
| `embed_pose` | 20,480 | 0.000076 GiB |
| norm + token | 8,201 | 约 0 |
| 总计 | 216,174,610 | 0.805 GiB |

结论：head-only 中最该轻量化的是 `CameraCausalHead.trunk`，不是 depth head。

`CameraCausalHead` 参数变体统计：

| `dim_in` | `trunk_depth` | 参数量 | FP32 大小 |
|---:|---:|---:|---:|
| 2048 | 4 | 216,174,610 | 0.805 GiB |
| 2048 | 3 | 165,812,242 | 0.618 GiB |
| 2048 | 2 | 115,449,874 | 0.430 GiB |
| 2048 | 1 | 65,087,506 | 0.242 GiB |
| 1536 | 4 | 121,629,714 | 0.453 GiB |
| 1536 | 2 | 64,960,530 | 0.242 GiB |
| 768 | 4 | 30,438,930 | 0.113 GiB |
| 768 | 2 | 16,260,114 | 0.061 GiB |

如果保持原始 aggregator 的 `embed_dim=1024`，则 camera head 的 `dim_in=2048`。此时把 `trunk_depth` 从 4 降到 2，可以把 head-only 训练参数从 2.49 亿降到约 1.48 亿，收益明显。

### 5.2 DPT depth head 内部拆分

当前 `depth_head`：

| 子模块 | 参数量 | 大小 |
|---|---:|---:|
| norm | 4,096 | 0.016 MiB |
| `projects` 1x1 conv | 5,769,984 | 22.011 MiB |
| `resize_layers` | 11,536,128 | 44.007 MiB |
| `scratch.layer*_rn` | 6,488,064 | 24.750 MiB |
| `scratch.refinenet*` | 8,524,288 | 32.518 MiB |
| `scratch.output_conv*` | 332,002 | 1.266 MiB |
| 总计 | 32,654,562 | 0.122 GiB |

DPT head 单独不算巨大，但如果你做 ViT-S 级别的小模型，DPT 会变成较明显的占比瓶颈。

DPTHead 轻量变体：

| 变体 | `dim_in` | `features` | `out_channels` | 参数量 | FP32 大小 |
|---|---:|---:|---|---:|---:|
| 原始 depth head | 2048 | 256 | `[256,512,1024,1024]` | 32,654,562 | 0.122 GiB |
| 只降 `features` | 2048 | 128 | `[256,512,1024,1024]` | 22,778,786 | 0.085 GiB |
| 通道减半 | 2048 | 128 | `[128,256,512,512]` | 9,620,130 | 0.036 GiB |
| 通道四分之一 | 2048 | 64 | `[64,128,256,256]` | 3,134,850 | 0.012 GiB |

优先建议：如果要轻量化 DPT，不要只改 `features`，应同时降低 `out_channels`，否则 `projects/resize_layers` 仍然很大。

## 6. Backbone/Aggregator 替换的参数收益

当前代码支持 `dinov2_vitl14_reg`、`dinov2_vitb14_reg`、`dinov2_vits14_reg`、`dinov2_vitg2_reg`，但替换时必须同步修改 `embed_dim`：

| patch_embed | embed_dim |
|---|---:|
| `dinov2_vitl14_reg` | 1024 |
| `dinov2_vitb14_reg` | 768 |
| `dinov2_vits14_reg` | 384 |
| `dinov2_vitg2_reg` | 1536 |

在保留当前 24 个 frame/global blocks、保留 camera/depth/point heads 的情况下，实测变体参数为：

| 架构 | 总参数量 | FP32 大小 | 相对原模型 |
|---|---:|---:|---:|
| 当前 ViT-L, width 1024, depth 24 | 1,190,598,168 | 4.435 GiB | 100% |
| ViT-B, width 768, depth 24 | 610,946,840 | 2.276 GiB | 51.3% |
| ViT-S, width 384, depth 24 | 195,812,504 | 0.729 GiB | 16.4% |

如果进一步把 frame/global block 深度从 24 降到 12，按当前 block 参数量估算：

| 架构 | 估算总参数量 | FP32 大小 | 相对原模型 |
|---|---:|---:|---:|
| ViT-B, width 768, depth 12 | 440,796,440 | 1.642 GiB | 37.0% |
| ViT-S, width 384, depth 12 | 153,204,632 | 0.571 GiB | 12.9% |

重要限制：

1. `GCTStream` 当前没有把 `AggregatorBase.depth` 暴露出来。需要改构造函数才能真正训练 depth 12 的 student。
2. 从 ViT-L 换到 ViT-B/ViT-S 后，`embed_dim` 不同，原始 checkpoint 不能直接 strict load。
3. 建议用 DINOv2 B/S 预训练初始化 student backbone，再用原始 LingBot-Map 作为 teacher 做蒸馏。

## 7. 本地 DINOv2 仓库补充

你已经把官方 DINOv2 仓库 clone 到：

```text
/home/lizhengwu/desktop/temp_proj/dinov2/
```

当前本地仓库信息：

```text
remote: https://github.com/facebookresearch/dinov2.git
commit: 7b187bd
```

这个仓库对 LingBot-Map 轻量化最有用的部分是：

| 文件 | 用途 |
|---|---|
| `/home/lizhengwu/desktop/temp_proj/dinov2/hubconf.py` | `torch.hub.load(..., source="local")` 的入口 |
| `/home/lizhengwu/desktop/temp_proj/dinov2/dinov2/hub/backbones.py` | 官方 backbone 构造和权重下载逻辑 |
| `/home/lizhengwu/desktop/temp_proj/dinov2/dinov2/models/vision_transformer.py` | 官方 ViT-S/B/L/g 架构定义 |
| `/home/lizhengwu/desktop/temp_proj/dinov2/dinov2/configs/eval/*reg4_pretrain.yaml` | reg4 模型的配置，确认 `num_register_tokens=4`、`patch_size=14`、`global_crops_size=518` |
| `/home/lizhengwu/desktop/temp_proj/dinov2/dinov2/eval/depth/models/decode_heads/dpt_head.py` | 官方 DINOv2 depth evaluation 的 DPT head，可作结构参考，但不建议直接替换 LingBot DPTHead |

### 7.1 官方 DINOv2 模型族

DINOv2 README 中的 backbone 参数量和本地代码实测基本一致。下面是本地 `pretrained=False` 实例化后的参数统计：

| hub name | registers | embed dim | depth | heads | 参数量 | FP32 大小 |
|---|---:|---:|---:|---:|---:|---:|
| `dinov2_vits14` | 0 | 384 | 12 | 6 | 22,056,576 | 84.14 MiB |
| `dinov2_vits14_reg` | 4 | 384 | 12 | 6 | 22,058,112 | 84.15 MiB |
| `dinov2_vitb14` | 0 | 768 | 12 | 12 | 86,580,480 | 330.28 MiB |
| `dinov2_vitb14_reg` | 4 | 768 | 12 | 12 | 86,583,552 | 330.29 MiB |
| `dinov2_vitl14` | 0 | 1024 | 24 | 16 | 304,368,640 | 1161.07 MiB |
| `dinov2_vitl14_reg` | 4 | 1024 | 24 | 16 | 304,372,736 | 1161.09 MiB |

LingBot-Map 当前使用的是 `dinov2_vitl14_reg` 对应的结构，所以 `aggregator.patch_embed = 304,372,736 params` 正好等于官方 ViT-L/14 reg4 backbone。

官方 DINOv2 hub 中 register 版本的构造逻辑是：

```text
dinov2_vits14_reg -> vit_small, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0
dinov2_vitb14_reg -> vit_base,  num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0
dinov2_vitl14_reg -> vit_large, num_register_tokens=4, interpolate_antialias=True, interpolate_offset=0.0
dinov2_vitg14_reg -> vit_giant2, num_register_tokens=4, ffn_layer=swiglufused
```

这和 LingBot-Map `AggregatorBase._build_patch_embed()` 的 `vit_models` 映射是一致的：

```text
dinov2_vitl14_reg -> vit_large
dinov2_vitb14_reg -> vit_base
dinov2_vits14_reg -> vit_small
dinov2_vitg2_reg  -> vit_giant2
```

### 7.2 官方 DINOv2 与 LingBot 内置 ViT 的差异

LingBot-Map 内部已经复制/改写了一份 DINOv2 ViT：

```text
lingbot_map/layers/vision_transformer.py
```

它和官方 `/home/lizhengwu/desktop/temp_proj/dinov2/dinov2/models/vision_transformer.py` 大体一致，但有几个对轻量化很重要的差异：

| 点 | 官方 DINOv2 | LingBot-Map 内置版本 | 影响 |
|---|---|---|---|
| class token | 固定使用 `cls_token` | 增加 `drop_cls_token` 选项，但当前默认不 drop | 默认兼容官方权重 |
| QK norm | 官方 backbone 构造未显式传 `qk_norm` | LingBot 版本增加 `qk_norm` 参数 | 从官方权重加载时可能出现 q/k norm missing，当前 `strict=False` 可处理 |
| gradient checkpoint | 官方 ViT 普通 forward | LingBot 训练时对 blocks 使用 checkpoint | 省显存，不改变权重形状 |
| pos_embed | 官方 hub strict load | LingBot 当前加载 DINO 权重前 `del ckpt["pos_embed"]` | fresh student 初始化时需要特别注意 |
| frame/global blocks | 官方没有 | LingBot 把 DINO block 权重复制初始化到 `frame_blocks/global_blocks` | 这是 GCT aggregator 初始化的核心 |

最值得注意的是 `pos_embed`：当前 LingBot 的 `_build_patch_embed()` 会先删除官方 checkpoint 里的 `pos_embed`，再 `strict=False` 加载。这样做可以绕开输入尺寸/长宽变化带来的 shape 问题，但如果你训练一个新的 ViT-B/ViT-S student，它会导致 `patch_embed.pos_embed` 保持随机初始化。

建议：

1. 如果 `pos_embed` shape 匹配，保留并加载官方 `pos_embed`。
2. 如果 shape 不匹配，使用 DINOv2 自带的插值逻辑，或至少把 `pos_embed` 初始化策略写清楚。
3. 对轻量 student 来说，不建议让 DINO backbone 的 `pos_embed` 随机起步，除非后续用大量蒸馏训练补回来。

### 7.3 如何用本地 DINOv2 仓库导出 LingBot 可用的 backbone 权重

LingBot 的 `pretrained_path` 期望的是“裸 DINOv2 backbone state_dict”，也就是 keys 类似：

```text
cls_token
pos_embed
register_tokens
mask_token
patch_embed.proj.weight
blocks.0.norm1.weight
...
```

官方 hub 的 backbone pretrain `.pth` 正是这种格式。可以用本地仓库入口下载/加载，再保存：

```bash
/home/lizhengwu/miniconda3/envs/lingbot-map/bin/python - <<'PY'
import torch

DINOV2_REPO = "/home/lizhengwu/desktop/temp_proj/dinov2"
OUT = "/home/lizhengwu/desktop/temp_proj/lingbot-map/pretrained/dinov2_vitb14_reg4_pretrain.pth"

model = torch.hub.load(
    DINOV2_REPO,
    "dinov2_vitb14_reg",
    source="local",
    pretrained=True,
)
torch.save(model.state_dict(), OUT)
print(OUT)
PY
```

如果你已经手动下载了官方权重，也可以直接传本地权重路径给官方 hub：

```python
model = torch.hub.load(
    "/home/lizhengwu/desktop/temp_proj/dinov2",
    "dinov2_vitb14_reg",
    source="local",
    pretrained=True,
    weights="/path/to/dinov2_vitb14_reg4_pretrain.pth",
)
```

随后在 LingBot-Map student 中使用：

```python
model = GCTStream(
    patch_embed="dinov2_vitb14_reg",
    embed_dim=768,
    pretrained_path="pretrained/dinov2_vitb14_reg4_pretrain.pth",
    use_sdpa=True,
    enable_point=False,
)
```

ViT-S student 对应：

```python
model = GCTStream(
    patch_embed="dinov2_vits14_reg",
    embed_dim=384,
    pretrained_path="pretrained/dinov2_vits14_reg4_pretrain.pth",
    use_sdpa=True,
    enable_point=False,
)
```

### 7.4 GCT aggregator 初始化为什么不只是加载 DINO backbone

LingBot-Map 的 aggregator 不是简单的 DINOv2 backbone。它有三层 DINO/GCT 关系：

```text
1. patch_embed
   完整 DINOv2 ViT backbone，用来提取每帧 patch features。

2. frame_blocks
   24 个帧内 Transformer blocks，每帧独立处理。

3. global_blocks
   24 个跨帧 causal/global blocks，用于几何上下文聚合。
```

`AggregatorBase._init_blocks_from_dino()` 会把官方 DINO checkpoint 的 `blocks.i.*` 复制到：

```text
aggregator.frame_blocks.i.*
aggregator.global_blocks.i.*
```

因此，传入 `pretrained_path` 的价值不只是初始化 `patch_embed`，还会给 GCT 的 frame/global blocks 一个 DINO block 初始化。这一点对 ViT-B/ViT-S student 很重要：

```text
如果 student 从官方 DINOv2-B/S reg4 checkpoint 初始化：
  patch_embed 有 DINO 预训练特征
  frame_blocks/global_blocks 也可从 DINO block 初始化

如果 student 完全随机初始化：
  aggregator 三大块都随机，重训成本会大很多
```

对 depth 12 的 student，初始化逻辑仍然可用：`frame_blocks/global_blocks` 只会加载前 12 个 DINO blocks。如果未来做 depth 大于 DINO block 数的结构，当前逻辑会用 `i % num_dino_blocks` 循环复用 DINO blocks。

### 7.5 本地 DINOv2 仓库对轻量化路线的直接建议

结合官方模型族，推荐的 student 初始化优先级是：

| 目标 | 推荐 DINOv2 初始化 | 原因 |
|---|---|---|
| 稳健轻量化 | `dinov2_vitb14_reg` | 参数约为 ViT-L 的 28.4%，ImageNet linear 仍接近 L；适合第一版 student |
| 激进轻量化 | `dinov2_vits14_reg` | 参数约 22M，完整 GCT student 可以压到很小，但需要更强蒸馏 |
| 不推荐 | `dinov2_vitg14_reg` | 比当前 ViT-L 更大，不符合轻量化目标 |
| 不推荐直接用 | 非 register 版本 | LingBot aggregator 默认 `num_register_tokens=4`，非 reg 权重会缺少 register tokens |

这里的“参数约为 ViT-L 的 28.4%”只指 DINOv2 backbone 本身：

```text
ViT-B-reg backbone: 86.58M / 304.37M = 28.4%
ViT-S-reg backbone: 22.06M / 304.37M = 7.2%
```

但完整 GCT student 还包含 frame/global blocks 和 heads，所以最终完整模型比例见第 6 节。

### 7.6 对 `try_train/head_only_model.py` 的影响

`try_train/head_only_model.py` 早期使用：

```python
torch.hub.load("facebookresearch/dinov2", model_name)
```

现在有本地 DINOv2 仓库后，可以改为：

```python
torch.hub.load(
    "/home/lizhengwu/desktop/temp_proj/dinov2",
    model_name,
    source="local",
)
```

这样有两个好处：

1. 不依赖在线 GitHub 拉取代码。
2. 可以固定本地 DINOv2 commit，避免 torch hub 缓存或远端代码变化导致结果不一致。

不过这个早期 `HeadOnlyModel` 是“DINOv2 backbone + heads”的简化路线，不是当前 GCTStream head-only 路线。当前轻量化主线仍建议围绕 `GCTStream` 做。

## 8. 轻量化建议优先级

### P0: 先修 checkpoint 保存策略

这不改变网络结构，但能立刻减少磁盘和分发体积。

推荐保存三种 checkpoint：

| 用途 | 保存字段 | 预计大小 |
|---|---|---:|
| 继续训练 head-only | `head_state_dict + optimizer + scheduler` | 约 2.8 GiB |
| 推理分发完整模型 | `model.state_dict()`，不含 optimizer | FP32 4.44 GiB，FP16 2.22 GiB |
| 只分发微调 heads | `head_state_dict` | FP32 0.93 GiB，FP16 0.46 GiB |

当前保存函数虽然注释写了 only head parameters，但实际同时保存了 `full_model_state_dict`。建议增加参数：

```text
--save_mode full_train | head_train | full_infer | head_infer
```

### P1: 保持原始 aggregator，先轻量化 head-only 训练

这是风险最低的路线，因为你已经能训练当前 head-only。

推荐结构：

```text
aggregator: 保持冻结，沿用原始 ViT-L + frame/global blocks
camera_head: trunk_depth 4 -> 2
depth_head: out_channels [256,512,1024,1024] -> [128,256,512,512]
point_head: 如果训练不使用，构造时 enable_point=False
```

参数收益：

| 训练配置 | trainable 参数量 | 相对当前 head-only |
|---|---:|---:|
| 当前 `camera_head d4 + depth_head orig` | 248,829,172 | 100% |
| `camera_head d2 + depth_head orig` | 148,104,436 | 59.5% |
| `camera_head d2 + depth_head half` | 125,070,004 | 50.3% |
| `camera_head d1 + depth_head half` | 74,707,636 | 30.0% |
| `camera_head d1 + depth_head quarter` | 68,222,356 | 27.4% |

建议先试：

```text
camera_head.trunk_depth = 2
DPTHead.features = 128
DPTHead.out_channels = [128, 256, 512, 512]
num_iterations = 2 或 3
```

说明：

1. `trunk_depth` 影响参数和计算量。
2. `num_iterations` 只影响计算量，不影响参数量，但对训练/推理速度有帮助。
3. camera head 降到 1 层可能对长序列 pose 稳定性影响较大，建议先从 2 层开始。

### P2: 替换 backbone 为 ViT-B student

如果 P1 精度仍可接受但模型仍太大，建议做 ViT-B student：

```text
patch_embed = dinov2_vitb14_reg
embed_dim = 768
aggregator depth = 24 或 12
camera_head dim_in = 1536
camera_head trunk_depth = 2
DPTHead features = 128
DPTHead out_channels = [128,256,512,512]
```

优点：

1. 参数约为原模型 37% 到 51%，再配合 head 缩小可继续下降。
2. ViT-B 表达能力仍较强，精度风险比 ViT-S 小。
3. 可以用 DINOv2 ViT-B 预训练权重初始化 patch_embed。

缺点：

1. 不能直接从原始 ViT-L checkpoint strict load。
2. 需要 teacher-student 蒸馏，不建议从零开始训练。
3. 需要改代码暴露 aggregator depth/head/DPT 参数。

### P3: ViT-S student 或更激进 Tiny 模型

ViT-S depth 24 全模型约 1.96 亿参数，已经接近当前 head-only 的 2.49 亿参数，但它是完整模型，不只是 heads。

适合场景：

1. 目标是边缘部署或实时推理。
2. 可以接受较明显精度下降。
3. 有足够 teacher 蒸馏数据，或者有较强的真实监督数据。

建议：

```text
patch_embed = dinov2_vits14_reg
embed_dim = 384
aggregator depth = 12 或 24
camera_head trunk_depth = 2
DPTHead features = 128
DPTHead out_channels = [128,256,512,512]
```

对于 ViT-S，原始 DPT head 会占比较高，必须同步轻量化 DPT。

## 9. 剪枝建议

剪枝建议做“结构化剪枝”，最终导出为真实变小的 dense 模型，而不是只把权重置零。

### 8.1 优先剪 DPT head channel

DPT 是最容易做结构化剪枝的部分：

1. 对 `projects` 的输出通道做 channel pruning。
2. 同步裁剪 `resize_layers`、`scratch.layer*_rn`、`refinenet*` 的输入输出通道。
3. 直接把结构改成较小 `out_channels`，通常比先训练大头再剪更简单。

推荐从下面两个结构开始：

```text
half:
features = 128
out_channels = [128, 256, 512, 512]

quarter:
features = 64
out_channels = [64, 128, 256, 256]
```

### 8.2 Camera head 剪枝

camera head 的 93% 以上参数在 4 个 `CameraBlock` trunk 中。

优先级：

1. Layer dropping：`trunk_depth 4 -> 2`。这是最简单可靠的结构剪枝。
2. FFN hidden channel pruning：每个 Transformer/CameraBlock 中 MLP 占约 66.6% 参数，剪 FFN hidden channel 收益最大。
3. Attention head pruning：每个 block attention 占约 33.3%，可以按 head importance 剪，但实现复杂度高于 FFN channel。

实现要点：

```text
FFN pruning:
mlp.fc1: Linear(dim -> hidden)
mlp.fc2: Linear(hidden -> dim)
剪掉 fc1 的若干输出 channel，同时剪掉 fc2 对应输入 channel。

Attention head pruning:
attn.qkv: Linear(dim -> 3*dim)
attn.proj: Linear(dim -> dim)
按 head 维度裁剪 q/k/v 对应切片，同时裁剪 proj 输入切片。
```

### 8.3 Aggregator 剪枝

Aggregator 占完整模型 76.1%，但它现在是冻结的，直接剪会影响全局几何上下文。建议顺序：

1. 先做 student 架构替换，降低 `embed_dim/depth`。
2. 再对 student 做 FFN channel pruning。
3. 最后尝试 attention head pruning。

不建议第一步就对原始 ViT-L aggregator 做大比例非结构化剪枝，因为：

1. 参数会变稀疏但模型文件和推理不一定真的变小。
2. 稀疏加速依赖硬件和 kernel。
3. 对几何一致性影响不可控。

### 8.4 剪枝实验流程

建议流程：

```text
1. 先冻结 teacher，跑一份稳定 validation baseline。
2. 选择结构化目标：camera trunk depth、DPT channels、aggregator depth/width。
3. 做小比例消融：25%, 50%, 75% 参数量三个档。
4. 使用 teacher distillation + 原始 depth/pose loss 微调。
5. 每个档记录 depth loss、abs pose、rel pose、长序列漂移、推理 FPS。
6. 保留最小且精度可接受的 student。
```

## 10. 蒸馏建议

轻量化重新训练时，不建议只依赖 Replica/ScanNet 的监督信号从头训练 student。更稳的方式是 teacher-student 蒸馏：

Teacher：

```text
原始 LingBot-Map GCTStream ViT-L checkpoint
```

Student：

```text
ViT-B 或 ViT-S GCTStream
可选较浅 frame/global blocks
可选较小 camera/depth heads
```

蒸馏损失建议：

| Loss | 目标 |
|---|---|
| `L_depth_gt` | 对齐 GT depth |
| `L_pose_gt` | 对齐 GT absolute/relative pose |
| `L_depth_teacher` | student depth 对齐 teacher depth |
| `L_pose_teacher` | student pose 对齐 teacher pose |
| `L_feature` | 对齐 selected_idx token/features，可加 1x1/Linear adapter |
| `L_reproj` | 用预测 depth + pose 做重投影一致性 |

推荐总 loss：

```text
L = L_gt_depth
  + L_gt_pose
  + lambda_d * L_teacher_depth
  + lambda_p * L_teacher_pose
  + lambda_f * L_feature
  + lambda_r * L_reproj
```

初始建议：

```text
lambda_d = 0.5
lambda_p = 0.5
lambda_f = 0.1
lambda_r = 0.1
```

如果 student 和 teacher 维度不同，例如 ViT-L 1024 到 ViT-B 768，不要强行直接对齐 token 通道。应加一个轻量 projection：

```text
student_feature -> Linear(student_dim, teacher_dim)
或 teacher_feature -> Linear(teacher_dim, student_dim)
```

## 11. 量化与训练内存建议

量化主要用于推理部署或 checkpoint 体积，不是结构性减少参数。

推荐：

1. 推理 checkpoint 导出 FP16/BF16：完整模型从 4.44 GiB 降到约 2.22 GiB。
2. head-only checkpoint 导出 FP16/BF16：从 0.93 GiB 降到约 0.46 GiB。
3. 训练时继续用 AMP/BF16，减少 activation 显存。
4. AdamW optimizer 可换 8-bit Adam 或者只保存必要状态，减少 optimizer checkpoint。
5. 对冻结 aggregator，推理/训练前向可考虑权重量化，但要验证 depth/pose 数值稳定性。

不建议一开始就做 INT4/INT8 量化训练，因为深度和位姿回归对数值误差较敏感。更稳的顺序是：

```text
结构轻量化 -> 蒸馏重训 -> FP16/BF16 导出 -> PTQ/QAT 小规模验证
```

## 12. 推荐的三条落地路线

### 路线 A：最小改动，快速得到轻量 head-only

目标：保持原始 aggregator，不碰 backbone，尽快减少训练参数。

改动：

```text
CameraCausalHead.trunk_depth: 4 -> 2
DPTHead.features: 256 -> 128
DPTHead.out_channels: [256,512,1024,1024] -> [128,256,512,512]
save checkpoint 时不保存 full_model_state_dict
```

预期：

```text
trainable params: 248.8M -> 125.1M
约减少 49.7%
```

这是我建议你第一优先尝试的路线。

### 路线 B：中等改动，ViT-B student

目标：完整模型真正变小，同时尽量保精度。

改动：

```text
patch_embed = dinov2_vitb14_reg
embed_dim = 768
Aggregator depth = 12 或 24
CameraCausalHead.trunk_depth = 2
DPTHead half channels
teacher-student distillation
```

预期：

```text
未缩 head 的 ViT-B depth24: 610.9M params, 原模型 51.3%
ViT-B depth12: 440.8M params, 原模型 37.0%
再缩 camera/DPT 后可继续下降
```

这是精度和轻量化之间最平衡的路线。

### 路线 C：激进部署，ViT-S student

目标：边缘部署或低显存推理。

改动：

```text
patch_embed = dinov2_vits14_reg
embed_dim = 384
Aggregator depth = 12
CameraCausalHead.trunk_depth = 2
DPTHead half channels
强蒸馏 + 长序列验证
```

预期：

```text
ViT-S depth24 full: 195.8M params, 原模型 16.4%
ViT-S depth12 full: 153.2M params, 原模型 12.9%
```

风险：长序列几何一致性和 pose 稳定性可能明显下降，需要更严格验证。

## 13. 建议代码改造清单

为了支持轻量化实验，建议先做这些小改动：

1. 在 `GCTStream.__init__` 增加：

```python
aggregator_depth: int = 24
aggregator_num_heads: int = 16
aggregator_mlp_ratio: float = 4.0
camera_trunk_depth: int = 4
camera_num_heads: int = 16
dpt_features: int = 256
dpt_out_channels: list[int] = [256, 512, 1024, 1024]
```

2. 在 `_build_aggregator()` 中把 `depth/num_heads/mlp_ratio` 传给 `AggregatorStream`。

3. 在 `_build_camera_head()` 中把 `trunk_depth/num_heads` 传给 `CameraCausalHead`。

4. 在 `_build_depth_head()` 和 `_build_point_head()` 中把 `features/out_channels` 传给 `DPTHead`。

5. 在训练脚本中增加 partial load：

```text
strict=False
只加载 shape 完全匹配的参数
对 trunk_depth 变小的 camera head，可加载 trunk.0/trunk.1 等前几层
对 DPT channel 改变的 head，多数参数需要重新初始化或蒸馏
```

6. 增加 teacher-student forward：

```text
teacher.eval(), no_grad
student.train()
同时计算 GT loss 和 teacher distillation loss
```

7. 增加 checkpoint `save_mode`，避免每次保存完整 6.3 GiB 训练断点。

## 14. 最终建议

如果你的目标是“现在能继续训练，并尽快把模型变轻”，建议按下面顺序做：

```text
Step 1: 修 checkpoint 保存，导出 head-only / fp16 checkpoint。
Step 2: 保持 aggregator 冻结，把 camera_head trunk_depth 从 4 改到 2。
Step 3: 把 depth_head 改成 half DPT: features=128, out_channels=[128,256,512,512]。
Step 4: 用当前完整模型作为 teacher，对轻量 heads 做蒸馏重训。
Step 5: 如果精度可接受，再做 ViT-B student，并从本地 DINOv2 的 `dinov2_vitb14_reg` 权重初始化。
Step 6: 最后再考虑 ViT-S、aggregator depth 12、FFN/head pruning。
```

当前最值得优先动的参数不是 DINOv2 patch projection，也不是 special tokens，而是：

```text
1. camera_head.trunk
2. aggregator 的 frame/global Transformer 深度和宽度
3. Transformer FFN hidden channels
4. DPTHead 的 out_channels/features
```
