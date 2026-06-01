# GCT Checkpoint 权重大小分布分析

本报告基于本地 checkpoint：

```text
try_train/checkpoints/gct_stage2_5k/checkpoint_iter_1000.pt
```

实际文件大小：

```text
6,753,755,698 bytes = 6.29 GiB ≈ 6.75 GB
```

## 结论

当前 6G+ checkpoint 主要由两部分组成：

```text
full_model_state_dict   ≈ 4.435 GiB
optimizer_state_dict    ≈ 1.854 GiB
metadata / pickle 开销  很小
--------------------------------
checkpoint 总大小       ≈ 6.29 GiB
```

也就是说，文件变大的原因不是 KV cache 或训练数据被保存，而是保存函数同时写入了：

1. 完整 GCTStream 模型权重
2. AdamW optimizer 状态
3. head-only 权重副本
4. scheduler、loss、args 等元数据

## Checkpoint 字段

保存逻辑位于：

- `try_train/train_replica_gct.py`
- `try_train/train_replica_gct_stage2.py`

当前保存字段结构：

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

虽然 Stage1 脚本注释写的是 `only head parameters for efficiency`，但实际仍保存了 `full_model_state_dict`，因此不是 head-only checkpoint。

## 模型组件权重分布

`full_model_state_dict` 中共有：

```text
1,190,598,168 parameters
全部为 torch.float32
模型权重大小 ≈ 4.435 GiB
```

按参数名前缀统计如下：

| 组件 | 参数量 | FP32 大小 |
|---|---:|---:|
| `aggregator.patch_embed_dinov2` | 304,372,736 | 1.134 GiB |
| `aggregator.frame_blocks` | 302,364,672 | 1.126 GiB |
| `aggregator.global_blocks` | 302,364,672 | 1.126 GiB |
| `camera_head` | 216,174,610 | 0.805 GiB |
| `point_head` | 32,654,628 | 0.122 GiB |
| `depth_head` | 32,654,562 | 0.122 GiB |
| `aggregator.other_tokens_norm_rope` | 12,288 | ~0 GiB |
| **总计** | **1,190,598,168** | **4.435 GiB** |

## DINOv2 部分

对应参数名前缀：

```text
aggregator.patch_embed_dinov2
```

大小：

```text
304.37M params ≈ 1.134 GiB
```

这部分基本对应 DINOv2 ViT-L/14 图像特征提取主干。

## GCT / Aggregator 中间层

对应参数名前缀：

```text
aggregator.frame_blocks
aggregator.global_blocks
```

大小：

```text
frame_blocks   302.36M params ≈ 1.126 GiB
global_blocks  302.36M params ≈ 1.126 GiB
------------------------------------------
合计           604.73M params ≈ 2.252 GiB
```

其中：

- `frame_blocks`：帧内 self-attention，每帧独立处理
- `global_blocks`：跨帧 / causal attention，用于 GCT 的时序聚合

这部分是模型最大的主体，约占完整模型权重的一半。

## Head 部分

对应参数名前缀：

```text
camera_head
depth_head
point_head
```

大小：

```text
camera_head  216.17M params ≈ 0.805 GiB
depth_head    32.65M params ≈ 0.122 GiB
point_head    32.65M params ≈ 0.122 GiB
----------------------------------------
合计         281.48M params ≈ 1.049 GiB
```

注意：`camera_head` 明显大于 `depth_head` 和 `point_head`，因为它内部包含 4 层 camera transformer / iterative refinement blocks。

## Optimizer 状态大小

当前训练使用 AdamW。checkpoint 中 optimizer 状态统计：

```text
optimizer_state_dict tensors: 390
optimizer state numel: 497,658,456
optimizer state size: 1.854 GiB
```

AdamW 会为参与训练的参数保存：

1. `exp_avg`
2. `exp_avg_sq`
3. `step`

因此 optimizer 状态通常接近训练参数权重大小的 2 倍。

## 为什么最终是 6G+

当前 checkpoint 大小可由以下公式解释：

```text
完整模型权重：
1.190B params × 4 bytes ≈ 4.76 GB decimal ≈ 4.435 GiB

AdamW optimizer 状态：
≈ 1.854 GiB

合计：
4.435 GiB + 1.854 GiB ≈ 6.289 GiB
```

这与实际文件大小一致：

```text
实际 checkpoint: 6.29 GiB / 6.75 GB
```

## 可选瘦身方案

### 方案 A：只保存 heads + optimizer

适合继续训练时从原始 base checkpoint 加载完整模型，再加载 fine-tuned heads。

保留：

```python
{
    "iteration": iteration,
    "head_state_dict": head_state_dict,
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
    "loss_dict": loss_dict,
    "args": vars(args),
}
```

去掉：

```python
"full_model_state_dict"
```

预计大小约：

```text
head weights ≈ 0.927 GiB
optimizer    ≈ 1.854 GiB
合计         ≈ 2.8 GiB
```

### 方案 B：保存完整 inference-only checkpoint

适合推理，不需要继续训练。

保存：

```python
{
    "model": model.state_dict(),
}
```

不保存 optimizer。

预计大小：

```text
FP32: ≈ 4.44 GiB
FP16/BF16: ≈ 2.22 GiB
```

### 方案 C：保存 head-only inference checkpoint

适合只分发 fine-tuned heads，由使用者自行加载 base model。

保存：

```python
{
    "head_state_dict": head_state_dict,
}
```

预计大小：

```text
FP32: ≈ 0.93 GiB
FP16/BF16: ≈ 0.46 GiB
```

## 建议

如果训练阶段需要完整断点恢复，保留当前格式是合理的，但文件会很大。

如果只需要保存最终训练结果，建议额外导出一个 inference-only 或 head-only checkpoint，避免把 AdamW optimizer 状态一起分发。
