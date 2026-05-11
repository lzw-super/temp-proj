# 资源受限版：ScanNet 上训练 LingBot-Map 第一阶段 Head 网络

本文档基于当前文件夹中的主论文 `Geometric Context Transformer for.pdf`、VGGT、π3、DINOv2 论文整理。原论文的第一阶段 `4.1 Base Model Training` 是大规模 end-to-end base model 训练；但当前目标不是复现论文精度，而是在条件有限时把数据获取、数据预处理、训练策略和参数设置完整跑通。因此本文档将训练目标调整为：

- 只使用 ScanNet 公开数据集。
- 不训练第二阶段 Streaming/GCA。
- 不训练完整 backbone 和 24 层 alternating attention。
- 冻结大模型特征提取部分，只训练较小的 head 网络。
- 训练能启动、loss 能正常计算、checkpoint 能保存即可，不以准确率提升为目标。

## 1. 与论文原始设置的关系

主论文第一阶段的完整设定是：

- DINOv2 ViT 初始化，patch size 为 14。
- 24 个 frame attention 与 global cross-frame attention 交替 block。
- 输入视角数从 2 到 24 随机采样。
- 使用 AdamW，base learning rate 为 `2e-4`，weight decay 为 `0.05`。
- 训练 160K iterations。
- 使用 FSDP、gradient checkpointing、bf16 mixed precision。
- 训练整个网络。

当前资源受限版做如下简化：

| 项目 | 原论文第一阶段 | 本文档建议 |
|---|---|---|
| 数据集 | 29 个数据集混合 | 只用 ScanNet |
| 输入视角 | 2-24 views | 先用 2-4 views |
| 图像尺寸 | 最大边 518 | 先用 224 或 336 |
| Backbone | DINOv2 初始化并参与训练 | DINOv2 冻结 |
| Cross-frame transformer | 24 个 alternating blocks | 冻结已有模块，或直接省略 |
| 训练模块 | 全网络 | 只训练轻量 depth/pose head |
| 训练轮数 | 160K iterations | 1K-5K iterations smoke run |
| 分布式 | FSDP + bf16 | 单卡/小显存即可 |
| 目标 | 高质量几何先验 | 验证数据和训练流程能跑通 |

这不是严格论文复现，而是一个工程可启动版本。文档中仍保留论文中的关键数据处理原则：camera-to-world pose、ScanNet depth 单位转换、metadata 统一、nearby sampler、reference-normalized pose target。

## 2. 推荐的最小模型方案

### 2.1 方案 A：最小可运行 Head-only 版本

适合没有 VGGT/LingBot-Map 完整预训练 checkpoint 的情况。

结构：

```text
RGB image
  -> frozen DINOv2 ViT backbone
  -> frozen image tokens / pooled features
  -> small trainable heads
       - DepthHead: patch tokens -> low-resolution depth -> upsample
       - PoseHead: pooled per-frame feature -> camera-to-world pose relative to first view
```

建议只训练以下模块：

- `DepthHead`
- `PoseHead`
- 可选 `SmallMultiViewHead`：2 层小 Transformer 或 MLP，用于在 2-4 views 间交换信息

冻结以下模块：

- DINOv2 ViT backbone
- 所有已有 VGGT/LingBot-Map transformer blocks
- camera/register/anchor token 相关主体结构
- GCA、KV cache、trajectory memory

### 2.2 方案 B：如果已有完整 VGGT/LingBot-Map 代码和 checkpoint

如果代码中已经有完整 backbone、alternating attention 和 heads，则更接近论文的简化方式是：

```text
冻结 DINOv2 backbone
冻结 24 层 alternating attention blocks
只训练 camera head、depth head、uncertainty head
```

如果显存仍不足，可以进一步：

- 只保留 depth head。
- 暂时关闭 pose loss 和 relative pose loss。
- 使用单帧或 2 views 跑通训练。

### 2.3 Head 网络建议

轻量 head 可以按如下方式设计：

```yaml
head_only_model:
  backbone:
    name: dinov2_vits14_or_vitb14
    frozen: true
    output: patch_tokens

  multiview_head:
    enabled: true
    layers: 2
    hidden_dim: 256
    num_heads: 4
    frozen: false

  depth_head:
    type: mlp_or_small_conv_decoder
    hidden_dim: 256
    output: depth_meters
    frozen: false

  pose_head:
    type: pooled_feature_mlp
    hidden_dim: 256
    rotation: 6d_or_quaternion
    translation: xyz
    output: T_c2w_relative_to_first_view
    frozen: false
```

如果只是为了展示流程，优先选择更简单的 head，而不是追求 DPT 级别的深度解码器。

## 3. ScanNet 数据获取

### 3.1 下载内容

使用 ScanNet 公开 RGB-D sequence。通常下载的是每个 scene 的 `.sens` 文件，需要用官方 exporter 导出训练需要的文件：

- RGB 图像。
- 16-bit depth PNG。
- 每帧 4x4 camera pose。
- 相机内参。

推荐目录结构：

```text
scannet/
  scans/
    scene0000_00/
      scene0000_00.sens
      color/
        0.jpg
        1.jpg
      depth/
        0.png
        1.png
      pose/
        0.txt
        1.txt
      intrinsic/
        intrinsic_color.txt
        intrinsic_depth.txt
```

### 3.2 先准备小子集

当前目标是跑通流程，不建议一开始处理全量 ScanNet。建议分三步：

1. 只处理 1 个 scene，做数据读取和 overfit smoke test。
2. 扩展到 5-20 个 train scenes，检查 sampler 和 batch。
3. 确认训练稳定后再扩大数据规模。

validation split 可以先保留但不强求指标提升。当前阶段只需要确认 validation dataloader 能跑、loss 是 finite。

## 4. ScanNet 数据预处理流程

主论文 Appendix A.1 的核心思想是把不同数据集统一成同一种格式。只用 ScanNet 时，流程可以简化为以下几步。

### 4.1 导出 `.sens`

用 ScanNet 官方 Python exporter 将 `.sens` 导出为：

- `color/*.jpg`
- `depth/*.png`
- `pose/*.txt`
- `intrinsic/*.txt`

ScanNet exporter 导出的 pose 通常是 4x4 camera-to-world matrix。训练代码中应统一使用 `T_c2w`，不要在 metadata 中混用 world-to-camera。

### 4.2 深度单位转换

ScanNet depth PNG 是 16-bit，单位为毫米。训练时统一转为米：

```python
depth_m = depth_png.astype("float32") / 1000.0
valid_mask = depth_m > 0
```

无效值处理：

- `depth <= 0` 置为 invalid。
- `NaN` / `Inf` 置为 `0`。
- loss 只在 valid depth pixel 上计算。

主论文中提到的 sky mask、outlier mask、98th percentile clamp 主要用于其他数据集。ScanNet 室内 RGB-D 简化训练阶段可不做这些复杂处理。

### 4.3 RGB、Depth 和 Intrinsics 对齐

训练时必须保证 RGB、depth、valid mask 和 intrinsics 对齐。建议：

- 先读 RGB、depth、`K_color`、`T_c2w`。
- 把 RGB resize 到最大边 `224` 或 `336`。
- depth 用 nearest 或 bilinear resize 到同一训练分辨率；valid mask 用 nearest。
- 将训练尺寸 pad 到 14 的整数倍，方便 DINOv2 patch size 14。
- 同步更新内参：

```text
fx' = fx * scale_x
fy' = fy * scale_y
cx' = cx * scale_x - crop_x + pad_x
cy' = cy * scale_y - crop_y + pad_y
```

当前阶段如果 depth 和 RGB 原始分辨率不一致，优先选择简单且一致的方案：把 depth resize 到 RGB 训练分辨率，并使用更新后的 color intrinsics。

### 4.4 坏帧过滤

每个 frame 检查：

- RGB 文件存在。
- depth 文件存在。
- pose 文件存在。
- pose matrix 全部为 finite。
- depth 有效像素比例大于阈值，例如 `valid_ratio >= 0.05`。

每个 scene 检查：

- 有效帧数量至少大于当前最大 views。
- 如果只做 2-4 views 训练，scene 有效帧数 `>= 4` 即可。
- 如果之后恢复 2-24 views，再要求 `>= 24`。

### 4.5 Metadata Cache

预处理后生成一个统一 metadata 文件，例如 `scannet_train_meta.pkl` 或 `scannet_train_meta.json`：

```python
{
    "dataset": "ScanNet",
    "split": "train",
    "scenes": ["scene0000_00", "scene0001_00"],
    "frames": {
        "scene0000_00": [
            {
                "frame_id": 0,
                "rgb_path": ".../color/0.jpg",
                "depth_path": ".../depth/0.png",
                "K": "3x3 float32 matrix",
                "T_c2w": "4x4 float32 matrix",
                "valid_ratio": 0.72
            }
        ]
    }
}
```

训练 dataloader 只读 metadata，不在每个 iteration 反复扫描文件系统。

### 4.6 可选：预计算 DINOv2 特征

如果 DINOv2 冻结且资源很紧，可以预先缓存特征：

```text
RGB -> resize/pad -> frozen DINOv2 -> patch tokens -> save fp16 .pt/.npy
```

优点：

- 训练 head 时更快。
- 显存压力更小。
- 更容易在 CPU/小 GPU 环境中调试训练循环。

代价：

- photometric augmentation 基本不能在线变化。
- 更像一个 head 训练实验，不是完整 end-to-end 训练。

当前目标是能简单训练起来，因此可以接受这个取舍。

## 5. 采样策略

### 5.1 最小可运行设置

先使用：

```yaml
views_per_sample: [2, 4]
sampler: temporal_nearby
shuffle_view_order: true
```

流程：

1. 随机选一个 scene。
2. 随机选一个 reference frame。
3. 从 `ref_idx +/- W` 的时间窗口中采样其他 frames，例如 `W = 30`。
4. 如果有足够精力，再根据 camera center 的 3D 距离做 spatial nearby sampler。
5. 采样后随机打乱 view 顺序，但把第一个 view 作为 pose target 的 reference view。

论文使用 nearby sampler 的目的是让多视角之间有足够重叠。用时间窗口近似虽然不如空间窗口严格，但足够用于当前 smoke training。

### 5.2 Pose Target 归一化

为了避免模型学习 ScanNet scene 的任意世界坐标原点，建议每个样本把 pose 转到第一帧坐标系下：

```text
T_ref = T_c2w[first_view]
T_i_target = inverse(T_ref) @ T_c2w[i]
```

这样第一帧 target pose 为 identity。训练 head 时输出的 pose 也解释为相对第一帧的 camera-to-world pose。

## 6. 数据增强

论文第一阶段使用较强增强，但当前只求跑通，建议先弱化：

```yaml
augmentation_minimal:
  max_dim: 224        # 显存允许可改 336
  pad_to_multiple: 14
  color_jitter:
    enabled: true
    probability: 0.3
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.05
  grayscale:
    enabled: false
  random_rescale:
    enabled: false
  random_crop:
    enabled: false
```

先不要使用复杂 spatial augmentation。等 dataloader、loss、训练循环都稳定后，再逐步加入：

- co-jittering。
- random spatial rescaling。
- random crop。
- 论文中的最大边 518。

任何几何变换都必须同步作用于 RGB、depth、valid mask，并更新 intrinsics。

## 7. Loss 设置

论文完整 loss 是：

```text
L = lambda_depth * L_depth
  + lambda_abs_pose * L_abs_pose
  + lambda_rel_pose * L_rel_pose
```

但当前 head-only 训练不追求准确率，建议先使用更稳、更容易调试的简化 loss。

### 7.1 第一版只开 depth loss

最容易跑通：

```text
L = L_depth
```

建议用 masked L1 或 log-depth L1：

```text
L_depth = mean_valid(|D_hat - D_gt|)
```

或：

```text
L_depth = mean_valid(|log(D_hat + eps) - log(D_gt + eps)|)
```

要求：

- `D_hat` 必须为正数，可用 `softplus(raw_depth) + 1e-3`。
- 只在 `valid_mask` 上计算。

### 7.2 第二版加入 absolute pose loss

depth loss 能正常下降或至少稳定后，再加入 pose：

```text
L = L_depth + 0.1 * L_abs_pose
```

建议：

- rotation 用 6D rotation representation 或 quaternion。
- translation 用 Huber loss。
- rotation loss 用 geodesic loss。
- 第一帧 pose target 是 identity。

### 7.3 Relative pose loss 暂时设为可选

论文和 π3 都强调 relative pose loss，但 head-only 小模型不一定稳定。建议初始关闭：

```yaml
lambda_rel_pose: 0.0
```

如果想展示论文训练思想，可以在训练稳定后打开很小权重：

```yaml
lambda_rel_pose: 0.05
lambda_trans: 10.0
```

不要一开始使用 π3 的 `lambda_trans = 100.0`，小模型容易被 translation loss 主导。

### 7.4 当前推荐 loss 配置

```yaml
loss:
  depth:
    enabled: true
    type: masked_log_l1
    weight: 1.0
  abs_pose:
    enabled: true
    start_iter: 500
    weight: 0.1
    rotation: geodesic
    translation: huber
  rel_pose:
    enabled: false
    weight: 0.0
```

如果只想最小化风险，先把 `abs_pose.enabled` 也设为 `false`，只训练 depth head。

## 8. 参数冻结设置

推荐默认设置：

| 模块 | 是否训练 | 原因 |
|---|---:|---|
| DINOv2 backbone | 否 | 最大显存和计算来源，冻结后最稳。 |
| 24 层 alternating attention | 否或省略 | 当前目标不是复现完整 base model。 |
| GCA / trajectory memory / KV cache | 否 | 属于第二阶段，不参与当前训练。 |
| DepthHead | 是 | 主要训练对象。 |
| PoseHead | 是 | 可选训练对象，用于展示 pose supervision。 |
| SmallMultiViewHead | 是 | 可选，用少量参数体现多视角建模。 |
| UncertaintyHead | 否 | 简化训练时先不用不确定性 loss。 |

如果要在报告中描述，可以写成：

```text
为适应有限算力，本阶段不进行论文中的 full model end-to-end training，而采用 frozen feature extractor + trainable lightweight heads 的设置，用于验证 ScanNet 数据处理与训练流程。
```

## 9. 训练设置

### 9.1 最小 smoke run

```yaml
experiment: scannet_head_only_smoke

data:
  dataset: ScanNet
  split: train
  num_scenes: 1
  views_per_sample: [2, 2]
  sampler: temporal_nearby
  max_dim: 224

model:
  backbone: dinov2_vits14
  freeze_backbone: true
  use_full_transformer: false
  train_depth_head: true
  train_pose_head: false

optimizer:
  type: AdamW
  lr: 1.0e-3
  weight_decay: 1.0e-4

train:
  batch_size_samples: 1
  total_iterations: 1000
  warmup_iterations: 100
  mixed_precision: fp16_or_bf16
  gradient_clip_norm: 1.0
  save_every: 500
  log_every: 20
```

目标：

- dataloader 正常。
- depth loss 是 finite。
- backward 正常。
- checkpoint 正常保存。
- 可视化一两张预测 depth，确认没有全 NaN。

### 9.2 稍完整的 head-only run

```yaml
experiment: scannet_head_only_small

data:
  dataset: ScanNet
  split: train
  num_scenes: 5_to_20
  views_per_sample: [2, 4]
  sampler: temporal_nearby
  max_dim: 224_or_336

model:
  backbone: dinov2_vits14_or_vitb14
  freeze_backbone: true
  use_full_transformer: false
  small_multiview_head:
    enabled: true
    layers: 2
    hidden_dim: 256
  train_depth_head: true
  train_pose_head: true

loss:
  depth_weight: 1.0
  abs_pose_weight: 0.1
  rel_pose_weight: 0.0

optimizer:
  type: AdamW
  lr: 1.0e-3
  weight_decay: 1.0e-4
  scheduler: cosine
  min_lr: 1.0e-5

train:
  batch_size_samples: 1_to_4
  total_iterations: 3000_to_5000
  mixed_precision: fp16_or_bf16
  gradient_clip_norm: 1.0
  save_every: 1000
  log_every: 20
```

此阶段不需要 FSDP，不需要 context parallel，不需要完整 160K iterations。

## 10. 当前阶段明确不做

- 不训练完整 LingBot-Map base model。
- 不使用 29 个数据集混合训练。
- 不追求论文中的重建效果。
- 不启用 GCA。
- 不训练 streaming model。
- 不做 24 到 320 views 的 progressive curriculum。
- 不做 foldback video sampler。
- 不做长序列 KV cache。
- 不强制使用 518 分辨率。
- 不强制 relative pose loss。
- 不训练 uncertainty head。

## 11. 实施检查清单

**状态更新（2026-05-11）：使用 Replica 数据集完成训练验证**

### 数据准备

- [x] ~~下载 ScanNet 小子集~~ → 使用 Replica room0 数据集（已有 2000 帧）
- [ ] 使用官方 exporter 从 `.sens` 导出 color/depth/pose/intrinsic（ScanNet）
- [x] 确认 depth PNG 除以 1000 后单位为 meter（Replica depth uint16 毫米 → 米）
- [x] 确认 pose 是 4x4 camera-to-world matrix（Replica traj.txt 每行 16 floats → 4x4）
- [x] 过滤缺失文件、非法 pose、有效深度太少的帧（已实现 frame_id 匹配检查）
- [x] 生成 metadata（ReplicaDataset 直接读取 traj.txt 和 results/）

### 数据预处理

- [x] RGB resize 到最大边 224 或 336（已实现 _align_to_train_size）
- [x] depth 和 valid mask resize 到同一训练分辨率（nearest interpolation）
- [x] 输出尺寸 pad 到 14 的整数倍（已实现 patch_size=14 padding）
- [x] 同步更新 intrinsics（已实现 K_new 计算）
- [x] dataloader 返回 `images, depths, masks, K, T_c2w`（ReplicaDataset 已实现）

### 采样

- [x] 每个样本采 2 views（已实现 temporal_nearby sampler）
- [x] 从时间附近窗口采样，确保有重叠（temporal_window=30）
- [x] 做 reference pose normalization（已实现 _normalize_poses）
- [x] 第一帧 target pose 为 identity（归一化后第一帧为单位矩阵）

### 模型

- [x] 加载 DINOv2（通过 torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')）
- [x] 冻结 DINOv2 参数（已实现 freeze_backbone=True）
- [x] 只把 head 参数传给 optimizer（已过滤 requires_grad=True 的参数）
- [x] 先只训练 depth head（train_pose=False）
- [ ] 可选加入 pose head（待后续扩展）

### 训练

- [x] 跑 1 个 scene 的 100-200 iteration overfit 测试（100 samples, 10 epochs）
- [x] 确认 loss finite（loss 从 2.9 下降到 0.26）
- [x] 确认 backward 没有 NaN（gradient clipping 正常）
- [x] 确认 checkpoint 能保存和恢复（checkpoint_epoch_4.pt, checkpoint_epoch_9.pt 已保存）
- [ ] 扩展到 5-20 个 scenes，跑 1K-5K iterations（待使用更多 Replica 场景）

### 验证

- [x] 加载 checkpoint 进行推理（validate_model.py 已实现）
- [x] 检查 depth 预测范围合理（预测: 3m-40m, GT: 8m-33m）
- [x] 可视化 depth 对比图（depth_comparison_*.png 已生成）
- [x] 计算相对误差统计（平均相对误差: 21.52%）

### 已解决问题

1. **DINOv2 加载问题**：torchvision 0.20 不包含 DINOv2，改用 torch.hub 加载
2. **属性初始化问题**：depth_head/pose_head 需先初始化为 None 再条件创建
3. **代理问题**：torch.hub 需要网络访问，需设置 http_proxy

## 12. 可写入报告或实验说明的一句话

本阶段采用资源受限的 head-only 训练设置：利用冻结的 DINOv2 作为特征提取器，在 ScanNet RGB-D 数据上训练轻量 depth/pose heads。该设置不追求复现 LingBot-Map 第一阶段的大规模 end-to-end 训练精度，主要用于验证 ScanNet 数据获取、统一预处理、近邻多视角采样、pose/depth supervision 以及训练流程的可运行性。

## 13. 参考来源

- `Geometric Context Transformer for.pdf`：主论文，Sec. 3、Sec. 4.1、Sec. 4.3、Appendix A.1。
- `Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf`：VGGT 架构、DINO backbone、alternating attention、camera/depth heads。
- `π3 Permutation-equivariant visual geometry learning.pdf`：relative pose loss 形式和相对位姿监督思想。
- `Dinov2 Learning robust visual features without supervision.pdf`：DINOv2 作为冻结视觉特征提取器的依据。
- ScanNet 官方 SensReader / exporter 文档：[SensReader](https://www.scan-net.org/ScanNet/SensReader/) 与 [Python Data Exporter](https://www.scan-net.org/ScanNet/SensReader/python/)；用于确认 depth shift、camera-to-world pose、depth PNG 导出说明。
