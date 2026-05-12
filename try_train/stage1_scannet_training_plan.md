# ScanNet Head-only 训练方案：进一步贴近论文 4.1 Base Model Training

本文档基于当前文件夹中的主论文 `Geometric Context Transformer for.pdf`，以及 VGGT、π3、DINOv2 论文整理，并结合本次已跑通的训练验证记录 `README.md` 与 `stage1_scannet_training_plan_modified.md` 修改而来。

当前目标不是训练完整 LingBot-Map 全网络，而是在资源受限条件下，尽量贴近论文第一个阶段 `4.1 Base Model Training` 的数据、采样、增强、优化策略，同时仍然采用 head-only 训练：

- 使用 ScanNet 公开 RGB-D 数据。
- 不训练第二阶段 Streaming/GCA。
- 冻结 DINOv2 backbone 和大部分主体网络。
- 训练深度头 `DepthHead` 和位姿头 `PoseHead`。
- 可选训练轻量多视角 head，用较小参数量模拟论文第一阶段的 global cross-frame reasoning。
- 训练目标从“只要跑通”升级为“按论文第一阶段设置组织数据和训练流程，但只更新 head 参数”。

## 1. 已验证结果与本次修改方向

本次已有验证说明：

- 已使用 head-only 训练框架完成 smoke training。
- 使用 DINOv2 `dinov2_vits14` 作为冻结特征提取器，通过 `torch.hub` 加载。
- 已验证 RGB/depth/pose dataloader、resize/pad、valid mask、pose normalization、checkpoint 保存、checkpoint 加载与深度推理可视化。
- Replica room0 测试中，100 samples、10 epochs、2 views、224 resolution 能完成训练；loss 从约 `2.91` 下降到约 `0.26`。
- 当前验证主要启用了 depth head，pose head 已在模型结构和训练配置中预留，但后续需要默认启用。

因此，本文档的修改重点是：

- 从“最小 smoke run”转向“ScanNet stage1-head 训练”。
- 默认训练 `DepthHead + PoseHead`，而不是只训练 depth head。
- 视角采样从 `2 views` 扩展到论文第一阶段的 `2-24 views`。
- 分辨率从 `224/336` 逐步扩展到论文设置的最大边 `518`。
- 优化器从 smoke run 的 `lr=1e-3, wd=1e-4` 改为论文第一阶段的 `AdamW, lr=2e-4, wd=0.05`。
- 学习率调度改为论文第一阶段的 `5% warmup + cosine decay to 1e-8`。
- 数据增强从弱增强升级到论文第一阶段增强策略。

## 2. 与论文 4.1 的对应关系

论文第一阶段 `Base Model Training` 的完整设置：

| 项目 | 论文 4.1 设置 | 当前 head-only 复现设置 |
|---|---|---|
| 数据 | 29 个公开/合成/内部数据集混合 | 只使用 ScanNet |
| 模型 | DINOv2 + 24 个 alternating attention blocks + heads | 冻结 DINOv2/主体，只训练 depth/pose heads |
| Attention | global attention，不使用 GCA | 不使用 GCA；可选轻量 multi-view head |
| 输入 views | 每个样本随机 2-24 views | 保持 2-24 views |
| 图像尺寸 | 最大边 518 | 目标使用 518，显存不足可先 224/336 |
| Augmentation | 强 photometric + spatial rescale + co-jitter | 目标采用同样策略，可分阶段打开 |
| Optimizer | AdamW | AdamW |
| LR | `2e-4` | `2e-4`，仅作用于 heads |
| Weight decay | `0.05` | `0.05`，仅作用于 heads |
| Schedule | 前 5% 从 `1e-8` warmup 到 base lr，后 95% cosine 到 `1e-8` | 保持一致 |
| Iterations | 160K | 推荐最终仍按 160K 配置；资源不足先跑 5K-20K pilot |
| 分布式 | FSDP + checkpointing + bf16 | 单卡或小规模训练，不强制 FSDP |

这份方案可以写成：

```text
We follow the first-stage data sampling, augmentation, and optimization recipe of Sec. 4.1, but freeze the pretrained visual backbone and train only lightweight depth and camera heads due to limited compute.
```

## 3. 模型设置

### 3.1 Head-only 总体结构

推荐结构：

```text
RGB images: [B, V, 3, H, W]
  -> frozen DINOv2 ViT-S/14 or ViT-B/14
  -> patch tokens + cls token
  -> optional lightweight multi-view head
  -> DepthHead
  -> PoseHead
```

其中：

- `B`：batch size。
- `V`：每个样本的 view 数，训练时从 `[2, 24]` 随机采样。
- `H, W`：resize/pad 后的训练分辨率，目标最大边为 `518`，并 pad 到 14 的整数倍。

### 3.2 冻结与训练参数

| 模块 | 是否训练 | 说明 |
|---|---:|---|
| DINOv2 backbone | 否 | 使用预训练视觉特征，减少显存和训练成本。 |
| 原始 24 层 alternating attention | 否或不加载 | 当前不训练完整 base model。 |
| GCA / trajectory memory / KV cache | 否 | 第二阶段 streaming model 组件，本阶段不用。 |
| DepthHead | 是 | 必训，预测每帧 metric depth。 |
| PoseHead | 是 | 必训，预测每帧相对第一帧的 camera-to-world pose。 |
| Lightweight MultiViewHead | 可选训练 | 用 1-2 层小 Transformer/MLP 在 views 间交换信息，近似论文第一阶段 global attention 思路。 |
| Uncertainty branch | 可选训练 | 如果实现方便，可作为 depth head 的一部分输出 `log_sigma`；否则先用 masked log-L1。 |

当前阶段的核心变化是：`PoseHead` 不再是可选项，而是与 `DepthHead` 一起训练。

### 3.3 推荐 head 结构

```yaml
model:
  backbone:
    name: dinov2_vits14
    patch_size: 14
    frozen: true
    load_method: torch_hub

  multi_view_head:
    enabled: true
    layers: 2
    hidden_dim: 256
    num_heads: 4
    trainable: true
    note: "资源不足时可关闭；关闭后每帧主要依赖 frozen DINOv2 features"

  depth_head:
    enabled: true
    type: dpt_or_lightweight_decoder
    trainable: true
    output_activation: exp_or_softplus
    output: depth_meters

  pose_head:
    enabled: true
    type: camera_head_mlp
    trainable: true
    output: relative_camera_to_world_pose
    rotation_representation: quaternion_or_6d
    translation_representation: xyz
```

输出约定：

- `DepthHead` 输出 `D_hat_i`，单位为米。
- `PoseHead` 输出 `T_i_pred`，表示第 `i` 个 view 相对当前样本第一帧的 camera-to-world pose。
- 第一帧 pose target 是 identity。

## 4. ScanNet 数据获取与导出

### 4.1 原始数据

使用 ScanNet 公开 RGB-D sequence。通常每个 scene 下载为 `.sens` 文件，需要用官方 exporter 导出：

- RGB 图像：`color/*.jpg`
- 深度图：`depth/*.png`
- 位姿：`pose/*.txt`
- 内参：`intrinsic/*.txt`

推荐目录结构：

```text
scannet/
  scans/
    scene0000_00/
      scene0000_00.sens
      color/
        0.jpg
      depth/
        0.png
      pose/
        0.txt
      intrinsic/
        intrinsic_color.txt
        intrinsic_depth.txt
```

### 4.2 从验证方案迁移到 ScanNet

本次验证中 Replica 数据流已经跑通：

- `traj.txt` 解析为 4x4 camera-to-world matrix。
- `depth uint16` 除以 1000 转为米。
- RGB/depth resize 后 pad 到 14 的整数倍。
- 采样 2 views temporal nearby。
- pose 做 reference normalization。

ScanNet 数据处理与之高度一致，主要差异是：

- pose 文件来自 `pose/*.txt`。
- intrinsics 来自 `intrinsic/*.txt`。
- scene split 应使用 ScanNet 官方 train split。
- 最终训练需要扩展到更多 scenes 和 2-24 views。

## 5. 数据预处理流程

### 5.1 坐标系统一

论文 Appendix A.1 要求统一为 camera-to-world representation。ScanNet exporter 导出的 pose 通常已经是 4x4 `T_c2w`，训练 metadata 中统一保存：

```text
T_c2w: shape [4, 4]
```

不要在 metadata 中混用 world-to-camera。如果模型内部需要 world-to-camera，应在 dataloader 或 loss 中显式转换。

### 5.2 深度单位转换

ScanNet depth 是 16-bit PNG，单位为毫米：

```python
depth_m = depth_png.astype("float32") / 1000.0
valid_mask = depth_m > 0
```

处理规则：

- `depth <= 0` 视为 invalid。
- `NaN` / `Inf` 置为 `0`。
- loss 只在 `valid_mask` 上计算。
- 当前阶段不需要 sky mask。
- 当前阶段不强制 98th percentile clamp；如果出现异常深度，可对 `depth > 20m` 或 scene-specific outlier 做裁剪。

### 5.3 RGB、Depth、Mask、Intrinsics 对齐

训练输入需要 RGB、depth、mask、intrinsics 在同一图像坐标系下。

推荐流程：

1. 读取 RGB、depth、`K_color`、`T_c2w`。
2. 将 RGB resize 到目标最大边。
3. 将 depth resize 到同一训练尺寸。
4. 将 valid mask resize 到同一训练尺寸。
5. 将输出尺寸 pad 到 14 的整数倍。
6. 同步更新内参。

内参更新：

```text
fx' = fx * scale_x
fy' = fy * scale_y
cx' = cx * scale_x - crop_x + pad_x
cy' = cy * scale_y - crop_y + pad_y
```

插值建议：

- RGB：bilinear。
- depth：nearest 或 area。为了避免生成不存在的深度值，优先 nearest。
- valid mask：nearest。

### 5.4 坏帧过滤

每个 frame 检查：

- RGB 文件存在。
- depth 文件存在。
- pose 文件存在。
- pose matrix 全部为 finite。
- depth 有效像素比例大于阈值，例如 `valid_ratio >= 0.05`。

每个 scene 检查：

- 若目标使用 2-24 views，scene 有效帧数建议 `>= 24`。
- 若先做 pilot，可临时要求 `>= 4`。

### 5.5 Metadata Cache

生成统一 metadata，例如 `scannet_train_meta.json` 或 `scannet_train_meta.pkl`：

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

训练阶段只读 metadata，避免每个 iteration 扫描原始目录。

## 6. 视角采样策略

### 6.1 论文第一阶段采样原则

论文 Sec. 4.3 的 Stage 1 使用 nearby sampler：

- 每个 iteration 随机采 scene。
- 每个样本随机采 `2-24` views。
- 先选 reference frame。
- 剩余 views 从 reference 附近的空间窗口采样。
- 不强制时间顺序。
- dynamic batch sampler 每 GPU 最多 packing 48 images。

### 6.2 当前 head-only 复现采样

目标配置：

```yaml
sampling:
  views_per_sample: [2, 24]
  sampler: spatial_nearby
  fallback_sampler: temporal_nearby
  temporal_window: 30
  shuffle_view_order: true
  max_images_per_gpu: 48
```

实现建议：

- 优先使用 camera center 的 3D 距离做 `spatial_nearby`。
- 如果暂时没有空间索引，使用已验证过的 `temporal_nearby` 作为 fallback。
- 采样后随机打乱 view 顺序。
- 打乱后的第一帧作为当前样本 reference view。
- 如果 batch 中 view 数不同，使用 dynamic batch packing：保证 `batch_size_samples * views_per_sample <= 48`。

示例：

```text
V = random_integer(2, 24)
max_samples_this_batch = floor(48 / V)
```

### 6.3 Pose reference normalization

每个样本以第一帧为参考坐标系：

```text
T_ref = T_c2w[first_view]
T_i_target = inverse(T_ref) @ T_c2w[i]
```

这样：

- 第一帧 target pose 为 identity。
- 所有 pose supervision 都与样本内部 reference frame 对齐。
- 模型不用学习 ScanNet scene 的任意 global origin。

## 7. 数据增强

### 7.1 论文 4.1 增强策略

目标逐步对齐论文：

```yaml
augmentation_stage1_target:
  image_max_dim: 518
  pad_to_multiple: 14
  color_jitter:
    probability: 0.9
    brightness: 0.5
    contrast: 0.5
    saturation: 0.5
    hue: 0.1
  grayscale:
    probability: 0.05
  spatial_rescale:
    enabled: true
    scale_range: [0.8, 1.2]
    aspect_ratio_range: [0.33, 1.0]
  co_jitter:
    probability: 0.3
```

co-jitter 含义：

- 以 `0.3` 概率，对同一个 multi-view sample 内所有 frames 使用同一组颜色变换。
- 其余情况下，各 frame 使用独立颜色变换。
- 目的是防止模型只依赖外观一致性，让模型更多学习几何线索。

### 7.2 推荐打开顺序

由于已验证的版本使用较弱增强，建议分阶段打开：

| 阶段 | 分辨率 | 增强 | 目标 |
|---|---:|---|---|
| Pilot A | 224 | color jitter 0.3 | 确认 depth+pose head 同训稳定 |
| Pilot B | 336 | color jitter 0.5 + grayscale 0.05 | 扩展 views 和 scenes |
| Target | 518 | 论文完整增强 | 对齐 Sec. 4.1 |

任何 geometric augmentation 都必须同步作用于 RGB、depth、valid mask，并更新 intrinsics。

## 8. Loss 设置

论文 Sec. 3.3 的总体 loss：

```text
L = lambda_depth * L_depth
  + lambda_abs_pose * L_abs_pose
  + lambda_rel_pose * L_rel_pose
```

当前 head-only 训练默认启用 depth loss 和 absolute pose loss。relative pose loss 用较小权重逐步加入。

### 8.1 Depth loss

已验证版本使用 masked log-L1，稳定且适合 head-only：

```text
L_depth = mean_valid(|log(D_hat + eps) - log(D_gt + eps)|)
```

要求：

- `D_hat` 必须为正数，推荐 `exp` 或 `softplus + eps`。
- 只在 valid mask 上计算。
- `eps` 可取 `1e-3`。

更贴近论文的可选版本：

```text
L_depth = Sigma_D * |D_hat - D|
        + Sigma_D * |grad(D_hat) - grad(D)|
        - alpha * log(Sigma_D)
```

如果实现不方便，先保留 masked log-L1。若要更贴近论文，可把 uncertainty branch 作为 depth head 的一部分加入。

### 8.2 Absolute pose loss

当前目标默认训练 pose head。

target pose：

```text
T_i_target = inverse(T_ref) @ T_c2w[i]
```

推荐 loss：

```text
L_abs_pose = L_rot + lambda_trans_abs * L_trans
```

其中：

- `L_rot`：rotation geodesic loss。
- `L_trans`：Huber loss 或 L1 loss。
- 第一帧 target pose 为 identity，也参与监督或作为固定 reference 忽略均可。若第一帧输出恒为 identity，可不计算第一帧 pose loss。

初始权重：

```yaml
abs_pose:
  enabled: true
  weight: 0.1
  lambda_trans_abs: 1.0
```

### 8.3 Relative pose loss

论文在 local window 中使用 relative pose loss。第一阶段没有 GCA sliding window，可在当前 sampled views 内计算所有 pair：

```text
T_i_to_j_target = inverse(T_i_target) @ T_j_target
T_i_to_j_pred = inverse(T_i_pred) @ T_j_pred
```

loss：

```text
L_rel_pose = mean_{i != j}(L_rot(i, j) + lambda_trans_rel * L_trans(i, j))
```

推荐先使用较小权重：

```yaml
rel_pose:
  enabled: true
  start_iter: 1000
  weight: 0.05
  lambda_trans_rel: 10.0
```

说明：

- π3 中 `lambda_trans = 100.0`，但 head-only 小模型容易被 translation loss 主导。
- 当前更稳的做法是从 `10.0` 开始，观察 loss magnitude 后再调。

### 8.4 推荐总 loss 配置

```yaml
loss:
  depth:
    enabled: true
    type: masked_log_l1
    weight: 1.0

  abs_pose:
    enabled: true
    weight: 0.1
    rotation: geodesic
    translation: huber
    lambda_trans_abs: 1.0

  rel_pose:
    enabled: true
    start_iter: 1000
    weight: 0.05
    rotation: geodesic
    translation: huber
    lambda_trans_rel: 10.0
```

若训练早期出现 `inf` 或 `NaN`，处理顺序：

1. 检查 depth prediction 是否强制为正。
2. 加大 `eps`，例如从 `1e-3` 到 `1e-2`。
3. 暂时关闭 `rel_pose`。
4. 将 `abs_pose.weight` 从 `0.1` 降到 `0.05`。
5. 保持 `gradient_clip_norm = 1.0`。

## 9. 训练配置

### 9.1 ScanNet head-only pilot

用于从已验证的 Replica smoke 迁移到 ScanNet：

```yaml
experiment: scannet_stage1_head_pilot

data:
  dataset: ScanNet
  split: train
  num_scenes: 5_to_20
  views_per_sample: [2, 8]
  sampler: temporal_nearby
  temporal_window: 30
  image_max_dim: 336
  pad_to_multiple: 14

model:
  backbone: dinov2_vits14
  freeze_backbone: true
  train_depth_head: true
  train_pose_head: true
  multi_view_head:
    enabled: true
    layers: 2
    hidden_dim: 256

optimizer:
  type: AdamW
  lr: 2.0e-4
  weight_decay: 0.05

scheduler:
  type: warmup_cosine
  warmup_ratio: 0.05
  min_lr: 1.0e-8

train:
  total_iterations: 5000_to_20000
  max_images_per_gpu: 24
  mixed_precision: fp16_or_bf16
  gradient_clip_norm: 1.0
  save_every: 1000
  log_every: 20
```

### 9.2 论文 4.1 对齐版 Head-only 目标配置

这是当前阶段最终建议配置：

```yaml
experiment: scannet_stage1_head_paper_aligned

data:
  dataset: ScanNet
  split: train
  views_per_sample: [2, 24]
  sampler: spatial_nearby
  fallback_sampler: temporal_nearby
  shuffle_view_order: true
  max_images_per_gpu: 48

image:
  max_dim: 518
  pad_to_multiple: 14

augmentation:
  color_jitter_probability: 0.9
  brightness: 0.5
  contrast: 0.5
  saturation: 0.5
  hue: 0.1
  grayscale_probability: 0.05
  spatial_rescale: [0.8, 1.2]
  aspect_ratio: [0.33, 1.0]
  co_jitter_probability: 0.3

model:
  backbone: dinov2_vits14
  freeze_backbone: true
  use_gca: false
  train_depth_head: true
  train_pose_head: true
  train_uncertainty_head: false
  multi_view_head:
    enabled: true
    layers: 2
    hidden_dim: 256
    num_heads: 4

optimizer:
  type: AdamW
  lr: 2.0e-4
  weight_decay: 0.05

scheduler:
  type: linear_warmup_cosine_decay
  warmup_ratio: 0.05
  start_lr: 1.0e-8
  base_lr: 2.0e-4
  min_lr: 1.0e-8

loss:
  depth_weight: 1.0
  abs_pose_weight: 0.1
  rel_pose_weight: 0.05

train:
  total_iterations: 160000
  mixed_precision: bf16_if_available_else_fp16
  gradient_clip_norm: 1.0
  save_every: 5000
  log_every: 20
```

资源不足时可以只减少：

- `total_iterations`
- `image.max_dim`
- `views_per_sample` 上限
- `max_images_per_gpu`

不建议再关闭 pose head，因为当前文档目标就是 depth head 和 pose head 同训。

## 10. 建议训练流程

### Step 1：ScanNet 数据 smoke

目的：确认 ScanNet exporter、metadata、dataloader 没问题。

```yaml
num_scenes: 1
views_per_sample: [2, 2]
image_max_dim: 224
train_depth_head: true
train_pose_head: true
rel_pose_weight: 0.0
iterations: 500_to_1000
```

验收：

- loss finite。
- `depth_loss` 可计算。
- `pose_loss` 可计算。
- checkpoint 可保存。

### Step 2：启用多视角和 pose

```yaml
num_scenes: 5_to_20
views_per_sample: [2, 8]
image_max_dim: 336
train_pose_head: true
rel_pose_weight: 0.0
iterations: 5000
```

验收：

- pose rotation loss 不出现 NaN。
- translation loss 量级不压过 depth loss。
- 第一帧 pose target 确认为 identity。

### Step 3：加入 relative pose loss

```yaml
views_per_sample: [2, 8]
rel_pose_weight: 0.05
rel_pose_start_iter: 1000
lambda_trans_rel: 10.0
```

验收：

- relative pose loss finite。
- 总 loss 不被 relative translation 主导。

### Step 4：对齐论文第一阶段

```yaml
views_per_sample: [2, 24]
image_max_dim: 518
augmentation: paper_stage1
optimizer: AdamW_lr_2e-4_wd_0.05
scheduler: 5_percent_warmup_cosine
iterations: up_to_160000
```

验收：

- 训练循环稳定。
- checkpoint 能定期保存。
- 验证脚本能加载 depth/pose heads。
- 可视化 depth 不为全零、全 NaN 或全常数。

## 11. 验证与记录

### 11.1 已验证内容

来自本次 README 记录：

- DINOv2 torch.hub 加载可用。
- frozen backbone + depth head 可训练。
- AMP、gradient clipping、checkpoint 保存可用。
- dataloader 能输出 RGB、depth、mask、pose。
- depth 预测可视化流程可用。

### 11.2 下一步需要验证

为满足当前文档目标，需要新增验证：

- `train_pose=true` 时，pose head 正常参与训练。
- `abs_pose_loss` 正常下降或至少保持 finite。
- `relative pose loss` 打开后不导致 NaN。
- ScanNet metadata exporter 正常生成 train split。
- ScanNet 2-24 views sampler 正常工作。
- 518 resolution 下显存可接受；若不可接受，记录降级到 336 的原因。

### 11.3 建议记录指标

当前不追求 SOTA，但建议记录：

- `loss_total`
- `loss_depth`
- `loss_abs_pose`
- `loss_rel_pose`
- `rotation_error_deg`
- `translation_error_m`
- valid depth ratio
- depth prediction min/max/mean
- checkpoint path
- training config yaml

## 12. 当前阶段明确不做

- 不训练完整 LingBot-Map base model。
- 不训练 DINOv2 backbone。
- 不训练 24 层完整 alternating attention。
- 不启用 GCA。
- 不训练 streaming model。
- 不做 24 到 320 views 的第二阶段 curriculum。
- 不做 foldback video sampler。
- 不做长序列 KV cache。
- 不追求论文最终 reconstruction 指标。

注意：不做这些并不影响当前目标，因为当前目标是“第一阶段训练策略的 head-only 近似复现”，重点是把论文 4.1 的数据、采样、增强和优化设置迁移到已验证的轻量训练框架中。

## 13. 实施检查清单

### 数据

- [ ] 下载 ScanNet train split。
- [ ] 用官方 exporter 从 `.sens` 导出 color/depth/pose/intrinsic。
- [ ] depth PNG 除以 1000 转为 meter。
- [ ] pose 统一为 4x4 camera-to-world。
- [ ] 过滤无效 pose、缺失文件、有效深度过低的帧。
- [ ] 生成 `scannet_train_meta.json` 或 `scannet_train_meta.pkl`。

### 预处理

- [ ] RGB/depth/mask resize 到一致尺寸。
- [ ] 输出尺寸 pad 到 14 的整数倍。
- [ ] 同步更新 intrinsics。
- [ ] dataloader 返回 `images, depths, masks, K, T_c2w`。
- [ ] 每个样本做 reference pose normalization。

### 采样

- [ ] 支持 `views_per_sample=[2, 24]`。
- [ ] 支持 temporal nearby fallback。
- [ ] 支持 spatial nearby sampler。
- [ ] 支持 view order shuffle。
- [ ] 支持 dynamic batch packing，限制每 GPU 最多 48 images。

### 模型

- [ ] 加载 DINOv2 `dinov2_vits14`。
- [ ] 冻结 DINOv2 参数。
- [ ] 启用 DepthHead。
- [ ] 启用 PoseHead。
- [ ] optimizer 只接收 trainable heads 参数。
- [ ] 可选启用 2 层 lightweight MultiViewHead。

### Loss

- [ ] depth loss 使用 valid mask。
- [ ] depth prediction 强制为正。
- [ ] pose loss 使用 reference-normalized target。
- [ ] rotation loss 使用 geodesic。
- [ ] translation loss 使用 Huber。
- [ ] relative pose loss 延迟开启。

### 训练

- [ ] ScanNet 1 scene smoke run。
- [ ] ScanNet 5-20 scenes pilot。
- [ ] 启用 pose head 同训。
- [ ] 启用 2-24 views。
- [ ] 启用 518 max dimension。
- [ ] 使用 AdamW `lr=2e-4, wd=0.05`。
- [ ] 使用 5% warmup + cosine decay。
- [ ] 保存并验证 checkpoint。

## 14. 可写入报告的总结

本阶段在已验证的 head-only 框架基础上，进一步向 LingBot-Map 论文第 4.1 节 Base Model Training 对齐。由于算力受限，DINOv2 backbone 和主体网络保持冻结，只训练深度头与位姿头；训练数据使用 ScanNet RGB-D，遵循论文 Appendix A.1 的数据统一流程，将 depth 转为米、pose 统一为 camera-to-world，并采用 reference-normalized pose target。训练策略上保留第一阶段的 2-24 views nearby sampling、最大边 518、强 photometric augmentation、AdamW `2e-4`、weight decay `0.05`、5% warmup 和 cosine decay。该方案不追求完整论文精度，而用于在有限资源下复现第一阶段的数据组织、监督信号和训练设置。

## 15. 参考来源

- `Geometric Context Transformer for.pdf`：主论文，Sec. 3、Sec. 4.1、Sec. 4.3、Appendix A.1。
- `Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf`：VGGT 架构、DINO backbone、alternating attention、camera/depth heads。
- `π3 Permutation-equivariant visual geometry learning.pdf`：relative pose loss 和相对位姿监督思想。
- `Dinov2 Learning robust visual features without supervision.pdf`：DINOv2 作为冻结视觉特征提取器的依据。
- `README.md`：本次 head-only 训练验证记录。
- `stage1_scannet_training_plan_modified.md`：上一版资源受限训练计划。
- ScanNet 官方 SensReader / exporter 文档：[SensReader](https://www.scan-net.org/ScanNet/SensReader/) 与 [Python Data Exporter](https://www.scan-net.org/ScanNet/SensReader/python/)。
