# Replica Head-only 训练方案：从论文 4.1 Base Model 到 4.2 Streaming 长序列训练

本文档基于当前文件夹中的主论文 `Geometric Context Transformer for.pdf`，以及 VGGT、π3、DINOv2 论文整理，并结合本次已跑通的训练验证记录 `README.md` 与 `stage1_scannet_training_plan_modified.md` 修改而来。

当前目标不是训练完整 LingBot-Map 全网络，而是在资源受限条件下，先完成论文第一个阶段 `4.1 Base Model Training` 的 head-only 验证，再进一步吸收第二阶段 `4.2 Streaming Model Training` 的长序列训练策略。由于本地已下载 Replica 数据，后续验证使用 Replica 替代 ScanNet；文件名保留 `stage1_scannet_training_plan.md`，但当前实际训练数据以 Replica 为准。

- 使用 Replica RGB-D 长序列数据进行验证，ScanNet 流程保留为可迁移参考。
- 第二阶段不训练完整 GCA/streaming 主干，只迁移其长序列 curriculum、foldback sampler、relative pose window 等训练思想。
- 冻结 DINOv2 backbone 和大部分主体网络。
- 训练深度头 `DepthHead` 和位姿头 `PoseHead`。
- 不训练额外 lightweight multi-view head；若代码中存在该模块，默认禁用或冻结，optimizer 只接收两个 heads 的参数。
- 训练目标从“只要跑通”升级为“先对齐论文 4.1 的 head-only 第一阶段，再用 Replica 长序列验证论文 4.2 的 streaming 训练策略，但仍只更新两个 heads”。

## 1. 已验证结果与本次修改方向

本次已有验证说明：

- 已使用 head-only 训练框架完成 smoke training。
- 使用 DINOv2 `dinov2_vits14` 作为冻结特征提取器，通过 `torch.hub` 加载。
- 已验证 RGB/depth/pose dataloader、resize/pad、valid mask、pose normalization、checkpoint 保存、checkpoint 加载与深度推理可视化。
- Replica room0 测试中，100 samples、10 epochs、2 views、224 resolution 能完成训练；loss 从约 `2.91` 下降到约 `0.26`。
- 当前验证主要启用了 depth head，pose head 已在模型结构和训练配置中预留，但后续需要默认启用。

因此，本文档的修改重点是：

- 从“最小 smoke run”转向“Replica stage1-head 训练”，并继续扩展到 “Replica stage2 long-sequence head 训练”。
- 默认训练 `DepthHead + PoseHead`，而不是只训练 depth head。
- 视角采样从 `2 views` 扩展到论文第一阶段的 `2-24 views`。
- 分辨率从 `224/336` 逐步扩展到论文设置的最大边 `518`。
- 优化器从 smoke run 的 `lr=1e-3, wd=1e-4` 改为论文第一阶段的 `AdamW, lr=2e-4, wd=0.05`。
- 学习率调度改为论文第一阶段的 `5% warmup + cosine decay to 1e-8`。
- 数据增强从弱增强升级到论文第一阶段增强策略。
- 第二阶段从已验证的第一阶段 head checkpoint 初始化，使用 Replica 长序列、foldback sampler、views curriculum 和局部 relative pose window 继续训练两个 heads。

## 2. 与论文 4.1 的对应关系

论文第一阶段 `Base Model Training` 的完整设置：

| 项目 | 论文 4.1 设置 | 当前 head-only 复现设置 |
|---|---|---|
| 数据 | 29 个公开/合成/内部数据集混合 | 当前使用 Replica，ScanNet 作为可迁移参考 |
| 模型 | DINOv2 + 24 个 alternating attention blocks + heads | 冻结 DINOv2/主体，只训练 depth/pose heads |
| Attention | global attention，不使用 GCA | 不使用 GCA；不训练额外 multi-view head |
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

### 2.1 与论文 4.2 Streaming Model Training 的对应关系

论文第二阶段的目的，是把第一阶段得到的 base model 迁移到 streaming/long-sequence setting。主论文 4.2 的核心训练策略如下：

- **Initialization from Base Model**：从第一阶段 base model checkpoint 初始化 streaming model。
- **Attention replacement**：将第一阶段 global attention 替换为 GCA；由于 Q/K/V projection 参数化一致，权重可以直接迁移。
- **Optimization**：训练 `160K iterations`，base learning rate 改为 `5e-4`，仍使用第一阶段相同的 5% warmup + cosine decay 到 `1e-8`。
- **Progressive view curriculum**：训练 views 数从 `24` 线性增加到 `320`。
- **Local pose reference window**：GCA 的局部参考窗口 `k` 在 `[16, 64]` 中随机采样。
- **Context parallelism**：论文使用 Ulysses context parallel，parallel dimension 为 `16`，以支持长序列训练。
- **Long-trajectory data**：第二阶段转向长轨迹视频数据，并用 foldback video sampler 生成时序连续但无固定前进偏置的训练片段。

当前 head-only 版本不能也不需要完整训练 GCA 主干，因此对 4.2 做如下改写：

| 项目 | 论文 4.2 设置 | Replica head-only 第二阶段设置 |
|---|---|---|
| 初始化 | 从第一阶段 base checkpoint 初始化 | 从已验证的 stage1 head checkpoint 初始化 |
| GCA | 替换 global attention 为 GCA 并训练 streaming model | 不训练完整 GCA；不训练额外 window head，仅用长序列采样和局部 relative pose loss 迁移训练思想 |
| 训练模块 | 全 streaming model | 仍只训练 `DepthHead + PoseHead` |
| 数据 | 长轨迹视频数据 | 使用 Replica room 序列作为长轨迹 RGB-D 数据 |
| Views curriculum | `24 -> 320` 线性增加 | 目标保持 `24 -> 320`，显存不足时先用 `8 -> 64` 或 `16 -> 128` pilot |
| Local window k | `[16, 64]` 随机采样 | 用于 relative pose loss 的局部窗口，先从 `[8, 32]` pilot，再扩展到 `[16, 64]` |
| Optimizer | AdamW, `lr=5e-4` | AdamW, `lr=5e-4`，仅作用于两个 heads |
| Schedule | 5% warmup + cosine | 保持一致 |
| Context parallel | Ulysses, dimension 16 | 不强制；用小分辨率、gradient accumulation 和截断长序列控制显存 |

这份第二阶段 head-only 方案可以写成：

```text
We initialize the two trainable heads from the stage-1 head checkpoint and follow the stage-2 long-sequence curriculum of Sec. 4.2 on Replica videos, while keeping the pretrained backbone frozen and omitting full GCA/context-parallel training.
```

## 3. 模型设置

### 3.1 Head-only 总体结构

推荐结构：

```text
RGB images: [B, V, 3, H, W]
  -> frozen DINOv2 ViT-S/14 or ViT-B/14
  -> patch tokens + cls token
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
| GCA / trajectory memory / KV cache | 否 | 第二阶段只借鉴长序列训练策略，不训练完整 GCA/KV cache 主干。 |
| DepthHead | 是 | 必训，预测每帧 metric depth。 |
| PoseHead | 是 | 必训，预测每帧相对第一帧的 camera-to-world pose。 |
| Lightweight MultiViewHead | 否 | 为满足“只训练两个 heads”，默认禁用或冻结，不传入 optimizer。 |
| Uncertainty branch | 否 | 当前只训练 depth/pose 两个 heads，先使用 masked log-L1，不单独训练 uncertainty branch。 |

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
    enabled: false
    trainable: false
    note: "当前实验只训练 DepthHead 与 PoseHead，不训练额外 multi-view head"

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

## 4. 数据获取与导出：Replica 当前训练，ScanNet 可迁移参考

### 4.0 当前实际数据选择

当前验证和后续第二阶段长序列训练使用本地已下载的 Replica 数据集，而不是 ScanNet。这个替换不改变训练策略本身：Replica 同样提供 RGB、metric depth 和 camera-to-world trajectory，足以验证 `DepthHead + PoseHead` 的监督、长序列 sampling、progressive view curriculum 和 relative pose window。

Replica 典型目录结构：

```text
Replica/room0/
  traj.txt
  results/
    frame000000.jpg
    frame000001.jpg
    depth000000.png
    depth000001.png
```

Replica 处理规则：

- `traj.txt`：每行 16 个浮点数，解析为 4x4 `T_c2w`。
- `frame*.jpg`：RGB 输入。
- `depth*.png`：uint16 毫米深度，除以 1000 转为米。
- `room0` 或其他 room 序列可直接作为长轨迹视频数据；第二阶段不再需要 unordered multi-view collection。

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

### 4.2 Replica 与 ScanNet 的对应关系

本次验证中 Replica 数据流已经跑通：

- `traj.txt` 解析为 4x4 camera-to-world matrix。
- `depth uint16` 除以 1000 转为米。
- RGB/depth resize 后 pad 到 14 的整数倍。
- 采样 2 views temporal nearby。
- pose 做 reference normalization。

如果后续迁移回 ScanNet，数据处理与 Replica 高度一致，主要差异是：

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

Replica 和 ScanNet 的常见导出 depth 都可以按 16-bit PNG 毫米深度处理；训练中统一转为米：

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

生成统一 metadata，例如 `replica_train_meta.json` / `replica_train_meta.pkl`。如果 dataloader 已经能直接读取 `traj.txt` 与 `results/`，metadata 可以作为 cache 或索引文件：

```python
{
    "dataset": "Replica",
    "split": "train",
    "scenes": ["room0"],
    "frames": {
        "room0": [
            {
                "frame_id": 0,
                "rgb_path": ".../results/frame000000.jpg",
                "depth_path": ".../results/depth000000.png",
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
- 模型不用学习 Replica/ScanNet scene 的任意 global origin。

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

### 7.2 当前验证状态与未完成增强项

当前 `README_new.md` 中的验证属于 Replica 本地数据集上的论文对齐 pilot。由于显存和数据可用性限制，验证中已经覆盖了第一阶段增强策略的一部分，但还不能等同于完整论文 4.1 augmentation：

| 增强项 | 论文 4.1 目标 | 当前验证状态 | 说明 |
|---|---|---|---|
| 图像最大边 | `518` | 使用 `224` | 小尺寸用于降低显存占用，属于合理 pilot 设置。 |
| pad 到 patch multiple | `14` 的整数倍 | 已验证 | 与 DINOv2 patch size 14 对齐。 |
| color jitter 概率 | `0.9` | 已验证 | brightness/contrast/saturation 已按目标方向实现。 |
| hue jitter | `0.1` | 需要确认或补齐 | 若代码只实现 brightness/contrast/saturation，应补上 hue。 |
| grayscale | `0.05` | 已验证或已配置 | 需要确认训练脚本中实际开启。 |
| spatial rescale | `[0.8, 1.2]` | 未完整验证 | 需要同步更新 RGB、depth、mask 和 intrinsics。 |
| aspect ratio sampling | `[0.33, 1.0]` | 未完整验证 | 属于 geometric augmentation，需和 intrinsics 更新一起实现。 |
| co-jitter | `0.3` | 未完整验证 | 需要支持同一 multi-view sample 内共享颜色扰动。 |

因此，README 中可以说“已完成论文 4.1 对齐版的核心训练策略验证”，但如果严格描述增强部分，应写成：

```text
已验证 color jitter 等 photometric augmentation；完整论文 4.1 augmentation 中的 spatial rescale、aspect-ratio sampling 和 co-jitter 仍是后续对齐项。
```

这样既保留当前 Replica 小尺寸验证的有效性，也避免把部分增强误写成完整论文增强已经全部完成。

### 7.3 推荐打开顺序

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

当前实验约束是只训练 `DepthHead + PoseHead`，因此先保留 masked log-L1，不额外加入 uncertainty branch。若未来放宽“只训练两个 heads”的限制，再考虑补论文式 uncertainty term。

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

### 9.1 Replica head-only pilot

用于在本地 Replica 数据上继续扩大第一阶段 head-only 验证：

```yaml
experiment: replica_stage1_head_pilot

data:
  dataset: Replica
  scene: room0
  num_rooms_or_sequences: 1_to_many
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
    enabled: false
    trainable: false

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

### 9.2 论文 4.1 对齐版 Replica Head-only 目标配置

这是当前阶段最终建议配置：

```yaml
experiment: replica_stage1_head_paper_aligned

data:
  dataset: Replica
  scene: room0_or_multi_room
  views_per_sample: [2, 24]
  sampler: temporal_nearby_or_spatial_nearby
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
    enabled: false
    trainable: false

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

### 9.3 论文 4.2 对齐版 Replica 长序列 Head-only 配置

这是第二阶段建议配置。它对应主论文 `4.2 Streaming Model Training`，但仍然只训练 `DepthHead + PoseHead`：

```yaml
experiment: replica_stage2_streaming_head_paper_aligned

data:
  dataset: Replica
  scene: room0
  sequence_format: video
  sampler: foldback_video
  start_frame: random
  stride: random
  reverse_at_boundary: true
  redraw_stride_after_reverse: true

curriculum:
  # 论文目标：24 -> 320
  views_start: 24
  views_end: 320
  schedule: linear
  # 显存不足 pilot：可先用 8 -> 64 或 16 -> 128

local_window:
  # 论文 GCA local pose reference window k: [16, 64]
  k_min: 16
  k_max: 64
  sample_each_iteration: true
  use_for: relative_pose_loss_only

image:
  # 论文目标仍是 518；Replica pilot 可继续使用 224/336 降低显存
  max_dim: 224_or_336_for_pilot
  target_max_dim: 518
  pad_to_multiple: 14

model:
  init_from: stage1_head_checkpoint
  backbone: dinov2_vits14
  freeze_backbone: true
  use_full_gca: false
  train_depth_head: true
  train_pose_head: true
  train_uncertainty_head: false
  multi_view_head:
    enabled: false
    trainable: false
    note: "只训练 DepthHead 与 PoseHead；长序列信息通过 sampler/curriculum/relative pose window 体现"

optimizer:
  type: AdamW
  lr: 5.0e-4
  weight_decay: 0.05

scheduler:
  type: linear_warmup_cosine_decay
  warmup_ratio: 0.05
  start_lr: 1.0e-8
  base_lr: 5.0e-4
  min_lr: 1.0e-8

loss:
  depth_weight: 1.0
  abs_pose_weight: 0.1
  rel_pose_weight: 0.05
  rel_pose_start_iter: 1000
  rel_pose_pairs: within_sampled_local_window

train:
  total_iterations: 160000
  mixed_precision: bf16_if_available_else_fp16
  gradient_clip_norm: 1.0
  gradient_accumulation: as_needed
  save_every: 5000
  log_every: 20
```

与第一阶段配置相比，第二阶段必须变化的点：

- `init_from` 改为第一阶段 head checkpoint。
- `lr` 从 `2e-4` 改为 `5e-4`。
- sampler 从 nearby/temporal nearby 改为 foldback video sampler。
- views 从固定短序列范围改为 progressive curriculum。
- relative pose loss 不再只作为辅助验证，而是长序列局部一致性的核心监督之一。
- 不引入额外 trainable window head；长序列训练通过 foldback sequence、progressive views 和 relative pose window 约束两个 heads。

资源不足时可以只减少：

- `total_iterations`
- `image.max_dim`
- `views_per_sample` 上限
- `max_images_per_gpu`
- stage2 的 `views_end`
- stage2 的 `local_window.k_max`

不建议再关闭 pose head，因为当前文档目标就是 depth head 和 pose head 同训。

## 10. 建议训练流程：Replica Stage 1 到 Stage 2

### Step 1：Replica 数据 smoke

目的：确认 Replica dataloader、RGB/depth/pose 对齐、reference normalization 没问题。

```yaml
dataset: Replica
scene: room0
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
num_rooms_or_sequences: 1_to_many
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

### Step 5：进入论文 4.2 第二阶段长序列训练

从 Step 4 的 checkpoint 初始化，只训练两个 heads，训练策略切换为 streaming/long-sequence：

```yaml
init_from: stage1_head_checkpoint
dataset: Replica
sampler: foldback_video
views_curriculum: [24, 320]
local_window_k: [16, 64]
optimizer: AdamW_lr_5e-4_wd_0.05
scheduler: 5_percent_warmup_cosine
train_depth_head: true
train_pose_head: true
freeze_backbone: true
use_full_gca: false
iterations: up_to_160000
```

显存不足时的合法降级：

- `views_curriculum: [8, 64]` 或 `[16, 128]`。
- `local_window_k: [8, 32]`。
- `image_max_dim: 224` 或 `336`。
- 使用 gradient accumulation 模拟更大的 effective batch。

验收：

- foldback sampler 能产生连续长序列，且边界反向后不会在两个帧之间来回震荡。
- view curriculum 随 iteration 增大而增加输入长度。
- local window 内的 relative pose loss finite。
- depth/pose heads 可以从第一阶段 checkpoint 正常继续训练。
- 不启用完整 GCA 时，文档和实验记录中明确说明这是 head-only approximation。

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
- Replica 2-24 views sampler 正常工作。
- Replica foldback video sampler 正常工作。
- 从 stage1 head checkpoint 初始化 stage2 head training 正常。
- stage2 view curriculum 能从短序列逐步增加到长序列。
- local window `k` 随机采样后，relative pose loss 只在窗口内计算且保持 finite。
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
- 不训练完整 GCA 主干。
- 不训练完整 streaming model。
- 不做 Ulysses context parallel / TorchTitan / Magi Attention 级别的大规模并行训练。
- 不做长序列 KV cache。
- 不追求论文最终 reconstruction 指标。

注意：第二阶段会采用 `24 -> 320` view curriculum 和 foldback video sampler，但它们只服务于 `DepthHead + PoseHead` 的长序列 head-only 训练，不代表完整复现论文中的 GCA streaming model。

## 13. 实施检查清单

### 数据

- [ ] 准备 Replica room 序列，例如 `Replica/room0`。
- [ ] 解析 `traj.txt` 为每帧 4x4 `T_c2w`。
- [ ] 匹配 `frame*.jpg` 与 `depth*.png`。
- [ ] depth PNG 除以 1000 转为 meter。
- [ ] 如果后续迁移回 ScanNet，再下载 ScanNet train split 并用官方 exporter 从 `.sens` 导出 color/depth/pose/intrinsic。
- [ ] pose 统一为 4x4 camera-to-world。
- [ ] 过滤无效 pose、缺失文件、有效深度过低的帧。
- [ ] 生成 Replica metadata/cache，或确认 dataloader 能直接读取 `traj.txt` 与 `results/`。

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
- [ ] 支持 foldback video sampler。
- [ ] 支持 stage2 `views_curriculum=[24, 320]`，显存不足时先支持 `[8, 64]` 或 `[16, 128]`。
- [ ] 支持 stage2 local window `k=[16, 64]`，显存不足时先支持 `[8, 32]`。
- [ ] 支持 view order shuffle。
- [ ] 支持 dynamic batch packing，限制每 GPU 最多 48 images。

### 模型

- [ ] 加载 DINOv2 `dinov2_vits14`。
- [ ] 冻结 DINOv2 参数。
- [ ] 启用 DepthHead。
- [ ] 启用 PoseHead。
- [ ] optimizer 只接收 trainable heads 参数。
- [ ] 确认未将 DINOv2、GCA、multi-view/window 模块参数传入 optimizer。

### Loss

- [ ] depth loss 使用 valid mask。
- [ ] depth prediction 强制为正。
- [ ] pose loss 使用 reference-normalized target。
- [ ] rotation loss 使用 geodesic。
- [ ] translation loss 使用 Huber。
- [ ] relative pose loss 延迟开启。

### 训练

- [ ] Replica room0 smoke run。
- [ ] Replica 多 room 或长序列 pilot。
- [ ] 启用 pose head 同训。
- [ ] 启用 2-24 views。
- [ ] 启用 518 max dimension。
- [ ] 补齐完整论文增强：hue jitter、spatial rescale、aspect-ratio sampling、co-jitter。
- [ ] 确认 geometric augmentation 后 RGB/depth/mask/intrinsics 全部同步更新。
- [ ] Stage1 使用 AdamW `lr=2e-4, wd=0.05`。
- [ ] Stage2 从 stage1 checkpoint 初始化，并使用 AdamW `lr=5e-4, wd=0.05`。
- [ ] 两个阶段都使用 5% warmup + cosine decay。
- [ ] Stage2 启用 foldback sampler 和 progressive view curriculum。
- [ ] 保存并验证 checkpoint。

## 14. 可写入报告的总结

本阶段在已验证的 head-only 框架基础上，先向 LingBot-Map 论文第 4.1 节 Base Model Training 对齐，再迁移第 4.2 节 Streaming Model Training 的长序列训练策略。由于算力受限，DINOv2 backbone 和主体网络保持冻结，只训练深度头与位姿头；训练数据实际使用本地 Replica RGB-D 长序列，ScanNet 流程仅作为可迁移参考。第一阶段保留 2-24 views sampling、最大边 518 的目标设置、强 photometric augmentation、AdamW `2e-4`、weight decay `0.05`、5% warmup 和 cosine decay。第二阶段从第一阶段 head checkpoint 初始化，使用 Replica foldback video sampler、progressive view curriculum、局部 relative pose window，并将学习率调整为论文 4.2 的 `5e-4`。该方案不复现完整 GCA streaming model，而是在有限资源下验证 4.2 长序列训练策略对 `DepthHead + PoseHead` 的可运行迁移。

## 15. 参考来源

- `Geometric Context Transformer for.pdf`：主论文，Sec. 3、Sec. 4.1、Sec. 4.3、Appendix A.1。
- `Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf`：VGGT 架构、DINO backbone、alternating attention、camera/depth heads。
- `π3 Permutation-equivariant visual geometry learning.pdf`：relative pose loss 和相对位姿监督思想。
- `Dinov2 Learning robust visual features without supervision.pdf`：DINOv2 作为冻结视觉特征提取器的依据。
- `README.md`：本次 head-only 训练验证记录。
- `README_new.md`：论文 4.1 对齐版 head-only 验证总结。
- `stage1_scannet_training_plan_modified.md`：上一版资源受限训练计划。
- ScanNet 官方 SensReader / exporter 文档：[SensReader](https://www.scan-net.org/ScanNet/SensReader/) 与 [Python Data Exporter](https://www.scan-net.org/ScanNet/SensReader/python/)。
