# Replica Head-only 训练方案：从论文 4.1 Base Model 到 4.2 Streaming 长序列训练

本文档基于当前文件夹中的主论文 `Geometric Context Transformer for.pdf`，以及 VGGT、π3、DINOv2 论文整理，并结合本次已跑通的训练验证记录 `README.md` 与 `stage1_scannet_training_plan_modified.md` 修改而来。

当前目标不是对 LingBot-Map 做全参数训练，而是在完整 LingBot/GCTStream 模型架构内做 head-only 验证：加载完整预训练模型，冻结 aggregator/backbone/主体注意力模块，只更新 depth head 和 camera/pose head。第一步对齐论文第一个阶段 `4.1 Base Model Training` 的监督训练设置，第二步再吸收 `4.2 Streaming Model Training` 的长序列训练策略。由于本地已下载 Replica 数据，后续验证使用 Replica 替代 ScanNet；文件名保留 `stage1_scannet_training_plan.md`，但当前实际训练数据以 Replica 为准。

- 使用 Replica RGB-D 长序列数据进行验证，ScanNet 流程保留为可迁移参考。
- 第二阶段不训练完整 GCA/streaming 主干；若使用完整 GCTStream checkpoint，可保留 frozen streaming context 前向，同时迁移长序列 curriculum、foldback sampler、relative pose window 等训练思想。
- 加载完整 LingBot/GCTStream 模型，冻结 DINOv2 backbone、frame/global blocks 等主体 aggregator。
- 训练深度头 `DepthHead` 和位姿头 `PoseHead`。
- 不训练额外 lightweight multi-view head；若代码中存在该模块，默认禁用或冻结，optimizer 只接收两个 heads 的参数。
- 训练目标从“只要跑通”升级为“先对齐论文 4.1 的 head-only 第一阶段，再用 Replica 长序列验证论文 4.2 的 streaming 训练策略，但仍只更新两个 heads”。

## 1. 已验证结果与本次修改方向

本次已有验证说明：

- 已使用 head-only 训练框架完成 smoke training。
- 早期轻量 smoke run 使用 DINOv2 `dinov2_vits14` 作为冻结特征提取器，通过 `torch.hub` 加载；正式路线已升级为完整 GCTStream head-only。
- 已验证 RGB/depth/pose dataloader、resize/pad、valid mask、pose normalization、checkpoint 保存、checkpoint 加载与深度推理可视化。
- Replica room0 测试中，100 samples、10 epochs、2 views、224 resolution 能完成训练；loss 从约 `2.91` 下降到约 `0.26`。
- 当前验证主要启用了 depth head，pose head 已在模型结构和训练配置中预留，但后续需要默认启用。
- **新增：完整 GCTStream head-only Stage1 训练已跑通**。基于 `train_replica_gct.py`，从 `lingbot-map.pt` 预训练权重加载完整 GCTStream (DINOv2 ViT-L/14 + AggregatorStream + CameraCausalHead + DPTHead)，冻结 aggregator（含 DINOv2 + frame_blocks + global_blocks，约 909M 参数，76.36%），只训练 depth_head + camera_head（约 281M 参数，23.64%）。Replica room0 上 5000 iters 完成，loss 从 `0.7356` 下降到 `0.0482`，depth loss 从 `0.4315` 降到 `0.0317`，使用 518 原生分辨率。

因此，本文档的修改重点是：

- 从”最小 smoke run”转向”Replica stage1-head 训练”，并继续扩展到 “Replica stage2 long-sequence head 训练”。
- 默认训练 `DepthHead + PoseHead`，而不是只训练 depth head。
- 视角采样从 `2 views` 扩展到论文第一阶段的 `2-24 views`。
- 分辨率从 `224/336` 逐步扩展到论文设置的最大边 `518`（**GCT 版本已支持 518 原生分辨率**）。
- 优化器从 smoke run 的 `lr=1e-3, wd=1e-4` 改为论文第一阶段的 `AdamW, lr=2e-4, wd=0.05`。
- 学习率调度改为论文第一阶段的 `5% warmup + cosine decay to 1e-8`。
- 数据增强从弱增强升级到论文第一阶段增强策略。
- **从轻量 HeadOnlyModel 升级到完整 GCTStream 模型架构**（含 AggregatorStream 跨帧因果注意力 + KV cache + CameraCausalHead 4 次迭代优化）。
- 第二阶段从已验证的第一阶段 head checkpoint 初始化，使用 Replica 长序列、foldback sampler、views curriculum 和局部 relative pose window 继续训练两个 heads。

## 2. 与论文 4.1 的对应关系

论文第一阶段 `Base Model Training` 的完整设置：

| 项目 | 论文 4.1 设置 | 当前 head-only 复现设置 |
|---|---|---|
| 数据 | 29 个公开/合成/内部数据集混合 | 当前使用 Replica，ScanNet 作为可迁移参考 |
| 模型 | DINOv2 + 24 个 alternating attention blocks + heads | 加载完整 LingBot/GCTStream，冻结 DINOv2 + aggregator 主体，只训练 depth/camera heads |
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
We follow the first-stage data sampling, augmentation, and optimization recipe of Sec. 4.1 in the full LingBot/GCTStream architecture, but freeze the pretrained aggregator and train only the depth and camera heads due to limited compute.
```

### 2.1 与 DINOv2 论文训练设置的对应关系

DINOv2 论文中的训练设置不是 LingBot 的 Stage1 head-only 训练，而是视觉 backbone 的自监督预训练和末段高分辨率适配。它可以作为“为什么冻结 DINOv2 特征仍然可用于下游 depth/geometry head”的依据，但不应该把 DINOv2 的 teacher-student 预训练超参直接套到 Replica head-only 训练。

| 项目 | DINOv2 论文设置 | 当前 Replica Stage1 head-only 设置 | 结论 |
|---|---|---|---|
| 阶段含义 | 自监督预训练 ViT 特征；另有末段 high-resolution adaptation | LingBot/GCTStream 监督训练的 Stage1 近似，只更新 heads | 二者不是同一个 stage，不能直接等同 |
| 数据 | LVD-142M 图像集合，图像检索和去重构建 | Replica RGB-D 序列，带 metric depth 和 camera trajectory | 当前数据选择合理，但不复现 DINOv2 数据管线 |
| 目标函数 | DINO image-level loss + iBOT patch-level loss + SK centering + KoLeo | depth loss + absolute/relative pose loss | 不应加入 DINO/iBOT/KoLeo，除非重新预训练 backbone |
| 可训练模块 | student backbone、DINO head、iBOT head；teacher 为 EMA 或 frozen distillation teacher | 完整 GCTStream 前向保留，aggregator 冻结，只训练 DPT depth head 和 CameraCausalHead | 当前 head-only 边界正确 |
| crop/resolution | 大 crop 224、小 crop 98；末段短时间升到 518 x 518 | LingBot Stage1 输入最大边/原生分辨率 518，pad 到 14 的倍数 | 518 可保留，但含义来自 LingBot 输入设置，不是 DINOv2 预训练 schedule |
| 优化器 | AdamW；625k iters；100k warmup；wd 0.04->0.2；LR 为 1e-3 或 3.5e-4，取决于 distilled/from-scratch | AdamW `2e-4`、wd `0.05`、5% warmup + cosine 到 `1e-8`，只作用于 heads | 当前应沿用 LingBot Stage1 超参，不套 DINOv2 预训练 LR/wd |
| 高分辨率适配 | 用预训练权重继续 10k iters，原预训练 schedule 压缩，降低 base LR | 不重新适配 DINOv2；完整 LingBot checkpoint 已作为初始化 | 不是缺项，除非目标变成重新训练/适配 backbone |
| 冻结特征下游 | 论文用 frozen features + linear/DPT decoder 做 depth 等任务 | 当前用 frozen LingBot aggregator tokens + DPTHead/CameraCausalHead | 与当前 head-only 思路一致 |

因此，当前文档需要保留 DINOv2 的内容主要有三点：

- DINOv2/ViT-L/14 作为完整 LingBot aggregator 中的冻结视觉特征来源。
- 输入尺寸需要和 patch size 14 对齐，518 分辨率有利于 pixel-level depth。
- frozen feature + DPT-style decoder 是论文支持过的下游使用方式。

不应迁移到当前 Stage1 head-only 配置的 DINOv2 内容包括：DINO/iBOT/KoLeo loss、student-teacher EMA、DINO projection heads、625k/100k warmup、LVD-142M 数据管线、FSDP 预训练配置、distillation teacher 训练流程。

### 2.2 与论文 4.2 Streaming Model Training 的对应关系

论文第二阶段的目的，是把第一阶段得到的 base model 迁移到 streaming/long-sequence setting。主论文 4.2 的核心训练策略如下：

- **Initialization from Base Model**：从第一阶段 base model checkpoint 初始化 streaming model。
- **Attention replacement**：将第一阶段 global attention 替换为 GCA；由于 Q/K/V projection 参数化一致，权重可以直接迁移。
- **Optimization**：训练 `160K iterations`，base learning rate 改为 `5e-4`，仍使用第一阶段相同的 5% warmup + cosine decay 到 `1e-8`。
- **Progressive view curriculum**：训练 views 数从 `24` 线性增加到 `320`。
- **Local pose reference window**：GCA 的局部参考窗口 `k` 在 `[16, 64]` 中随机采样。
- **Context parallelism**：论文使用 Ulysses context parallel，parallel dimension 为 `16`，以支持长序列训练。
- **Long-trajectory data**：第二阶段转向长轨迹视频数据，并用 foldback video sampler 生成时序连续但无固定前进偏置的训练片段。
- **GCA context structure**：完整 GCA 训练同时依赖 anchor context、local pose-reference window 和 trajectory memory；trajectory memory 对被移出 local window 的历史帧保留 compact context tokens，并加入 temporal positional encoding。
- **Relative pose loss**：在 local pose-reference window 内对所有 frame pairs 计算 relative pose loss，而不是只对相邻帧计算。

当前 head-only 版本需要区分“使用完整 GCTStream/GCA 前向”和“训练完整 GCA 主干”。如果 `lingbot-map.pt` 已经是 GCTStream/streaming checkpoint，可以保留其 frozen aggregator/GCA 前向作为特征生成器；但 optimizer 仍只更新 `depth_head + camera_head`。如果使用的是轻量 HeadOnlyModel，则不能声称复现了 GCA、anchor context 或 trajectory memory，只能说迁移了长序列采样和局部 relative pose 监督。

| 项目 | 论文 4.2 设置 | Replica head-only 第二阶段设置 |
|---|---|---|
| 初始化 | 从第一阶段 base checkpoint 初始化；将 global attention 替换为 GCA，Q/K/V 权重可直接迁移 | 加载完整 `lingbot-map.pt` / GCTStream，再叠加 stage1 head checkpoint；冻结 aggregator，只训练 heads |
| GCA | 替换 global attention 为 GCA 并训练 streaming model | 若 checkpoint/模型已含 GCTStream/GCA，则 frozen 使用其前向；不训练 GCA 参数，也不新增 window head |
| Anchor context | 首批 anchor frames 建立坐标与尺度，并按 anchor point cloud scale 归一化 depth/translation | head-only 中不训练 anchor 机制；若 GCTStream 前向支持则保留 frozen anchor/context 行为，loss target 仍需 reference/scale normalization |
| Trajectory memory | 对 local window 外的历史帧保留 camera/anchor/register 等 compact context tokens，并加入 temporal positional encoding | 不训练 trajectory memory；若 frozen GCTStream 支持则使用并清理 KV cache，否则不能声称具备论文的 drift-correction memory |
| 训练模块 | 全 streaming model | 仍只训练 `DepthHead + PoseHead` |
| 数据 | 提高长轨迹、多场景视频数据权重，降低/丢弃非时序 multi-view 数据 | 使用 Replica room 序列作为 pilot；多 room/长轨迹不足会削弱 Stage2 的跨区域训练信号 |
| Views curriculum | `24 -> 320` 线性增加 | 目标保持 `24 -> 320`，显存不足时先用 `8 -> 64` 或 `16 -> 128` pilot |
| Local window k | `[16, 64]` 随机采样，用于 GCA receptive field 和 relative pose loss | head-only 中至少用于 relative pose loss；若使用 frozen GCTStream，还应同步设置/记录 GCA sliding window |
| Relative pose loss | local window 内所有 ordered frame pairs | 当前应使用 window 内 all-pairs；若为显存/速度降级为 adjacent pairs，必须标注为近似 |
| Optimizer | AdamW, `lr=5e-4` | AdamW, `lr=5e-4`，仅作用于两个 heads |
| Schedule | 5% warmup + cosine | 保持一致 |
| Context parallel | Ulysses, dimension 16 | 不强制；优先用更短 views/window、gradient accumulation 和截断长序列控制显存 |
| 最大序列长度 | 训练上限 320 views | 完整 GCTStream route 需要 `max_frame_num >= views_end`；当前 `train_replica_gct.py` 默认 `max_frame_num=100`，若跑 320 views 必须改 |

这份第二阶段 head-only 方案可以写成：

```text
We initialize the full frozen LingBot/GCTStream model from the pretrained checkpoint, load the trainable heads from the stage-1 head checkpoint, and follow the stage-2 long-sequence curriculum of Sec. 4.2 on Replica videos, while updating only the depth and camera heads and omitting full GCA/context-parallel training.
```

## 3. 模型设置

### 3.1 Head-only 总体结构

当前提供两种 head-only 实现版本：

**版本 A：轻量 HeadOnlyModel（`head_only_model.py` + `train_replica_v2.py`）**

```text
RGB images: [B, V, 3, H, W]
  -> frozen DINOv2 ViT-S/14 or ViT-B/14
  -> patch tokens + cls token（每帧独立处理，无跨帧注意力）
  -> DepthHead (DPTHead, multi-scale via hooks)
  -> PoseHead (CameraHead, 4-iter refinement)
```

**版本 B：完整 GCTStream Head-Only（`train_replica_gct.py`）**

```text
RGB images: [B, V, 3, H, W] (518 native resolution)
  -> frozen GCTStream.aggregator
       ├── DINOv2 ViT-L/14 patch embed
       ├── frame_blocks (intra-frame self-attention)
       └── global_blocks (cross-frame causal attention + KV cache)
  -> aggregated_tokens_list (4 scales, [B,V,P,2*1024])
  -> trainable depth_head (DPTHead, 2*embed_dim=2048)
  -> trainable camera_head (CameraCausalHead, 4-iter refinement, causal attn + KV cache)
```

其中：

- `B`：batch size。
- `V`：每个样本的 view 数，训练时从 `[2, 24]` 随机采样。
- `H, W`：resize/pad 后的训练分辨率，目标最大边为 `518`，并 pad 到 14 的整数倍。
- 版本 B 使用完整 GCTStream 预训练权重，aggregator 用 `torch.no_grad()` 前向以节省显存，仅 heads 追踪梯度。
- 版本 B 在 RTX 3090 (24GB) 上可跑通 518 原生分辨率训练。

### 3.2 冻结与训练参数

| 模块 | 是否训练 | 说明 |
|---|---:|---|
| DINOv2 backbone | 否 | 使用预训练视觉特征，减少显存和训练成本。 |
| 原始 24 层 alternating attention（版本 A） | 否或不加载 | HeadOnlyModel 不训练完整 base model。 |
| **完整 Aggregator (frame_blocks + global_blocks)（版本 B）** | **否（冻结）** | GCTStream 加载预训练权重后冻结，含跨帧因果注意力，约 909M 参数。 |
| GCA / trajectory memory / KV cache | 否（参数不训练；可在前向使用） | 版本 B 可在 forward 中使用 frozen streaming context/KV cache，但参数不训练；Stage2 不引入新的可训练 GCA 主干。 |
| DepthHead | 是 | 必训，预测每帧 metric depth。版本 B 使用 GCTStream 原生 DPTHead (2048 dim_in)。 |
| PoseHead | 是 | 必训，预测每帧相对第一帧的 camera-to-world pose。版本 B 使用 CameraCausalHead (4 次迭代优化)。 |
| Lightweight MultiViewHead | 否 | 为满足”只训练两个 heads”，默认禁用或冻结，不传入 optimizer。 |
| Uncertainty branch | 否 | 当前只训练 depth/pose 两个 heads，先使用 masked log-L1，不单独训练 uncertainty branch。 |

当前阶段的核心变化是：`PoseHead` 不再是可选项，而是与 `DepthHead` 一起训练。版本 B 进一步把跨帧注意力（AggregatorStream）从”未实现”升级为”冻结后参与前向”。

### 3.3 推荐 head 结构

```yaml
model:
  # 当前推荐：完整 LingBot/GCTStream head-only
  gct_stream_head_only:
    enabled: true
    init_from: lingbot-map.pt
    freeze_aggregator: true       # DINOv2 + frame_blocks + global_blocks 全部冻结
    aggregator_forward: no_grad   # 节省显存
    train_depth_head: true        # GCTStream 原生 DPTHead
    train_camera_head: true       # CameraCausalHead, 4-iter refinement
    image_size: 518               # 原生分辨率，RTX 3090 可跑通

  # 仅作为早期 sanity check，不作为当前正式 Stage1 目标
  legacy_lightweight_head_only:
    enabled: false
    backbone: dinov2_vits14
    load_method: torch_hub

  multi_view_head:
    enabled: false
    trainable: false
    note: “当前实验只训练 DepthHead 与 PoseHead，不训练额外 multi-view head”

  depth_head:
    enabled: true
    type: dpt_or_lightweight_decoder
    trainable: true
    output_activation: exp_or_softplus
    output: depth_meters

  pose_head:
    enabled: true
    type: CameraCausalHead
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

### 9.1 Replica GCTStream head-only pilot

用于在本地 Replica 数据上继续扩大第一阶段 head-only 验证。正式路线使用完整 GCTStream；`dinov2_vits14` 轻量模型只保留为早期 sanity check。

```yaml
experiment: replica_stage1_head_pilot

data:
  dataset: Replica
  scene: room0
  num_rooms_or_sequences: 1_to_many
  views_per_sample: [2, 4]
  sampler: temporal_nearby
  temporal_window: 30
  image_max_dim: 518
  pad_to_multiple: 14

model:
  base: GCTStream
  init_from: /home/shared_files/model_weights/linbo_map/lingbot-map.pt
  freeze_aggregator: true
  aggregator_forward: no_grad
  train_depth_head: true
  train_camera_head: true
  legacy_lightweight_head_only: false

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
  base: GCTStream
  init_from: /home/shared_files/model_weights/linbo_map/lingbot-map.pt
  patch_size: 14
  embed_dim: 1024
  backbone: dinov2_vitl14_inside_gct
  freeze_aggregator: true
  aggregator_forward: no_grad
  use_gca: false
  train_depth_head: true
  train_camera_head: true
  train_uncertainty_head: false
  legacy_lightweight_head_only: false

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

### 9.2.1 已验证版本：完整 GCTStream Head-Only Stage1 (`train_replica_gct.py`)

本节描述当前已跑通的 GCTStream head-only Stage1 训练配置，对应代码文件 `try_train/train_replica_gct.py` 和快速运行脚本 `quick-op/train_gct_stage1.sh`。该配置已在 RTX 3090 (24GB) 上完成 5000 iterations 训练验证。

```yaml
experiment: replica_stage1_gct_head_only_5k

data:
  dataset: Replica
  scene: room0
  data_root: /home/shared_files/datasets/dovsg/Replica/room0
  views_per_sample: [2, 2]          # 当前固定 2 views，可扩展到更大
  sampler: temporal_nearby
  shuffle_view_order: true

image:
  max_dim: 518                       # 原生分辨率（GCTStream 设计输入）
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
  co_jitter: true

model:
  base: GCTStream                    # 完整 lingbot-map 架构
  init_from: /home/shared_files/model_weights/linbo_map/lingbot-map.pt
  patch_size: 14
  embed_dim: 1024                    # ViT-L/14
  enable_3d_rope: true
  kv_cache_sliding_window: 64
  kv_cache_scale_frames: 8
  use_sdpa: true                     # 或 use_flashinfer

  frozen_modules:                    # 909M 参数 (76.36%)
    - aggregator (DINOv2 + frame_blocks + global_blocks)
    - aggregator forward 使用 torch.no_grad()
  trainable_modules:                 # 281M 参数 (23.64%)
    - depth_head (DPTHead, dim_in=2048)
    - camera_head (CameraCausalHead, 4-iter refinement)

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
  depth:
    type: masked_log_l1
    weight: 1.0
  abs_pose:
    weight: 0.1
  rel_pose:
    weight: 0.05
    start_iter: 1000

train:
  total_iterations: 5000             # 当前 pilot 验证，目标 160000
  mixed_precision: bf16              # RTX 3090 (compute_cap ≥ 8.0)
  gradient_clip_norm: 1.0
  save_every: 1000
  log_every: 50
  use_amp: true
```

实测训练曲线（5000 iters）：

| 指标 | 起始 (iter=50) | 结束 (iter=5000) |
|---|---:|---:|
| total loss | 0.7356 | 0.0482 |
| depth loss (masked_log_l1) | 0.4315 | 0.0317 |

显存峰值约 11 GB（518 分辨率，2 views，bf16 + no_grad aggregator）。

关键实现要点（`train_replica_gct.py`）：

- **`forward_gct_heads_only(model, images)`**：自定义前向函数，aggregator 用 `torch.no_grad()` + `model.aggregator(...)`，再把 `aggregated_tokens_list` `detach()` 后送入 heads。
- **每次迭代清理 KV cache**：`model.clean_kv_cache()` 在前向前后调用，防止显存累积。
- **深度分辨率对齐**：`align_depth_to_gt()` 用 bilinear 插值把模型 518 输出对齐到 GT 尺寸再计算 loss。
- **Checkpoint 同时保存 `head_state_dict`（仅 heads，节省存储）和 `full_model_state_dict`（完整模型，方便 resume 和验证）**。

**Stage1 = global attention 明确化（论文 4.1）：**

| 参数 | Stage1 取值 | 含义 |
|---|---|---|
| `sliding_window_size` | `-1` | aggregator 不使用 GCA 局部窗口，做全局注意力 |
| `num_frame_per_block` | `S` | 一次性处理所有 views，不分块逐帧 |
| `num_frame_for_scale` | `S` | 所有帧 bidirectional 参与 scale 估计 |
| `causal_inference` (camera_head) | `False` | camera_head 非 streaming，不用 KV cache |

这等价于把整个 sample 视为一个 multi-view set，对应主论文 4.1 base model 训练阶段的 global attention 设置。GCA streaming 是 Stage2 才打开的。

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

target_normalization:
  pose_reference: first_or_anchor_frame
  anchor_scale_normalization: preferred_for_stage2_if_available
  note: "论文 Stage2/GCA 用 anchor frames 建立尺度，并按 anchor point-cloud scale 归一化 depth 与 translation；若暂不实现，需标注为 head-only approximation"

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
  use_for: gca_sliding_window_if_available_and_relative_pose_loss
  rel_pose_pairs: all_ordered_pairs_within_window

image:
  # 完整 GCTStream 当前按 518 原生分辨率验证；显存不足时优先降 views/window
  max_dim: 518
  pad_to_multiple: 14

model:
  base: GCTStream
  init_base_from: /home/shared_files/model_weights/linbo_map/lingbot-map.pt
  init_heads_from: stage1_head_checkpoint
  load_order:
    - load_full_base_checkpoint
    - override_depth_head_and_camera_head_from_stage1_checkpoint
  backbone: dinov2_vitl14_inside_gct
  max_frame_num: 320                 # 必须 >= views_end；当前 train_replica_gct.py 默认 100，跑 320 前要改
  freeze_aggregator: true
  aggregator_forward: no_grad
  use_gct_stream_forward: true        # 若 checkpoint/代码支持 GCA/streaming context，则前向使用但不训练
  train_gca: false
  train_trajectory_memory: false
  train_depth_head: true
  train_camera_head: true
  train_uncertainty_head: false
  multi_view_head:
    enabled: false
    trainable: false
    note: "只训练 DepthHead 与 PoseHead；长序列信息通过 sampler/curriculum/relative pose window 体现"

streaming_context:
  anchor_frames: use_model_default    # 论文用 anchor context 建立尺度/坐标；head-only 不训练该机制
  trajectory_memory: frozen_if_available
  temporal_positional_encoding: frozen_if_available
  kv_cache:
    sliding_window: 64
    clear_each_iteration: true
    note: "训练时避免跨 batch 泄漏状态；长序列验证时可按 streaming state 评估"

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
  rel_pose_pairs: all_ordered_pairs_within_sampled_local_window
  rel_pose_pair_subsample_if_oom: true

train:
  total_iterations: 160000
  mixed_precision: bf16_if_available_else_fp16
  gradient_clip_norm: 1.0
  gradient_accumulation: as_needed
  save_every: 5000
  log_every: 20
```

与第一阶段配置相比，第二阶段必须变化的点：

- 初始化流程改为：先加载完整 `lingbot-map.pt` / GCTStream base，再从第一阶段 checkpoint 覆盖 `depth_head + camera_head`。
- `lr` 从 `2e-4` 改为 `5e-4`。
- sampler 从 nearby/temporal nearby 改为 foldback video sampler。
- views 从固定短序列范围改为 progressive curriculum。
- relative pose loss 不再只作为辅助验证，而是长序列局部一致性的核心监督之一；论文定义是 local window 内 all-pairs。
- 若使用完整 GCTStream，需要同步设置 `max_frame_num >= views_end`，并确认 GCA sliding window / KV cache 不跨 batch 污染。
- Stage2 target normalization 应尽量加入 anchor-scale normalization；若只做 first-frame pose normalization，需要明确这是对论文 GCA anchor normalization 的简化。
- 不引入额外 trainable window head；长序列训练通过 foldback sequence、progressive views、frozen streaming context 和 relative pose window 约束两个 heads。

当前实现状态需要特别注意：

- `try_train/train_replica_stage2.py` 仍是旧的轻量 `HeadOnlyModel` 路线，默认 `dinov2_vits14`、`img_size=224`、`views_start=8`、`views_end=24`，不能代表“完整 LingBot/GCTStream head-only Stage2”。
- 正式 Stage2 应基于 `train_replica_gct.py` 的完整 GCTStream 加载/冻结逻辑，移植 `foldback_video_sampler.py` 中的 foldback、view curriculum 和 local window sampler。
- `train_replica_gct.py` 当前 `max_frame_num=100`，如果按论文目标 `views_end=320` 训练，必须把模型初始化和位置编码容量改到至少 320，或把 pilot 的 `views_end` 限制在 100 以内并在实验记录中说明。

资源不足时可以只减少：

- `total_iterations`
- `views_per_sample` 上限
- `max_images_per_gpu`
- stage2 的 `views_end`
- stage2 的 `local_window.k_max`
- relative-pose pair 数量（从 all-pairs 改为 pair subsampling），但需要标注为近似

完整 GCTStream 路线下不建议优先降 `image.max_dim`，因为当前模型按 518 原生分辨率验证；如需降到 224/336，必须先确认位置编码、DPT head 输出尺度和 checkpoint 兼容。

不建议再关闭 pose head，因为当前文档目标就是 depth head 和 pose head 同训。

### 9.3.1 已验证版本：完整 GCTStream Head-Only Stage2 (`train_replica_gct_stage2.py`)

本节描述当前已实现并通过 smoke test 的 GCTStream head-only Stage2 训练配置，对应代码文件 `try_train/train_replica_gct_stage2.py` 和快速运行脚本 `quick-op/train_gct_stage2.sh`。该实现已在 RTX 3090 (24GB) 上完成 10 iters smoke 验证，loss 从 0.12 下降到 0.03，正式 5K iters 训练待执行。

```yaml
experiment: replica_stage2_gct_head_only_5k

data:
  dataset: Replica
  scene: room0
  data_root: /home/shared_files/datasets/dovsg/Replica/room0
  sampler: foldback_video                # 论文 4.2
  stride_range: [1, 3]
  redraw_stride_after_reverse: true

curriculum:
  # 显存约束：518 原生分辨率 + 完整 GCTStream，pilot 用 4 -> 8
  # 论文设置：24 -> 320（需要更大 GPU 显存）
  views_start: 4
  views_end: 8
  warmup_iterations: 1000

local_window:
  # 论文 GCA local pose reference window k: [16, 64]
  # 显存/views 数缩放后 pilot: [2, 4]
  k_min: 2
  k_max: 4
  # 同时驱动 aggregator GCA 窗口 + rel-pose loss 局部对
  use_for: [aggregator_sliding_window, camera_head_sliding_window, relative_pose_loss]

image:
  max_dim: 518
  pad_to_multiple: 14

model:
  base: GCTStream                        # 完整 lingbot-map 架构
  init_from: stage1_gct_checkpoint       # 从 Stage1 GCT ckpt full_model_state_dict
  patch_size: 14
  embed_dim: 1024                        # ViT-L/14
  enable_3d_rope: true
  max_frame_num: 400                     # Stage2 长序列需要扩大
  kv_cache_sliding_window: 64
  kv_cache_scale_frames: 8
  use_sdpa: true

  # 与 Stage1 相同：仅训练 heads，aggregator 冻结
  frozen_modules:                        # 909M 参数 (76.36%)
    - aggregator (DINOv2 + frame_blocks + global_blocks)
    - aggregator forward 使用 torch.no_grad()
  trainable_modules:                     # 281M 参数 (23.64%)
    - depth_head (DPTHead, dim_in=2048)
    - camera_head (CameraCausalHead, 4-iter refinement, KV cache 启用)

# Stage2 attention 模式（与 Stage1 区别的核心）
attention_mode:
  sliding_window_size: k                 # 来自 LocalWindowSampler，每 iter 重新采样
  num_frame_per_block: 1                 # Stage2 逐帧 streaming
  num_frame_for_scale: 8                 # 仅前 8 帧 bidirectional 做 scale 估计
  causal_inference: true                 # camera_head 用 causal + KV cache

optimizer:
  type: AdamW
  lr: 5.0e-4                              # 论文 4.2 (vs Stage1 lr=2e-4)
  weight_decay: 0.05

scheduler:
  type: linear_warmup_cosine_decay
  warmup_ratio: 0.05
  start_lr: 1.0e-8
  base_lr: 5.0e-4
  min_lr: 1.0e-8

loss:
  depth:
    type: masked_log_l1
    weight: 1.0
  abs_pose:
    weight: 0.1
  rel_pose:
    type: local_window_relative_pose      # 只在 k 内 pair
    weight: 0.05
    start_iter: 500
    window_pairs: from_LocalWindowSampler

train:
  total_iterations: 5000                  # pilot 验证，目标 160000
  mixed_precision: bf16
  gradient_clip_norm: 1.0
  save_every: 1000
  log_every: 50
  use_amp: true
```

**Stage2 = GCA streaming 明确化（论文 4.2，与 Stage1 的核心差异）：**

| 维度 | Stage1 (`forward_gct_heads_only`) | Stage2 (`forward_gct_heads_only_streaming`) |
|---|---|---|
| **attention 模式** | global attention（全局双向） | GCA（Geometric Context Attention）局部因果 |
| `sliding_window_size` | `-1`（无窗口） | `k`（每 iter 从 `LocalWindowSampler.sample_window_size()` 采样） |
| `num_frame_per_block` | `S`（一次处理全部 views） | `1`（逐帧 causal streaming） |
| `num_frame_for_scale` | `S`（所有帧 bidirectional） | `8`（仅前 8 帧 bidirectional，对应论文 scale frames） |
| `causal_inference` (camera_head) | `False` | `True`（带 KV cache） |
| 采样 | nearby (temporal/spatial)，短 multi-view sets | foldback video，长有序序列 |
| LR | `2e-4` | `5e-4` |
| Views 数 | 2-24 随机 | progressive curriculum（4→8 pilot / 24→320 论文） |
| Rel-pose loss | 所有 view pair | 局部窗口 `k` 内 pair（与 GCA 窗口同源） |
| 初始化 | 从 `lingbot-map.pt` 加载 | 从 Stage1 GCT ckpt 加载 `full_model_state_dict` |

关键实现要点（`train_replica_gct_stage2.py`）：

- **`forward_gct_heads_only_streaming(model, images, sliding_window_size, num_frame_for_scale=8)`**：新增 Stage2 专用前向。aggregator 走 GCA 局部窗口，camera_head 走 causal + KV cache，模拟论文 4.2 的 streaming inference 训练设置。
- **`k` 同时驱动 GCA 窗口和 rel-pose loss**：每个 iter `LocalWindowSampler.sample_window_size()` 给出 `k`，既作为 `sliding_window_size` 传给 aggregator/camera_head，也用于 `LocalWindowSampler.get_adjacent_pairs(V, k)` 构造 rel-pose pair。两者使用同一窗口尺度，与论文 4.2 描述一致。
- **复用 Stage1 的 `align_depth_to_gt` 与 `freeze_aggregator`**：Stage2 仍然只训练 heads，aggregator 全部 911M 参数冻结。
- **Checkpoint 与 Stage1 同格式**：同时保存 `head_state_dict` 与 `full_model_state_dict`，因此 `validate_gct_stage1.py` 可直接复用验证 Stage2 checkpoint，无需额外脚本。
- **Anchor-scale normalization（论文 4.2，已实现）**：新增 `compute_anchor_scale(depths, valid_masks, poses, num_anchor_frames, source)` 和 `apply_anchor_scale_normalization(depths, poses, s)` 两个工具。每个 iter：(1) 从前 N=`num_frame_for_scale` 帧 anchor frames 算尺度 `s`；(2) 把 GT `depth /= s`、`pose[:3,3] /= s`（rotation 不缩放）；(3) 再算 loss。
  - `s` 来源可选：`depth_median`（论文推荐，anchor 点云尺度）或 `translation_norm`（备选）。
  - CLI 开关：`--use_anchor_scale_norm` / `--no_anchor_scale_norm`、`--anchor_scale_source`。
  - 每条 log 输出 `anchor_s` 数值，便于监控尺度统计。

**Replica depth scale 关键 bug 修复（影响 Stage1/Stage2）：**

- **问题**：早期代码 (`replica_dataset.py` 等 4 个文件) 错用 `depth_png / 1000.0`（标准毫米），但 Replica 实际不是毫米单位。
- **正确做法**：使用 `depth_png / 6553.5`，与 NICE-SLAM / iMAP / GO-SLAM / MonoGS 等 Replica 学术工作一致（uint16 max=65535 映射 ~10m）。
- **修复影响**：room0 depth 中位数从错误的 ~17.6m 变成正确的 ~2.7m；anchor_s（depth_median 模式）从 ~18 变成 ~2.7。
- **统一方案**：`replica_dataset.py` 顶部定义 `REPLICA_DEPTH_SCALE = 6553.5` 常量，4 个文件统一 import 使用。
- **历史 checkpoint**：在错误 scale 下训练的 `gct_stage1_5k/checkpoint_final.pt` 等，模型学到的输出 ≈ true_depth × 6.5535（结构正确，绝对尺度错一个常数）。建议修复后重训以获得真正的 metric depth 输出。

实测 smoke（10 iters，views 4→6, k=[2,3]）：

| iter | views | k | total loss | depth | abs_pose | rel_pose |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 2 | 0.1228 | 0.1214 | 0.0146 | 0.0000 |
| 10 | 6 | 3 | 0.0313 | 0.0274 | 0.0361 | 0.0061 |

显存峰值与 Stage1 接近（~11 GB，518 + 2-6 views）；正式 5K iters 训练待执行。

## 10. 建议训练流程：Replica Stage 1 到 Stage 2

### Step 1：Replica 数据 smoke

目的：确认 Replica dataloader、RGB/depth/pose 对齐、reference normalization 没问题。

```yaml
dataset: Replica
scene: room0
base: GCTStream
init_from: lingbot-map.pt
views_per_sample: [2, 2]
image_max_dim: 518
train_depth_head: true
train_camera_head: true
freeze_aggregator: true
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
base: GCTStream
num_rooms_or_sequences: 1_to_many
views_per_sample: [2, 4]
image_max_dim: 518
train_camera_head: true
freeze_aggregator: true
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
base: GCTStream
init_base_from: lingbot-map.pt
init_heads_from: stage1_head_checkpoint
max_frame_num: 320
dataset: Replica
sampler: foldback_video
views_curriculum: [24, 320]
local_window_k: [16, 64]
rel_pose_pairs: all_ordered_pairs_within_local_window
optimizer: AdamW_lr_5e-4_wd_0.05
scheduler: 5_percent_warmup_cosine
train_depth_head: true
train_camera_head: true
freeze_aggregator: true
use_gct_stream_forward: true
train_gca: false
iterations: up_to_160000
```

显存不足时的合法降级：

- `views_curriculum: [8, 64]` 或 `[16, 128]`。
- `local_window_k: [8, 32]`。
- `max_frame_num` 可随 pilot 的 `views_end` 降低，但必须始终 `>= views_end`。
- relative pose all-pairs 可 subsample，但要记录为近似实现。
- 优先减少每次前向的 views 数；只有在确认完整 GCTStream 输入/位置编码支持时，再考虑 `image_max_dim: 224` 或 `336`。
- 使用 gradient accumulation 模拟更大的 effective batch。

验收：

- foldback sampler 能产生连续长序列，且边界反向后不会在两个帧之间来回震荡。
- view curriculum 随 iteration 增大而增加输入长度。
- local window 内 all-pairs relative pose loss finite；若使用 pair subsampling，日志要记录采样比例。
- `max_frame_num`、GCA/KV sliding window 和当前 views 数匹配，不因超过位置编码容量而 silently truncate。
- target normalization 记录清楚：仅 first-frame pose normalization，还是包含论文式 anchor scale normalization。
- depth/pose heads 可以从第一阶段 checkpoint 正常继续训练。
- 不训练完整 GCA 时，文档和实验记录中明确说明这是 frozen-GCTStream/head-only approximation。

## 11. 验证与记录

### 11.1 已验证内容

来自本次 README 记录：

- DINOv2 torch.hub 加载可用。
- frozen backbone + depth head 可训练。
- AMP、gradient clipping、checkpoint 保存可用。
- dataloader 能输出 RGB、depth、mask、pose。
- depth 预测可视化流程可用。

**新增 GCTStream Stage1 已验证内容**（`train_replica_gct.py` + `validate_gct_stage1.py`）：

- 从 `lingbot-map.pt` 加载完整 GCTStream 预训练权重，含 DINOv2 ViT-L/14、AggregatorStream、CameraCausalHead、DPTHead。
- `freeze_aggregator()` 正确冻结 909M 参数（aggregator 含 DINOv2 + frame_blocks + global_blocks），只暴露 281M 可训练参数（depth_head + camera_head）。
- `forward_gct_heads_only()` 中 aggregator 用 `torch.no_grad()` 前向、heads 用梯度追踪的混合 forward 在 bf16 AMP 下稳定收敛。
- **Stage1 attention 模式确认为 global attention**（`sliding_window_size=-1`、`num_frame_per_block=S`、`num_frame_for_scale=S`、`causal_inference=False`），与论文 4.1 base model 训练设置一致。
- KV cache 在每次迭代前后正确清理，不出现显存累积。
- 5000 iters 训练 loss 从 0.7356 单调下降到 0.0482，depth loss 从 0.4315 降到 0.0317。
- 518 原生分辨率在 RTX 3090 (24GB) 上可稳定训练，峰值显存 ~11 GB。
- Checkpoint 同时保存 `head_state_dict` 与 `full_model_state_dict`，可被 `validate_gct_stage1.py` 加载用于推理对比。
- 验证脚本支持训练前后单样本详细对比（深度图、误差热力图、AbsRel/RMSE/Log-L1 指标、pose encoding 数值差异）。

**新增 GCTStream Stage2 已验证内容**（`train_replica_gct_stage2.py` + `quick-op/train_gct_stage2.sh` + `quick-op/validate_gct_stage2.sh`）：

- 新增 `forward_gct_heads_only_streaming(model, images, sliding_window_size, num_frame_for_scale=8)`，与 Stage1 的 global attention forward 形成对照：**Stage2 用 GCA streaming**（`sliding_window_size=k`、`num_frame_per_block=1`、`num_frame_for_scale=8`、`causal_inference=True`），与论文 4.2 streaming model 设置一致。
- 从 Stage1 GCT checkpoint 的 `full_model_state_dict` 正确加载初始权重，aggregator 仍保持冻结（与 Stage1 同样的 909M / 281M 划分）。
- `LocalWindowSampler.sample_window_size()` 给出的 `k` 同时作为 aggregator/camera_head 的 GCA 窗口和 rel-pose loss 的局部 pair 来源，与论文 4.2 描述一致。
- `FoldbackVideoSampler`、`ProgressiveViewCurriculum`、`LocalRelativePoseLoss`、`ReplicaLongSequenceDataset` 全部复用 HeadOnly Stage2 已有组件，无需重写。
- Smoke 10 iters（views 4→6, k=[2,3]）下 loss 从 0.1228 单调下降到 0.0313，bf16 AMP 稳定。
- Checkpoint 与 Stage1 同格式（含 `full_model_state_dict`），因此 `validate_gct_stage1.py` 可直接复用验证 Stage2 checkpoint。
- 显存峰值与 Stage1 接近（~11 GB），证明 GCA streaming 没有引入额外不可控开销。

**新增 anchor-scale normalization 已验证内容（论文 4.2，`train_replica_gct_stage2.py`）：**

- `compute_anchor_scale` 支持两种 source：
  - `depth_median`：取前 N 帧 GT depth 的 median，等价于"anchor 点云尺度"（论文推荐）
  - `translation_norm`：取前 N 帧相机平移的均值
- `apply_anchor_scale_normalization` 把 `depth /= s` 与 `pose[:3,3] /= s`，rotation 保持不变（scale-invariant）。
- 修复 Replica depth scale bug 之后实测 smoke（views=4, N=4, k=2, anchor 模式 `depth_median`）：
  - anchor_s ≈ 2.6–3.1m（合理的 Replica 室内深度中位数）
  - 修复前 anchor_s ≈ 18（明显错误，因为 depth scale 用错为 /1000）
- `--no_anchor_scale_norm` 时 smoke loss=0.029，与之前不带 anchor 的基线一致，表明开关切换不破坏原有路径。

**新增 Replica depth scale 关键 bug 修复已验证内容：**

- `replica_dataset.py` 顶部新增模块级常量 `REPLICA_DEPTH_SCALE = 6553.5`，`_load_depth` 与其它 3 个文件（`train_replica_stage2.py`、`validate_with_original.py`、`validate_long_sequence.py`）统一 import 使用，避免 magic number。
- 验证：直接读 Replica room0 frame0 depth 后，view 0 median 从 17.63m → 2.91m，max 从 33.16m → 5.13m，符合典型室内房间尺寸。
- 未改动 `scannet_dataset.py` 的 `/1000`，因为 ScanNet 真的是毫米单位。

### 11.2 下一步需要验证

为满足当前文档目标，需要新增验证：

- `train_camera_head=true` 时，pose/camera head 正常参与训练。
- `abs_pose_loss` 正常下降或至少保持 finite。
- `relative pose loss` 打开后不导致 NaN。
- Replica 2-24 views sampler 正常工作。
- Replica foldback video sampler 正常工作。
- 从 stage1 head checkpoint 初始化 stage2 head training 正常。
- stage2 view curriculum 能从短序列逐步增加到长序列。
- local window `k` 随机采样后，relative pose loss 只在窗口内计算 all-pairs 或明确记录 pair subsampling，并保持 finite。
- 完整 GCTStream Stage2 路线中 `max_frame_num >= views_end`，并确认 320 views 不超过位置编码/KV cache 设计容量。
- Stage2 是否实现 anchor-scale normalization；若未实现，验证记录中注明当前只做 reference pose normalization。
- `train_replica_stage2.py` 是否升级为完整 GCTStream 路线；若继续使用旧轻量 HeadOnlyModel，实验记录必须标注为 legacy pilot。
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
- 不对完整 LingBot/GCTStream 做全参数训练。
- 不训练 DINOv2 backbone。
- 不训练 aggregator 中的 frame/global attention blocks。
- 不训练完整 GCA 主干。
- 不把 frozen GCTStream/GCA 前向等同于论文 4.2 的全参数 streaming model training。
- 不训练完整 streaming model。
- 不做 Ulysses context parallel / TorchTitan / Magi Attention 级别的大规模并行训练。
- 不新增或训练长序列 KV cache/trajectory memory 参数；若完整 GCTStream 前向使用 KV cache，需要按实现清理 cache。
- 不追求论文最终 reconstruction 指标。

注意：第二阶段会采用 `24 -> 320` view curriculum 和 foldback video sampler；若使用完整 GCTStream checkpoint，可以保留 frozen GCA/anchor/trajectory-memory 前向，但这些只服务于 `DepthHead + PoseHead` 的长序列 head-only 训练，不代表完整复现论文中的 GCA streaming model 全参数训练。

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
- [ ] Stage2 若使用 frozen GCTStream/GCA anchor context，评估是否加入 anchor-scale normalization；未加入时在实验记录标注。

### 采样

- [ ] 支持 `views_per_sample=[2, 24]`。
- [ ] 支持 temporal nearby fallback。
- [ ] 支持 spatial nearby sampler。
- [ ] 支持 foldback video sampler。
- [ ] 支持 stage2 `views_curriculum=[24, 320]`，显存不足时先支持 `[8, 64]` 或 `[16, 128]`。
- [ ] 支持 stage2 local window `k=[16, 64]`，显存不足时先支持 `[8, 32]`。
- [ ] 支持 local window 内 relative pose all-pairs；若 pair subsampling，日志记录 subsample ratio。
- [ ] 支持 view order shuffle。
- [ ] 支持 dynamic batch packing，限制每 GPU 最多 48 images。

### 模型

- [ ] 加载完整 LingBot/GCTStream checkpoint `lingbot-map.pt`。
- [ ] 确认使用 GCTStream 内置 DINOv2 ViT-L/14，而不是把 `dinov2_vits14` 作为正式 Stage1 模型。
- [ ] 冻结完整 aggregator：DINOv2 + frame_blocks + global_blocks。
- [ ] aggregator 前向使用 `torch.no_grad()`，但保留完整模型结构参与特征生成。
- [ ] Stage2 设置 `max_frame_num >= views_end`；若 pilot 降 views，也同步记录该上限。
- [ ] 若 GCTStream checkpoint 支持 GCA/streaming context，Stage2 前向可使用 frozen GCA/anchor/trajectory memory，但 `train_gca=false`。
- [ ] 每个 iteration/batch 清理 KV cache，避免 streaming state 跨样本泄漏。
- [ ] 启用 GCTStream 原生 DPT `depth_head`。
- [ ] 启用 GCTStream 原生 `camera_head` / PoseHead。
- [ ] optimizer 只接收 trainable heads 参数。
- [ ] 确认未将 DINOv2、aggregator、GCA、multi-view/window 模块参数传入 optimizer。

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
- [ ] Stage2 正式路线不再使用旧 `train_replica_stage2.py` 的 `HeadOnlyModel` 作为主实验；应迁移到完整 GCTStream head-only 训练脚本。
- [ ] 保存并验证 checkpoint。

## 14. 可写入报告的总结

本阶段在已验证的 head-only 框架基础上，先向 LingBot-Map 论文第 4.1 节 Base Model Training 对齐，再迁移第 4.2 节 Streaming Model Training 的长序列训练策略。由于算力受限，训练使用完整 LingBot/GCTStream 架构和 `lingbot-map.pt` 初始化，但冻结 DINOv2 backbone、frame/global blocks 等 aggregator 主体，只训练深度头与相机/位姿头；训练数据实际使用本地 Replica RGB-D 长序列，ScanNet 流程仅作为可迁移参考。第一阶段保留 2-24 views sampling、最大边 518 的目标设置、强 photometric augmentation、AdamW `2e-4`、weight decay `0.05`、5% warmup 和 cosine decay。DINOv2 论文中的 DINO/iBOT/KoLeo、teacher-student EMA、625k 自监督预训练和 10k high-resolution adaptation 不迁移到当前 head-only 训练，只作为冻结视觉特征与 DPT-style 下游 head 的依据。第二阶段应先加载完整 GCTStream base，再从第一阶段 checkpoint 覆盖 heads，使用 Replica foldback video sampler、progressive view curriculum、local window all-pairs relative pose loss，并将学习率调整为论文 4.2 的 `5e-4`；若使用 frozen GCTStream/GCA 前向，需要保证 `max_frame_num >= views_end`、KV cache 按 batch 清理，并明确 `train_gca=false`。该方案不复现完整 GCA streaming model 的全参数训练，而是在完整模型前向和有限资源下验证 4.2 长序列训练策略对 `DepthHead + PoseHead` 的可运行迁移。

## 15. 参考来源

- `Geometric Context Transformer for.pdf`：主论文，Sec. 3、Sec. 4.1、Sec. 4.3、Appendix A.1。
- `Wang_VGGT_Visual_Geometry_Grounded_Transformer_CVPR_2025_paper.pdf`：VGGT 架构、DINO backbone、alternating attention、camera/depth heads。
- `π3 Permutation-equivariant visual geometry learning.pdf`：relative pose loss 和相对位姿监督思想。
- `Dinov2 Learning robust visual features without supervision.pdf`：DINOv2 作为冻结视觉特征提取器的依据。
- `README.md`：本次 head-only 训练验证记录。
- `README_new.md`：论文 4.1 对齐版 head-only 验证总结。
- `stage1_scannet_training_plan_modified.md`：上一版资源受限训练计划。
- ScanNet 官方 SensReader / exporter 文档：[SensReader](https://www.scan-net.org/ScanNet/SensReader/) 与 [Python Data Exporter](https://www.scan-net.org/ScanNet/SensReader/python/)。
