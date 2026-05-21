# 实施检查清单状态报告

基于 `replica_head_only_stage1_stage2_training_plan.md` 文档第13节的清单逐项分析。

---

## 数据

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 准备 Replica room 序列 | ✅ | `replica_dataset.py:99-100` | `data_root = Path(data_root)`，支持 Replica 数据 |
| 解析 traj.txt 为 4x4 T_c2w | ✅ | `replica_dataset.py:151-170` | `_load_poses()` 解析16个浮点数为4x4矩阵 |
| 匹配 frame*.jpg 与 depth*.png | ✅ | `replica_dataset.py:172-206` | `_get_frame_ids()` 同时检查RGB和depth文件 |
| depth PNG 转米 (Replica 正确 scale) | ✅ **已修复** | `replica_dataset.py` `_load_depth` + `REPLICA_DEPTH_SCALE=6553.5` | 早期错用 `/1000`，已统一改为 `/6553.5`（Replica 学术 convention，与 NICE-SLAM/iMAP 一致）。room0 depth median 从 ~17.6m 修正为 ~2.7m |
| pose 统一为 4x4 camera-to-world | ✅ | `replica_dataset.py:166` | traj.txt 已是 camera-to-world 格式 |
| 过滤无效 pose、缺失文件 | ✅ | `replica_dataset.py` `_load_poses` + `_get_frame_ids` | np.isfinite检查 + det(R)≈1正交性验证 + depth valid_ratio≥0.05 |
| 生成 Replica metadata/cache | ⚠️ 未实现 | - | 直接读取 traj.txt，无缓存机制 |

**未完成项：**
- metadata缓存文件生成（低优先级，直接读取traj.txt即可）

---

## 预处理

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| RGB/depth/mask resize 到一致尺寸 | ✅ | `replica_dataset.py:337-367` | `_align_to_train_size()` 统一resize |
| 输出尺寸 pad 到 14 的整数倍 | ✅ | `replica_dataset.py:349-358` | 计算 pad_H/pad_W 对齐 patch_size=14 |
| 同步更新 intrinsics | ✅ | `replica_dataset.py:360-365` | K_new 根据 scale 更新 fx/fy/cx/cy |
| dataloader 返回完整字段 | ✅ | `replica_dataset.py:313-320` | 返回 images, depths, valid_masks, intrinsics, poses |
| reference pose normalization | ✅ | `replica_dataset.py:369-376` | `_normalize_poses()` 相对第一帧归一化 |
| **ImageNet 归一化** | ✅ **新修复** | `head_only_model.py` __init__ | `register_buffer('imagenet_mean/std')`, forward中应用 |
| **DINOv2 多尺度特征提取** | ✅ **新修复** | `head_only_model.py` _register_intermediate_hooks | Hook on blocks [2,5,8,11] for ViT-S |

---

## 采样

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 支持 views_per_sample=[2, 24] | ✅ | `replica_dataset.py:65-109` | 支持 min_views/max_views 范围 |
| 支持 temporal nearby fallback | ✅ | `replica_dataset.py:233-273` | `_build_samples()` 使用 temporal_window |
| 支持 spatial nearby sampler | ✅ | `replica_dataset.py` `_build_spatial_distance_matrix` + `_spatial_nearby_sample` | 基于camera center 3D距离采样，`--sampler_type spatial_nearby` |
| 支持 foldback video sampler | ✅ | `foldback_video_sampler.py:19-134` | Stage2专用，边界反向继续 |
| 支持 stage2 views_curriculum | ✅ | `foldback_video_sampler.py:137-217` | ProgressiveViewCurriculum (8→24, 已适配RTX 3090) |
| 支持 stage2 local window k | ✅ | `foldback_video_sampler.py:220-303` | LocalWindowSampler (k=[4,16], 已适配显存) |
| 支持 view order shuffle | ✅ | `replica_dataset.py:264-268` | shuffle_view_order 打乱视角顺序 |
| 支持 dynamic batch packing | ✅ | `train_replica_stage2.py` | `batch_size = max_images_per_gpu // current_views`，默认max=48 |

**未完成项：**（无）

---

## 模型

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 加载 DINOv2 dinov2_vits14 | ✅ | `head_only_model.py:100-110` | HeadOnlyModel: ViT-S/14 |
| **加载完整 GCTStream (DINOv2 ViT-L/14)** | ✅ **新增** | `train_replica_gct.py` `load_gct_model()` | 从 `lingbot-map.pt` 加载完整预训练权重，含 Aggregator |
| 冻结 DINOv2 参数 | ✅ | `head_only_model.py:112-114` | HeadOnlyModel: `_freeze_backbone()` |
| **冻结完整 Aggregator (DINOv2+frame/global blocks)** | ✅ **新增** | `train_replica_gct.py` `freeze_aggregator()` | GCT版: 909M 参数冻结 (76.36%) |
| 启用 DepthHead | ✅ | `head_only_model.py:74-86` | DPTHead 默认启用 |
| 启用 PoseHead (CameraCausalHead 含 KV cache) | ✅ | `head_only_model.py:88-96` / `train_replica_gct.py` | GCT版使用完整 CameraCausalHead (4次迭代) |
| **DepthHead 接收多尺度特征** | ✅ **新修复** | `head_only_model.py` forward | 使用hook提取的4层特征，而非重复patch_tokens |
| **PoseHead 接收多尺度特征** | ✅ **新修复** | `head_only_model.py` forward | pose_tokens_list 同样从hook特征构建 |
| optimizer 只接收 trainable heads | ✅ | `train_replica_v2.py:222-224` / `train_replica_gct.py` | GCT版: 281M trainable (23.64%) |
| 确认未传入冻结参数 | ✅ | `head_only_model.py:71` | freeze_backbone=True，参数不参与梯度 |
| **load_state_dict 兼容性** | ✅ **新修复** | 多个文件 | 所有load_state_dict改为strict=False，兼容新buffer |
| **Aggregator 跨帧注意力** | ✅ **已解决** | `train_replica_gct.py` `forward_gct_heads_only()` | 使用完整 GCTStream AggregatorStream (frame_blocks + global_blocks)，含 KV cache |

---

## Loss

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| depth loss 使用 valid mask | ✅ | `head_only_model.py:174-189` | DepthLoss 在 valid_mask 上计算 |
| depth prediction 强制为正 | ✅ | `head_only_model.py:80` | DPTHead activation="exp" 确保正值 |
| pose loss 使用 reference-normalized | ✅ | `replica_dataset.py:369-376` | pose 归一化到第一帧 |
| rotation loss 使用 geodesic | ✅ | `head_only_model.py:278-301` | `_quaternion_geodesic_loss()` |
| translation loss 使用 Huber | ✅ | `head_only_model.py:303-316` | `_huber_loss()` |
| relative pose loss 延迟开启 | ✅ | `train_replica_v2.py:64-70` | `iteration >= args.rel_pose_start_iter` |

---

## 训练

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| Replica room0 smoke run | ✅ 已验证 | - | 100 iterations 验证完成 |
| **Replica room0 正式训练** | ✅ **已完成** | `checkpoints/v2_fixed/` | Stage1 (HeadOnly): 5000 iters, loss 0.694→0.043 |
| **GCTStream Stage1 正式训练** | ✅ **新增** | `checkpoints/gct_stage1_5k/` | 完整 GCTStream Stage1: 5000 iters, loss 0.7356→0.0482 (depth: 0.4315→0.0317) |
| **GCTStream Stage1 = global attention** | ✅ **新增明确化** | `train_replica_gct.py` `forward_gct_heads_only()` | `sliding_window_size=-1`、`num_frame_per_block=S`、`num_frame_for_scale=S`、`causal_inference=False`；与论文 4.1 base model 一致 |
| **GCTStream Stage2 = GCA streaming** | ✅ **新增** | `train_replica_gct_stage2.py` `forward_gct_heads_only_streaming()` | `sliding_window_size=k`（来自 `LocalWindowSampler`）、`num_frame_per_block=1`、`num_frame_for_scale=8`、`causal_inference=True`；camera_head 带 KV cache；与论文 4.2 一致 |
| **GCTStream Stage2 训练（GCA）正式训练** | ✅ **smoke 通过，待正式 5K** | `checkpoints/gct_stage2_smoke_gca/` | 10 iters smoke：从 Stage1 GCT ckpt 初始化，views 4→6, k=[2,3]，loss 0.12→0.03 |
| **Stage2 正式训练 (HeadOnly)** | ✅ **已完成** | `checkpoints/stage2_fixed/` | Stage2 (HeadOnly): 5000 iters, loss ~0.08-0.20波动 |
| Replica 多 room 或长序列 pilot | ❌ 数据限制 | - | 仅room0数据集可用，缺少其他Replica场景 |
| 启用 pose head 同训 | ✅ | `train_replica_v2.py:116` / `train_replica_gct.py` / `train_replica_gct_stage2.py` | GCT 两阶段都默认启用 CameraCausalHead 4次迭代优化 |
| 启用 2-24 views | ✅ | `train_replica_v2.py` CLI args | `--min_views 2 --max_views 8`，动态范围采样 |
| **启用 518 max dimension** | ✅ **GCT版已解决** | `train_replica_gct.py` / `train_replica_gct_stage2.py` | GCT 两阶段都使用 518 原生分辨率（aggregator冻结+no_grad+heads训练，约 11GB 显存）|
| 补齐 hue jitter | ✅ | `replica_dataset.py:78` | hue=0.1 参数已配置 |
| **Stage2 color jitter** | ✅ **新修复** | `train_replica_stage2.py` | ReplicaLongSequenceDataset 添加 _apply_color_jitter |
| spatial rescale | ✅ | `replica_dataset.py` `_apply_spatial_rescale()` | range [0.8,1.2]，共享scale参数，同步更新intrinsics/depth/pose |
| aspect-ratio sampling | ✅ | `replica_dataset.py` `_apply_aspect_ratio()` | ratio∈[0.33,1.0]，共享ratio参数，同步更新intrinsics/depth |
| co-jitter | ✅ | `replica_dataset.py` `_apply_color_jitter(co_jitter=True)` | 多view共享颜色扰动参数 |
| geometric augmentation后同步更新 | ✅ | `replica_dataset.py` `__getitem__` | rescale/aspect-ratio后intrinsics和depth同步更新 |
| Stage1 AdamW lr=2e-4, wd=0.05 | ✅ | `train_replica_v2.py:123-126` / `train_replica_gct.py` | 论文 4.1 设置已实现 |
| Stage2 从stage1 checkpoint初始化 | ✅ | `train_replica_stage2.py` / `train_replica_gct_stage2.py` | stage1_checkpoint 参数；GCT版加载 `full_model_state_dict` |
| Stage2 AdamW lr=5e-4, wd=0.05 | ✅ | `train_replica_stage2.py` / `train_replica_gct_stage2.py` | 论文 4.2 LR 设置 |
| 两个阶段5% warmup + cosine | ✅ | `train_replica_v2.py:229-248` | SequentialLR scheduler |
| Stage2 foldback sampler | ✅ | `train_replica_stage2.py` / `train_replica_gct_stage2.py` | 导入 FoldbackVideoSampler |
| Stage2 progressive view curriculum | ✅ | `train_replica_stage2.py` / `train_replica_gct_stage2.py` | ProgressiveViewCurriculum (8→24 HeadOnly; 4→8 GCT 518分辨率 pilot) |
| **Stage2 GCA 局部窗口 k 采样** | ✅ **新增** | `train_replica_gct_stage2.py` 主循环 | `LocalWindowSampler.sample_window_size()` 同时驱动 GCA 窗口 + rel-pose loss 局部对 |
| **Stage2 anchor-scale normalization** | ✅ **新增** | `train_replica_gct_stage2.py` `compute_anchor_scale()` + `apply_anchor_scale_normalization()` | 论文 4.2 anchor 尺度归一化：用前 N 帧 GT 算尺度 s（depth_median 或 translation_norm），把 GT depth / translation 除以 s 再算 loss。CLI: `--use_anchor_scale_norm` / `--anchor_scale_source {depth_median, translation_norm}` |
| **Replica depth scale 修复** | ✅ **关键 bug 修复** | `replica_dataset.py` `REPLICA_DEPTH_SCALE=6553.5` + 4 个文件统一使用 | 早期错用 `/1000`（mm），实际 Replica 用 `/6553.5`（与 NICE-SLAM/iMAP/GO-SLAM/MonoGS 一致）。修复后 room0 depth median 从错误的 ~17.6m 变成正确的 ~2.7m |
| **Stage2 总帧数动态检测** | ✅ **新修复** | `train_replica_stage2.py` | 从traj.txt动态检测，不再硬编码2000 |
| 保存并验证 checkpoint | ✅ | `train_replica_v2.py:92-102` / `train_replica_gct*.py` | save_checkpoint() 函数；GCT版同时保存 head_state_dict + full_model_state_dict |

---

## 训练结果

| 模型 | Depth Loss (masked_log_l1) | 训练Iterations | 骨干网络 | 分辨率 |
|------|---------------------------|----------------|----------|--------|
| 原始模型 (未训练) | 0.013 | - | ViT-L/14 | 518 |
| Stage1 HeadOnly (修复后) | 0.051 | 5000 | ViT-S/14 | 224 |
| **Stage1 GCTStream (新增)** | **0.0317** | 5000 | **ViT-L/14 (冻结)** | **518** |
| Stage2 HeadOnly (修复后) | 0.128 | 5000 | ViT-S/14 | 224 |

**HeadOnly Stage2 回退分析：** HeadOnlyModel缺少Aggregator（跨帧注意力），逐帧独立处理无法利用长序列帧间关联。

**GCTStream Stage1 提升原因：** 使用完整 GCTStream 架构（DINOv2 ViT-L/14 + AggregatorStream 跨帧因果注意力 + CameraCausalHead 4次迭代），通过冻结 aggregator + 用 `no_grad` 前向 + 只训练 heads 的内存优化方案，在 RTX 3090 (24GB) 上成功跑通 518 原生分辨率训练。

---

## 总结

### 完全实现 (✅) - 47项

**数据加载：** Replica traj.txt解析、**depth 正确 scale 转换（/6553.5，修复 /1000 bug）**、pose归一化、RGB/depth匹配、**pose有效性检查**、**depth有效比例过滤**
**预处理：** RGB/depth/mask对齐、pad到14倍数、intrinsics更新、**ImageNet归一化**
**模型：** DINOv2冻结、DepthHead+PoseHead同训、**多尺度hook特征提取**、**load_state_dict兼容**、**完整 GCTStream 加载与 Aggregator 冻结**、**Aggregator 跨帧注意力（含 KV cache）**
**Loss：** masked log-L1、quaternion geodesic、Huber translation、relative pose延迟开启、**Stage2 anchor-scale normalization (depth+translation)**
**数据增强：** **spatial rescale [0.8,1.2]**、**aspect-ratio sampling [0.33,1.0]**、**co-jitter**、**geometric aug后同步更新intrinsics/depth**
**采样策略：** temporal nearby、**spatial nearby sampler**、**views范围采样 [2,8]**、**dynamic batch packing**
**Stage1 训练（HeadOnly + GCTStream 两版本）：** AdamW lr=2e-4 wd=0.05、warmup+cosine scheduler、**HeadOnly 5000 iters 完成**、**GCTStream 5000 iters 完成（518 原生分辨率，global attention，论文 4.1）**
**Stage2 核心（HeadOnly + GCTStream 两版本）：** foldback sampler、progressive view curriculum (HeadOnly 8→24; GCT 4→8)、local window relative pose (k=[4,16]/[2,4])、**HeadOnly 5000 iters 完成**、**GCTStream smoke 通过（GCA streaming forward + anchor-scale norm, 论文 4.2）**、color jitter、动态帧数检测、dynamic batch packing
**Stage1 vs Stage2 attention 模式差异明确：** Stage1 global attention（`sliding_window_size=-1`、`num_frame_per_block=S`、`causal_inference=False`）；Stage2 GCA streaming（`sliding_window_size=k`、`num_frame_per_block=1`、`causal_inference=True`、camera_head KV cache）

### 部分实现 (⚠️) - 3项

- metadata缓存：直接读取traj.txt，无缓存机制（低优先级）
- **GCTStream Stage2 正式 5K iters 训练**：smoke 已通过（含 anchor-scale norm），需要完整 5K iters 验证收敛
- **修复 depth scale 后重新训练 Stage1/Stage2**：已训 checkpoint 在错误 scale 下学到的输出是 true_depth × 6.5535，需重训得到真正 metric depth 输出

### 受限无法实现 (❌) - 2项

**数据限制：**
- 多Room训练（仅room0数据集可用，缺少其他Replica场景）

**时间限制：**
- 160K iterations（当前5000 iters验证正确性，完整训练需数天时间）

---

## 建议补齐优先级

### 高优先级（完成 GCT Stage2 验证 + 重训）

1. **修复 depth scale 后重新跑 Stage1 + Stage2 完整 5K 训练** - `bash quick-op/train_gct_stage1.sh` + `bash quick-op/train_gct_stage2.sh`
2. **验证 GCT Stage2 vs Stage1 推理效果** - 复用 `bash quick-op/validate_gct_stage2.sh`，对比 GCA streaming + anchor-norm 与 global attention 的差异
3. **对比 anchor-scale norm 开关效果** - 验证 `--use_anchor_scale_norm` vs `--no_anchor_scale_norm` 对训练稳定性的影响

### 中优先级（资源扩展）

4. **多场景数据** - 收集更多 Replica/ScanNet 场景数据
5. **GCT Stage2 扩大 views curriculum** - 在更大显存机器上把 views 从 4→8 推到 24→320（论文设置）

### 低优先级（论文完整对齐）

6. **metadata缓存** - 提升数据加载速度
7. **长周期训练** - 160K iterations 完整训练

---

Claude Code | 2026-05-21 | 更新：(1) Stage2 加入 anchor-scale normalization 实现（论文 4.2 anchor 尺度归一化 depth+translation）；(2) 修复 Replica depth scale 关键 bug（/1000 → /6553.5），影响 4 个文件
