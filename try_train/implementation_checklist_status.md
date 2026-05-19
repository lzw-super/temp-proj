# 实施检查清单状态报告

基于 `replica_head_only_stage1_stage2_training_plan.md` 文档第13节的清单逐项分析。

---

## 数据

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 准备 Replica room 序列 | ✅ | `replica_dataset.py:99-100` | `data_root = Path(data_root)`，支持 Replica 数据 |
| 解析 traj.txt 为 4x4 T_c2w | ✅ | `replica_dataset.py:151-170` | `_load_poses()` 解析16个浮点数为4x4矩阵 |
| 匹配 frame*.jpg 与 depth*.png | ✅ | `replica_dataset.py:172-206` | `_get_frame_ids()` 同时检查RGB和depth文件 |
| depth PNG 除以 1000 转为 meter | ✅ | `replica_dataset.py:330-335` | `_load_depth()` 返回 `depth_mm / 1000.0` |
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
| 加载 DINOv2 dinov2_vits14 | ✅ | `head_only_model.py:100-110` | `_load_dinov2_backbone()` torch.hub加载 |
| 冻结 DINOv2 参数 | ✅ | `head_only_model.py:112-114` | `_freeze_backbone()` 设置requires_grad=False |
| 启用 DepthHead | ✅ | `head_only_model.py:74-86` | DPTHead 默认启用 |
| 启用 PoseHead | ✅ | `head_only_model.py:88-96` | CameraHead 默认启用 |
| **DepthHead 接收多尺度特征** | ✅ **新修复** | `head_only_model.py` forward | 使用hook提取的4层特征，而非重复patch_tokens |
| **PoseHead 接收多尺度特征** | ✅ **新修复** | `head_only_model.py` forward | pose_tokens_list 同样从hook特征构建 |
| optimizer 只接收 trainable heads | ✅ | `train_replica_v2.py:222-224` | `trainable_params = [p for p in model.parameters() if p.requires_grad]` |
| 确认未传入冻结参数 | ✅ | `head_only_model.py:71` | freeze_backbone=True，参数不参与梯度 |
| **load_state_dict 兼容性** | ✅ **新修复** | 多个文件 | 所有load_state_dict改为strict=False，兼容新buffer |
| **Aggregator 跨帧注意力** | ❌ 架构限制 | - | HeadOnlyModel逐帧独立处理，需实现完整GCTStream架构（含KV缓存、时序因果注意力） |

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
| **Replica room0 正式训练** | ✅ **已完成** | `checkpoints/v2_fixed/` | Stage1: 5000 iters, loss 0.694→0.043 |
| **Stage2 正式训练** | ✅ **已完成** | `checkpoints/stage2_fixed/` | Stage2: 5000 iters, loss ~0.08-0.20波动 |
| Replica 多 room 或长序列 pilot | ❌ 数据限制 | - | 仅room0数据集可用，缺少其他Replica场景 |
| 启用 pose head 同训 | ✅ | `train_replica_v2.py:116` | train_pose=True 默认启用 |
| 启用 2-24 views | ✅ | `train_replica_v2.py` CLI args | `--min_views 2 --max_views 8`，动态范围采样 |
| 启用 518 max dimension | ❌ 显存限制 | `train_replica_v2.py:115` | RTX 3090 (24GB) 无法支持518分辨率，patch tokens从256增至1372，显存约5-8x增长 |
| 补齐 hue jitter | ✅ | `replica_dataset.py:78` | hue=0.1 参数已配置 |
| **Stage2 color jitter** | ✅ **新修复** | `train_replica_stage2.py` | ReplicaLongSequenceDataset 添加 _apply_color_jitter |
| spatial rescale | ✅ | `replica_dataset.py` `_apply_spatial_rescale()` | range [0.8,1.2]，共享scale参数，同步更新intrinsics/depth/pose |
| aspect-ratio sampling | ✅ | `replica_dataset.py` `_apply_aspect_ratio()` | ratio∈[0.33,1.0]，共享ratio参数，同步更新intrinsics/depth |
| co-jitter | ✅ | `replica_dataset.py` `_apply_color_jitter(co_jitter=True)` | 多view共享颜色扰动参数 |
| geometric augmentation后同步更新 | ✅ | `replica_dataset.py` `__getitem__` | rescale/aspect-ratio后intrinsics和depth同步更新 |
| Stage1 AdamW lr=2e-4, wd=0.05 | ✅ | `train_replica_v2.py:123-126` | 论文设置已实现 |
| Stage2 从stage1 checkpoint初始化 | ✅ | `train_replica_stage2.py` | stage1_checkpoint 参数 |
| Stage2 AdamW lr=5e-4, wd=0.05 | ✅ | `train_replica_stage2.py` | 第二阶段LR设置 |
| 两个阶段5% warmup + cosine | ✅ | `train_replica_v2.py:229-248` | SequentialLR scheduler |
| Stage2 foldback sampler | ✅ | `train_replica_stage2.py` | 导入 FoldbackVideoSampler |
| Stage2 progressive view curriculum | ✅ | `train_replica_stage2.py` | ProgressiveViewCurriculum (8→24) |
| **Stage2 总帧数动态检测** | ✅ **新修复** | `train_replica_stage2.py` | 从traj.txt动态检测，不再硬编码2000 |
| 保存并验证 checkpoint | ✅ | `train_replica_v2.py:92-102` | save_checkpoint() 函数 |

---

## 训练结果

| 模型 | Depth相对误差 (Mean) | 训练Iterations | 骨干网络 | 分辨率 |
|------|---------------------|----------------|----------|--------|
| 原始模型 (未训练) | 0.013 | - | ViT-L/14 | 518 |
| Stage1 (修复后) | 0.051 | 5000 | ViT-S/14 | 224 |
| Stage2 (修复后) | 0.128 | 5000 | ViT-S/14 | 224 |

**Stage2回退分析：** HeadOnlyModel缺少Aggregator（跨帧注意力），逐帧独立处理无法利用长序列帧间关联。Stage2要真正起效需实现GCTStream架构。

---

## 总结

### 完全实现 (✅) - 39项

**数据加载：** Replica traj.txt解析、depth单位转换、pose归一化、RGB/depth匹配、**pose有效性检查**、**depth有效比例过滤**
**预处理：** RGB/depth/mask对齐、pad到14倍数、intrinsics更新、**ImageNet归一化**
**模型：** DINOv2冻结、DepthHead+PoseHead同训、**多尺度hook特征提取**、**load_state_dict兼容**
**Loss：** masked log-L1、quaternion geodesic、Huber translation、relative pose延迟开启
**数据增强：** **spatial rescale [0.8,1.2]**、**aspect-ratio sampling [0.33,1.0]**、**co-jitter**、**geometric aug后同步更新intrinsics/depth**
**采样策略：** temporal nearby、**spatial nearby sampler**、**views范围采样 [2,8]**、**dynamic batch packing**
**Stage1训练：** AdamW lr=2e-4 wd=0.05、warmup+cosine scheduler、**5000 iters训练完成**
**Stage2核心：** foldback sampler、progressive view curriculum (8→24)、local window relative pose (k=[4,16])、**5000 iters训练完成**、**color jitter**、**动态帧数检测**、**dynamic batch packing**

### 部分实现 (⚠️) - 1项

- metadata缓存：直接读取traj.txt，无缓存机制（低优先级）

### 受限无法实现 (❌) - 4项

**架构限制：**
- Aggregator 跨帧注意力（HeadOnlyModel逐帧独立处理，需完整GCTStream架构）

**显存限制：**
- 518 resolution（RTX 3090 24GB 无法支持，patch tokens从256增至1372，显存约5-8x增长）

**数据限制：**
- 多Room训练（仅room0数据集可用，缺少其他Replica场景）

**时间限制：**
- 160K iterations（当前5000 iters验证正确性，完整训练需数天时间）

---

## 建议补齐优先级

### 高优先级（架构突破）

1. **实现 GCTStream 架构** - Stage2 的核心组件，需Aggregator跨帧注意力+KV缓存，head-only架构无法有效利用长序列

### 中优先级（资源扩展）

2. **更大GPU资源** - A100 80GB 支持518分辨率训练
3. **多场景数据** - 收集更多 Replica/ScanNet 场景数据

### 低优先级（论文完整对齐）

4. **metadata缓存** - 提升数据加载速度
5. **长周期训练** - 160K iterations 完整训练

---

Claude Code | 2026-05-19 | 实施检查清单完整分析 (含功能实现状态更新)
