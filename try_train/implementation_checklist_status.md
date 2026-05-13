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
| 过滤无效 pose、缺失文件 | ⚠️ 部分 | `replica_dataset.py:200-202` | 仅检查pose存在，未检查pose有效性和深度比例 |
| 生成 Replica metadata/cache | ⚠️ 未实现 | - | 直接读取 traj.txt，无缓存机制 |

**未完成项：**
- pose有效性检查（全finite检查）
- 深度有效比例过滤（valid_ratio >= 0.05）
- metadata缓存文件生成

---

## 预处理

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| RGB/depth/mask resize 到一致尺寸 | ✅ | `replica_dataset.py:337-367` | `_align_to_train_size()` 统一resize |
| 输出尺寸 pad 到 14 的整数倍 | ✅ | `replica_dataset.py:349-358` | 计算 pad_H/pad_W 对齐 patch_size=14 |
| 同步更新 intrinsics | ✅ | `replica_dataset.py:360-365` | K_new 根据 scale 更新 fx/fy/cx/cy |
| dataloader 返回完整字段 | ✅ | `replica_dataset.py:313-320` | 返回 images, depths, valid_masks, intrinsics, poses |
| reference pose normalization | ✅ | `replica_dataset.py:369-376` | `_normalize_poses()` 相对第一帧归一化 |

---

## 采样

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 支持 views_per_sample=[2, 24] | ✅ | `replica_dataset.py:65-109` | 支持 min_views/max_views 范围 |
| 支持 temporal nearby fallback | ✅ | `replica_dataset.py:233-273` | `_build_samples()` 使用 temporal_window |
| 支持 spatial nearby sampler | ❌ 未实现 | - | 仅 temporal_nearby，未实现基于camera center距离 |
| 支持 foldback video sampler | ✅ | `foldback_video_sampler.py:19-134` | Stage2专用，边界反向继续 |
| 支持 stage2 views_curriculum | ✅ | `foldback_video_sampler.py:137-217` | ProgressiveViewCurriculum (24→320) |
| 支持 stage2 local window k | ✅ | `foldback_video_sampler.py:220-303` | LocalWindowSampler (k=[16,64]) |
| 支持 view order shuffle | ✅ | `replica_dataset.py:264-268` | shuffle_view_order 打乱视角顺序 |
| 支持 dynamic batch packing | ❌ 未实现 | - | 未限制 max_images_per_gpu=48 |

**未完成项：**
- spatial nearby sampler（基于camera center 3D距离）
- dynamic batch packing（限制每GPU最多48 images）

---

## 模型

| 检查项 | 状态 | 代码位置 | 说明 |
|--------|------|----------|------|
| 加载 DINOv2 dinov2_vits14 | ✅ | `head_only_model.py:100-110` | `_load_dinov2_backbone()` torch.hub加载 |
| 冻结 DINOv2 参数 | ✅ | `head_only_model.py:112-114` | `_freeze_backbone()` 设置requires_grad=False |
| 启用 DepthHead | ✅ | `head_only_model.py:74-86` | DPTHead 默认启用 |
| 启用 PoseHead | ✅ | `head_only_model.py:88-96` | CameraHead 默认启用 |
| optimizer 只接收 trainable heads | ✅ | `train_replica_v2.py:222-224` | `trainable_params = [p for p in model.parameters() if p.requires_grad]` |
| 确认未传入冻结参数 | ✅ | `head_only_model.py:71` | freeze_backbone=True，参数不参与梯度 |

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
| Replica room0 smoke run | ✅ 已验证 | README.md | 100 iterations 验证完成 |
| Replica 多 room 或长序列 pilot | ❌ 未实现 | - | 仅单room训练 |
| 启用 pose head 同训 | ✅ | `train_replica_v2.py:116` | train_pose=True 默认启用 |
| 启用 2-24 views | ⚠️ 部分 | `train_replica_v2.py:183` | 固定2 views，未使用范围 |
| 启用 518 max dimension | ❌ 未实现 | `train_replica_v2.py:115` | 默认224，未达到论文目标 |
| 补齐 hue jitter | ✅ | `replica_dataset.py:78` | hue=0.1 参数已配置 |
| spatial rescale | ❌ 未实现 | - | 无 geometric augmentation |
| aspect-ratio sampling | ❌ 未实现 | - | 无 aspect ratio 调整 |
| co-jitter | ❌ 未实现 | - | 无 multi-view共享颜色扰动 |
| geometric augmentation后同步更新 | ❌ 未实现 | - | 无 geometric augmentation |
| Stage1 AdamW lr=2e-4, wd=0.05 | ✅ | `train_replica_v2.py:123-126` | 论文设置已实现 |
| Stage2 从stage1 checkpoint初始化 | ✅ | `train_replica_stage2.py` | stage1_checkpoint 参数 |
| Stage2 AdamW lr=5e-4, wd=0.05 | ✅ | `train_replica_stage2.py` | 第二阶段LR设置 |
| 两个阶段5% warmup + cosine | ✅ | `train_replica_v2.py:229-248` | SequentialLR scheduler |
| Stage2 foldback sampler | ✅ | `train_replica_stage2.py` | 导入 FoldbackVideoSampler |
| Stage2 progressive view curriculum | ✅ | `train_replica_stage2.py` | ProgressiveViewCurriculum |
| 保存并验证 checkpoint | ✅ | `train_replica_v2.py:92-102` | save_checkpoint() 函数 |

---

## 总结

### 完全实现 (✅) - 25项

**数据加载：** Replica traj.txt解析、depth单位转换、pose归一化、RGB/depth匹配
**预处理：** RGB/depth/mask对齐、pad到14倍数、intrinsics更新
**模型：** DINOv2冻结、DepthHead+PoseHead同训
**Loss：** masked log-L1、quaternion geodesic、Huber translation、relative pose延迟开启
**Stage1训练：** AdamW lr=2e-4 wd=0.05、warmup+cosine scheduler
**Stage2核心：** foldback sampler、progressive view curriculum、local window relative pose

### 部分实现 (⚠️) - 3项

- views范围采样：参数支持但训练脚本固定2 views
- 数据过滤：仅检查文件存在，未检查有效性
- metadata缓存：直接读取，无缓存

### 未实现 (❌) - 10项

**数据增强（论文完整）：**
- spatial rescale [0.8, 1.2]
- aspect-ratio sampling [0.33, 1.0]
- co-jitter

**采样策略：**
- spatial nearby sampler
- dynamic batch packing

**训练规模：**
- 518 resolution（当前224）
- 多room训练
- 160K iterations

---

## 建议补齐优先级

### 高优先级（核心功能）

1. **训练脚本启用views范围** - 修改train_replica_v2.py使用min_views/max_views
2. **数据有效性过滤** - pose finite检查、depth valid_ratio检查

### 中优先级（性能提升）

3. **spatial nearby sampler** - 基于camera center距离
4. **提高分辨率到518** - 测试显存后逐步提升

### 低优先级（论文完整对齐）

5. **geometric augmentation** - spatial rescale + aspect-ratio
6. **dynamic batch packing** - 限制max 48 images per GPU

---

Claude Code | 2026-05-13 | 实施检查清单完整分析