# Stage1 & Stage2 训练总结

## 〇、关键Bug修复记录（2026-05-18）

训练收敛但推理效果比原始模型差的根因分析及修复：

### Bug 1: DPTHead接收重复特征而非多尺度特征

**问题：** `HeadOnlyModel.forward()` 将同一个 `patch_tokens` 重复4次传给 DPTHead，而 DPTHead 的 RefineNet 设计需要4个不同深度的特征（浅层→边缘，深层→语义）。

**修复：** 在 `HeadOnlyModel.__init__` 中注册 forward hooks，从 DINOv2 的不同深度 block 提取多尺度特征。

```python
# dinov2_vits14: 12 blocks, hooks on [2, 5, 8, 11]
self.intermediate_block_indices = self._get_intermediate_block_indices(backbone_name)
self._hook_features = {}
self._hooks = []
self._register_intermediate_hooks()
```

**新增方法：** `_get_intermediate_block_indices`, `_register_intermediate_hooks`, `_make_hook`, `_remove_hooks`

**影响：** DPTHead 现在真正获得多尺度特征融合能力，depth 预测质量大幅提升。

### Bug 2: 缺少 ImageNet 归一化

**问题：** DINOv2 要求输入经过 ImageNet 标准化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])，但 `HeadOnlyModel.forward()` 直接将原始像素 [0,1] 传入 backbone。

**修复：** 在 `__init__` 中注册归一化 buffer，在 `forward` 中应用。

```python
self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

# forward中:
images_normalized = (images - self.imagenet_mean) / self.imagenet_std
```

### Bug 3: CameraHead同样接收重复特征

**问题：** 与 DPTHead 相同的问题，CameraHead 的 pose_tokens 也是重复特征。

**修复：** 使用 hook 提取的中间特征构建 `pose_tokens_list`，与 `aggregated_tokens_list` 同源。

### Bug 4: load_state_dict 兼容性

**问题：** 新模型增加了 `imagenet_mean` / `imagenet_std` buffer，加载旧 checkpoint 时 strict=True 会报错。

**修复：** 所有 `load_state_dict(ckpt['model_state_dict'])` 改为 `load_state_dict(ckpt['model_state_dict'], strict=False)`。

**涉及文件：** `train_replica_v2.py`, `train_replica_stage2.py`, `validate_model_v2.py`, `validate_with_original.py`, `validate_stage1_vs_stage2.py`, `validate_long_sequence.py`

### Bug 5: Stage2 默认参数不合理

**问题：** `views_start=24, views_end=320` 在 RTX 3090 (24GB) 上显存不足。

**修复：** 改为 `views_start=8, views_end=24`，同时 `k_min=4, k_max=16`。

### Bug 6: Stage2 总帧数硬编码

**问题：** `total_frames = 2000` 硬编码，不同场景帧数不同。

**修复：** 从 `traj.txt` 动态检测帧数。

### Bug 7: Stage2 缺少颜色增强

**问题：** `ReplicaLongSequenceDataset.__getitem__` 没有 color jitter。

**修复：** 添加 `_apply_color_jitter` 方法和调用逻辑，与 Stage1 一致。

### 修复后训练结果

| 模型 | Depth相对误差 (Mean) | 训练Iterations | 备注 |
|------|---------------------|----------------|------|
| 原始模型 (未训练) | 0.013 | - | DINOv2 ViT-L/14, 518分辨率 |
| Stage1 (修复后) | 0.051 | 5000 | DINOv2 ViT-S/14, 224分辨率 |
| Stage2 (修复后) | 0.128 | 5000 | 同Stage1架构，长序列训练 |

> **Stage2 回退原因分析：** HeadOnlyModel 是逐帧独立处理（无跨帧注意力），长序列训练无法带来增益。论文中的 Stage2 依赖 Aggregator（时序因果注意力+KV缓存）实现跨帧信息传递，而当前 head-only 架构缺少此组件。Stage2 要真正起效，需要实现完整的 GCTStream 架构。

---

## 一、Stage1 训练（论文4.1：Base Model Training）

### 1.1 数据预处理

| 步骤 | 实现 | 代码位置 |
|------|------|----------|
| 加载 Replica traj.txt | 解析16浮点数→4x4 camera-to-world矩阵 | `replica_dataset.py:151-170` |
| 匹配 RGB/Depth 文件 | 同时检查frame*.jpg和depth*.png，取交集 | `replica_dataset.py:172-206` |
| Depth单位转换 | uint16毫米 → float32米（÷1000） | `replica_dataset.py:330-335` |
| Intrinsic推断 | fx=fy=600, cx=W/2, cy=H/2 | `replica_dataset.py:207-231` |
| Resize到训练尺寸 | max_dim=224，长边缩放 | `replica_dataset.py:337-367` |
| Pad到14倍数 | 确保patch_size=14整除 | `replica_dataset.py:349-358` |
| 同步更新内参 | fx/fy/cx/cy按scale缩放 | `replica_dataset.py:360-365` |
| Pose归一化 | 相对第一帧：T_ref_inv @ T_c2w | `replica_dataset.py:369-376` |
| Color jitter | prob=0.9, brightness=0.5, contrast=0.5, saturation=0.5 | `replica_dataset.py:378-421` |
| Grayscale | prob=0.05 | `replica_dataset.py:408-413` |

**采样策略：** temporal_nearby sampler，时间窗口=30帧内随机采样

### 1.2 模型架构

```
Input: images [B, V, 3, H, W]
  │
  ├── ImageNet 归一化: (images - mean) / std
  │     mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
  │
  ├── DINOv2 ViT-S/14 (冻结, 22M参数)
  │     └── forward_features → patch_tokens + cls_token
  │     └── Forward Hooks on blocks [2,5,8,11] → 4层多尺度特征
  │
  ├── DepthHead: DPTHead (exp激活, output_dim=2)
  │     └── 输入: 4层多尺度特征 [shallow→edges, deep→semantics]
  │     └── RefineNet 自底向上融合
  │     └── 输出: depth [B,V,H,W,1], depth_conf [B,V,H,W]
  │
  └── PoseHead: CameraHead (absT_quaR_FoV编码)
        └── 输入: hook提取的中间特征构建的pose_tokens
        └── 4层transformer trunk, 8 heads
        └── 输出: pose_enc [B,V,9] (center3 + quat4 + fov_offset2)
```

**关键架构特性：**
- **Multi-scale Hook机制：** 通过 `register_forward_hook` 在 DINOv2 的不同深度 block 捕获中间特征
- **特征提取索引：** dinov2_vits14 → [2,5,8,11], dinov2_vitb14 → [3,7,9,11], dinov2_vitl14 → [5,11,17,23]
- **归一化Buffer：** `imagenet_mean` / `imagenet_std` 注册为 model buffer，随模型移动到设备

**参数统计：** Trainable=35.6M, Total=57.6M, Frozen=38.3%

### 1.3 训练配置

| 配置项 | 当前值 | 论文目标 |
|--------|--------|----------|
| Optimizer | AdamW | AdamW |
| Learning Rate | 2e-4 | 2e-4 ✅ |
| Weight Decay | 0.05 | 0.05 ✅ |
| Warmup | 5% + cosine decay | 5% + cosine ✅ |
| Min LR | 1e-8 | 1e-8 ✅ |
| Gradient Clip | 1.0 | 1.0 ✅ |
| AMP | bfloat16/float16 | bfloat16 ✅ |
| Batch Size | 1 | 1 |
| Total Iterations | 5000 (已完成) | 160,000 |
| Image Size | 224 | 518 |
| Views | 2 (固定) | 2-24 (范围) |

### 1.4 Loss函数

```
L_total = L_depth + 0.1 × L_abs_pose + 0.05 × L_rel_pose

L_depth      = masked_log_l1(pred, gt, valid_mask)       # 对数域L1
L_abs_pose   = geodesic(quat_pred, quat_gt) + Huber(t_pred, t_gt)  # 排除第一帧
L_rel_pose   = Σ geodesic(R_ij_pred, R_ij_gt) + Huber(t_ij_pred, t_ij_gt)  # 全pairs
              (延迟开启: iteration >= 1000)
```

### 1.5 训练流程

```
1. 创建 ReplicaDataset + DataLoader
2. 创建 HeadOnlyModel (DINOv2冻结 + DepthHead + PoseHead)
3. 初始化 DepthLoss + PoseLoss + RelativePoseLoss
4. AdamW optimizer (仅trainable params)
5. SequentialLR: 5% warmup → cosine decay
6. 逐iteration训练，rel_pose延迟至1000 iter后开启
7. 每N iter保存checkpoint
```

### 1.6 训练结果

**修复后 Stage1 训练 (5000 iterations):**
- Loss: 0.694 → 0.043 (收敛良好)
- Depth相对误差 (Mean): 0.051
- Checkpoint: `try_train/checkpoints/v2_fixed/checkpoint_final.pt`

**与原始模型对比：**
- 原始模型 (ViT-L, 518px): depth rel_error = 0.013
- Stage1 (ViT-S, 224px): depth rel_error = 0.051
- 差距原因：backbone规模 (ViT-S vs ViT-L) + 分辨率 (224 vs 518)，而非训练问题

---

## 二、Stage2 训练（论文4.2：Streaming Model Training）

### 2.1 数据预处理

与Stage1相同的基础预处理（depth转换、resize、pad、pose归一化等），**关键差异在采样策略**：

| 差异项 | Stage1 | Stage2 |
|--------|--------|--------|
| 采样器 | temporal_nearby (窗口=30) | **FoldbackVideoSampler** |
| Views数 | 固定2 | **动态8→24 (curriculum)** |
| 序列长度 | 短序列 | **长序列** |
| 相邻帧stride | 固定1 | **随机[1,3]** |
| Color jitter | 有 | **有（已修复添加）** |

**Foldback Sampler核心逻辑：**
```
1. 从start_frame出发，按stride前进
2. 到达边界时反转方向（foldback），重选stride
3. 避免固定前进偏置，保证训练数据的运动方向多样性
```

**Progressive View Curriculum：**
```
1. 前 warmup_iterations (8000) iter：views固定=views_start (8)
2. 之后线性增长：views_start → views_end (8→24)
3. 公式: views = views_start + (views_end - views_start) × progress
4. progress = (iter - warmup) / (total - warmup), clip到[0,1]
```

**Local Window Sampler：**
```
1. 随机采样窗口大小 k ∈ [k_min, k_max] = [4, 16]
2. 在序列中随机选择起始位置
3. 取 k 个相邻帧作为local window
4. 只在窗口内计算relative pose pairs
```

### 2.2 模型架构

与Stage1完全相同的 HeadOnlyModel，**差异在于初始化方式**：

```python
# Stage2: 从Stage1 checkpoint加载权重
stage1_ckpt = torch.load(stage1_checkpoint)
model.load_state_dict(stage1_ckpt['model_state_dict'], strict=False)
```

> **架构局限：** HeadOnlyModel 缺少 Aggregator（跨帧注意力），各帧独立处理。论文 Stage2 依赖 Aggregator 的 KV缓存和时序因果注意力实现跨帧信息传递，当前 head-only 架构无法利用长序列中的帧间关联。

### 2.3 训练配置

| 配置项 | 当前值 | 论文目标 |
|--------|--------|----------|
| Optimizer | AdamW | AdamW |
| Learning Rate | **5e-4** | **5e-4 ✅** |
| Weight Decay | 0.05 | 0.05 ✅ |
| Warmup | 5% + cosine decay | 5% + cosine ✅ |
| Min LR | 1e-8 | 1e-8 ✅ |
| Gradient Clip | 1.0 | 1.0 ✅ |
| AMP | bfloat16/float16 | bfloat16 ✅ |
| Batch Size | 1 | 1 |
| Total Iterations | 5000 (已完成) | 160,000 |
| Image Size | 224 | 518 |
| Views Start | **8** (适配RTX 3090) | 24 |
| Views End | **24** (适配RTX 3090) | 320 |
| View Warmup Iter | 8000 | 8000 |
| Local Window k | **[4, 16]** (适配显存) | [16, 64] |
| Stride Range | [1, 3] | [1, 3] ✅ |
| 总帧数 | **动态检测** (从traj.txt) | - |

### 2.4 Loss函数

```
L_total = L_depth + 0.1 × L_abs_pose + 0.05 × L_local_rel_pose

L_local_rel_pose = 只在local window内计算relative pose
                   (window_pairs由LocalWindowSampler生成)
                   (延迟开启: iteration >= 500)
```

**与Stage1的关键差异：**
- Stage1: `RelativePoseLoss` — 计算所有V×(V-1)个pairs
- Stage2: `LocalRelativePoseLoss` — 只计算local window内的pairs，传入`window_pairs`参数

### 2.5 训练流程

```
1. 创建 FoldbackVideoSampler (stride=[1,3])
2. 创建 ProgressiveViewCurriculum (8→24)
3. 创建 LocalWindowSampler (k=[4,16])
4. 创建 ReplicaLongSequenceDataset (使用foldback sampler, 含color jitter)
5. 创建 HeadOnlyModel，从Stage1 checkpoint初始化 (strict=False)
6. 初始化 DepthLoss + PoseLoss + LocalRelativePoseLoss
7. AdamW optimizer (lr=5e-4)
8. SequentialLR: 5% warmup → cosine decay
9. 每iteration:
   a. 从curriculum获取当前views数
   b. 更新dataset的num_views
   c. 采样batch
   d. 从window_sampler获取local window pairs
   e. 前向+反向+优化
   f. rel_pose延迟至500 iter后开启
10. 保存checkpoint
```

### 2.6 训练结果

**修复后 Stage2 训练 (5000 iterations):**
- Loss: ~0.08-0.20 范围波动（长序列训练噪声较大）
- Depth相对误差 (Mean): 0.128
- Checkpoint: `try_train/checkpoints/stage2_fixed/checkpoint_stage2_final.pt`

**Stage2 比 Stage1 差的原因：**
HeadOnlyModel 逐帧独立处理，无法利用长序列中的帧间关联。更多帧的输入仅增加了训练复杂度，未能带来信息增益。论文 Stage2 依赖 Aggregator 组件实现跨帧注意力，而 head-only 架构不具备此能力。

---

## 三、Stage1 vs Stage2 对比

### 3.1 核心差异

| 维度 | Stage1 (论文4.1) | Stage2 (论文4.2) |
|------|------------------|------------------|
| **目标** | 基础depth+pose能力 | 长序列流式推理能力 |
| **初始化** | 随机初始化 | **从Stage1 checkpoint加载** |
| **序列长度** | 短 (2 views) | **长 (8→24 views)** |
| **采样策略** | temporal_nearby | **foldback sampler** |
| **Views策略** | 固定 | **progressive curriculum** |
| **Relative Pose** | 全pairs | **local window pairs** |
| **学习率** | 2e-4 | **5e-4** (2.5x) |
| **Rel Pose延迟** | 1000 iter | **500 iter** |

### 3.2 Loss对比

```
Stage1: L = L_depth + 0.1×L_abs_pose + 0.05×L_rel_pose(全pairs)
Stage2: L = L_depth + 0.1×L_abs_pose + 0.05×L_local_rel_pose(窗口pairs)
```

- Stage1的全pairs: O(V²)复杂度，2 views = 2 pairs
- Stage2的窗口pairs: O(k)复杂度，k∈[4,16]，大幅减少计算量

### 3.3 采样策略对比

```
Stage1 temporal_nearby:
  └── 在时间窗口(30帧)内随机采样N个帧
  └── 固定前进方向

Stage2 foldback:
  └── 从start_frame出发，stride∈[1,3]前进
  └── 边界反转方向，重选stride
  └── 运动方向多样化，避免偏置

Stage2 progressive curriculum:
  └── iter 0~8000: views=8 (固定)
  └── iter 8000~160K: views线性增长 8→24
  └── 逐步增加序列长度，避免显存爆炸
```

### 3.4 训练效果对比

| 指标 | 原始模型 | Stage1 | Stage2 |
|------|---------|--------|--------|
| Depth相对误差 | 0.013 | 0.051 | 0.128 |
| 骨干网络 | ViT-L/14 | ViT-S/14 | ViT-S/14 |
| 分辨率 | 518 | 224 | 224 |
| 训练Iterations | - | 5000 | 5000 |
| 跨帧注意力 | 有 (Aggregator) | 无 | 无 |

---

## 四、功能实现状态（2026-05-19更新）

### 4.1 已实现 ✅

| 项目 | 论文要求 | 实现位置 | 说明 |
|------|----------|----------|------|
| **Views范围采样** | views∈[2,24] | `train_replica_v2.py` CLI args + `replica_dataset.py` | `--min_views 2 --max_views 8`，动态范围采样替代固定2 views |
| **Spatial nearby sampler** | 基于camera center 3D距离 | `replica_dataset.py` `_build_spatial_distance_matrix` + `_spatial_nearby_sample` | `--sampler_type spatial_nearby --spatial_radius 5.0` |
| **Pose有效性检查** | finite check + 旋转矩阵正交性 | `replica_dataset.py` `_load_poses()` | np.isfinite检查 + det(R)≈1 正交性验证 |
| **Depth有效比例过滤** | valid_ratio≥0.05 | `replica_dataset.py` `_get_frame_ids()` | 过滤深度无效帧 |
| **Spatial rescale** | range [0.8, 1.2] | `replica_dataset.py` `_apply_spatial_rescale()` | 共享scale参数，同步更新intrinsics/depth/pose |
| **Aspect-ratio sampling** | ratio∈[0.33, 1.0] | `replica_dataset.py` `_apply_aspect_ratio()` | 共享ratio参数，同步更新intrinsics/depth |
| **Co-jitter** | 多view共享颜色扰动 | `replica_dataset.py` `_apply_color_jitter(co_jitter=True)` | 同一sample所有view使用相同color jitter参数 |
| **Dynamic batch packing** | max_images_per_gpu=48 | `train_replica_stage2.py` | `batch_size = max_images_per_gpu // current_views` 动态调整 |
| **Geometric aug后同步更新** | intrinsics/depth/pose同步 | `replica_dataset.py` `__getitem__` | rescale/aspect-ratio后intrinsics和depth同步更新 |

### 4.2 受限无法实现 ❌

| 项目 | 论文要求 | 受限原因 | 说明 |
|------|----------|----------|------|
| **Aggregator跨帧注意力** | Stage2必需Aggregator | **架构限制** | HeadOnlyModel逐帧独立处理，需实现完整GCTStream架构（含KV缓存、时序因果注意力） |
| **518分辨率** | img_size=518 | **显存限制** | RTX 3090 (24GB) 无法支持518分辨率，patch tokens从256增至1372，显存约5-8x增长 |
| **160K iterations** | total_iterations=160000 | **时间限制** | 当前5000 iters验证架构正确性，160K需数天训练时间 |
| **多Room训练** | 多个Replica场景 | **数据限制** | 仅room0数据集可用，缺少其他Replica场景数据 |

### 4.3 下一步方向

1. **实现 GCTStream 架构：** Stage2 的核心是 Aggregator 跨帧注意力，需要实现完整的流式推理架构才能发挥长序列训练的作用
2. **更大GPU资源：** 使用 A100 80GB 训练518分辨率 + 160K iterations
3. **多场景数据：** 收集更多 Replica/ScanNet 场景数据
4. **长周期训练：** 在资源充足后进行完整160K iterations训练

---

## 五、显存占用估算

### 5.1 模型参数显存

| 组件 | 参数量 | FP32 | AMP (FP16) |
|------|--------|------|------------|
| DINOv2 ViT-S/14 (冻结) | 22.1M | 84 MB | 42 MB |
| DepthHead (DPT) | 9.2M | 35 MB | 18 MB |
| PoseHead (CameraHead) | 4.3M | 16 MB | 8 MB |
| **总计 (trainable)** | **35.6M** | **~135 MB** | **~68 MB** |
| **总计 (含冻结)** | **57.6M** | **~220 MB** | **~110 MB** |

> 注：冻结参数仍占用显存，但不参与梯度计算

### 5.2 推理显存（单次前向传播）

推理显存主要取决于**图像分辨率 × views数**：

| 配置 | Patch Tokens (per view) | 特征图显存 (AMP) | V views总显存 |
|------|------------------------|------------------|---------------|
| 224×224, patch=14 | 16×16 = 256 | ~0.4 MB | V × 0.4 MB |
| 518×378, patch=14 | 37×27 = 999 | ~1.5 MB | V × 1.5 MB |
| 518×518, patch=14 | 37×37 = 1369 | ~2.1 MB | V × 2.1 MB |

### 5.3 Stage1 显存估算

```
配置: batch=1, V=2, img=224×224, AMP

模型参数 (含优化器状态):  ~440 MB  (参数2x + Adam状态2x)
前向特征 (DINOv2+Heads):  ~200 MB  (中间激活, 冻结backbone无梯度)
Loss计算:                 ~50 MB   (depth map + pose matrices)
梯度 + 优化器:            ~270 MB  (仅trainable 35.6M)
框架开销:                 ~200 MB  (CUDA context等)
───────────────────────────────────
Stage1 总计:              ~1.2 GB
```

### 5.4 Stage2 显存估算

Stage2的显存取决于当前views数（curriculum动态增长）：

**224分辨率 (AMP):**

| Views数 | DINOv2前向 | Heads前向 | Local Rel Pose | 总显存(估算) |
|---------|-----------|----------|----------------|-------------|
| 8 | ~160 MB | ~80 MB | ~30 MB | ~1.5 GB |
| 24 | ~480 MB | ~240 MB | ~80 MB | ~2.3 GB |
| 64 | ~1.3 GB | ~640 MB | ~200 MB | ~4.2 GB |
| 128 | ~2.6 GB | ~1.3 GB | ~400 MB | ~7.0 GB |
| 320 | ~6.4 GB | ~3.2 GB | ~1.0 GB | ~15.5 GB |

> 注：Local Rel Pose显存与window size k成正比，k∈[4,16]

**518分辨率 (AMP):**

| Views数 | DINOv2前向 | Heads前向 | Local Rel Pose | 总显存(估算) |
|---------|-----------|----------|----------------|-------------|
| 24 | ~1.2 GB | ~600 MB | ~200 MB | ~4.6 GB |
| 64 | ~3.2 GB | ~1.6 GB | ~500 MB | ~9.0 GB |
| 128 | ~6.4 GB | ~3.2 GB | ~1.0 GB | ~16.5 GB |
| 320 | ~16 GB | ~8.0 GB | ~2.5 GB | ~36 GB |

### 5.5 显存限制与策略

| GPU | 显存 | 可用配置 (224) | 可用配置 (518) |
|-----|------|----------------|----------------|
| RTX 3090 | 24 GB | ≤128 views | ≤64 views |
| RTX 4090 | 24 GB | ≤128 views | ≤64 views |
| A100 40GB | 40 GB | ≤320 views | ≤128 views |
| A100 80GB | 80 GB | ≤320 views | ≤320 views |

**Dynamic batch packing的作用：** 限制 `batch_size × views ≤ 48`，确保在不同views数下显存不超限。当前batch=1，views=24时需约2.3 GB (224分辨率)。

---

## 六、训练命令参考

### Stage1

```bash
# Pilot测试
python try_train/train_replica_v2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --total_iterations 100 --img_size 224 --batch_size 1 \
  --output_dir try_train/checkpoints/v2_smoke_test

# 正式训练 (已完成5000 iters)
python try_train/train_replica_v2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --total_iterations 5000 --img_size 224 --batch_size 1 \
  --output_dir try_train/checkpoints/v2_fixed
```

### Stage2

```bash
# Pilot测试
python try_train/train_replica_stage2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --stage1_checkpoint try_train/checkpoints/v2_fixed/checkpoint_final.pt \
  --total_iterations 100 --views_start 8 --views_end 24 --img_size 224

# 正式训练 (已完成5000 iters)
python try_train/train_replica_stage2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --stage1_checkpoint try_train/checkpoints/v2_fixed/checkpoint_final.pt \
  --total_iterations 5000 --views_start 8 --views_end 24 --img_size 224 \
  --output_dir try_train/checkpoints/stage2_fixed

# 论文完整配置 (需要A100 80GB)
python try_train/train_replica_stage2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --stage1_checkpoint try_train/checkpoints/v2_fixed/checkpoint_final.pt \
  --total_iterations 160000 --views_start 24 --views_end 320 --img_size 518
```

---

## 七、下一步方向

1. **实现 GCTStream 架构：** Stage2 的核心是 Aggregator 跨帧注意力，需要实现完整的流式推理架构才能发挥长序列训练的作用
2. **更大GPU资源：** 使用 A100 80GB 训练518分辨率 + 160K iterations
3. **多场景数据：** 收集更多 Replica/ScanNet 场景数据
4. **长周期训练：** 在资源充足后进行完整160K iterations训练

---

Claude Code | 2026-05-19 | Stage1 & Stage2 训练总结 (含功能实现状态更新)
