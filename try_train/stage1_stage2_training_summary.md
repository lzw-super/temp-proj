# Stage1 & Stage2 训练总结

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
  ├── DINOv2 ViT-S/14 (冻结, 22M参数)
  │     └── forward_features → patch_tokens + cls_token
  │
  ├── DepthHead: DPTHead (exp激活, output_dim=2)
  │     └── 4层多尺度特征 [256,512,1024,1024]
  │     └── 输出: depth [B,V,H,W,1], depth_conf [B,V,H,W]
  │
  └── PoseHead: CameraHead (absT_quaR_FoV编码)
        └── 4层transformer trunk, 8 heads
        └── 输出: pose_enc [B,V,9] (center3 + quat4 + fov_offset2)
```

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
| Total Iterations | 100 (pilot) / 5000 | 160,000 |
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

---

## 二、Stage2 训练（论文4.2：Streaming Model Training）

### 2.1 数据预处理

与Stage1相同的基础预处理（depth转换、resize、pad、pose归一化等），**关键差异在采样策略**：

| 差异项 | Stage1 | Stage2 |
|--------|--------|--------|
| 采样器 | temporal_nearby (窗口=30) | **FoldbackVideoSampler** |
| Views数 | 固定2 | **动态24→320 (curriculum)** |
| 序列长度 | 短序列 | **长序列** |
| 相邻帧stride | 固定1 | **随机[1,3]** |

**Foldback Sampler核心逻辑：**
```
1. 从start_frame出发，按stride前进
2. 到达边界时反转方向（foldback），重选stride
3. 避免固定前进偏置，保证训练数据的运动方向多样性
```

**Progressive View Curriculum：**
```
1. 前 warmup_iterations (8000) iter：views固定=views_start (24)
2. 之后线性增长：views_start → views_end (24→320)
3. 公式: views = views_start + (views_end - views_start) × progress
4. progress = (iter - warmup) / (total - warmup), clip到[0,1]
```

**Local Window Sampler：**
```
1. 随机采样窗口大小 k ∈ [k_min, k_max] = [16, 64]
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
| Total Iterations | 100 (pilot) / 5000 | 160,000 |
| Image Size | 224 | 518 |
| Views Start | 8 (pilot) / 24 | 24 |
| Views End | 32 (pilot) / 320 | 320 |
| View Warmup Iter | 8000 | 8000 |
| Local Window k | [16, 64] | [16, 64] ✅ |
| Stride Range | [1, 3] | [1, 3] ✅ |

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
2. 创建 ProgressiveViewCurriculum (24→320)
3. 创建 LocalWindowSampler (k=[16,64])
4. 创建 ReplicaLongSequenceDataset (使用foldback sampler)
5. 创建 HeadOnlyModel，从Stage1 checkpoint初始化
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

---

## 三、Stage1 vs Stage2 对比

### 3.1 核心差异

| 维度 | Stage1 (论文4.1) | Stage2 (论文4.2) |
|------|------------------|------------------|
| **目标** | 基础depth+pose能力 | 长序列流式推理能力 |
| **初始化** | 随机初始化 | **从Stage1 checkpoint加载** |
| **序列长度** | 短 (2 views) | **长 (24→320 views)** |
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
- Stage2的窗口pairs: O(k)复杂度，k∈[16,64]，大幅减少计算量

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
  └── iter 0~8000: views=24 (固定)
  └── iter 8000~160K: views线性增长 24→320
  └── 逐步增加序列长度，避免显存爆炸
```

---

## 四、未实现的训练步骤

### 4.1 高优先级（影响训练效果）

| 项目 | 论文要求 | 当前状态 | 未实现原因 |
|------|----------|----------|------------|
| **Views范围采样** | Stage1: views∈[2,24] | 固定2 views | train_replica_v2.py:183 硬编码`num_views=2`，需改为`min_views=2, max_views=24` |
| **518分辨率** | img_size=518 | img_size=224 | 显存不足：518下DINOv2 ViT-S的patch tokens从256增至1372，显存占用约5-8x增长 |
| **160K iterations** | total_iterations=160000 | pilot=100 | 训练时间长，pilot验证优先 |

### 4.2 中优先级（提升泛化能力）

| 项目 | 论文要求 | 当前状态 | 未实现原因 |
|------|----------|----------|------------|
| **Spatial nearby sampler** | 基于camera center 3D距离采样 | 仅temporal_nearby | 需计算所有帧的camera center距离矩阵，数据预处理增加 |
| **Multi-room训练** | 多个Replica场景 | 仅room0 | 需要遍历多个room目录，DataLoader需支持多场景混合 |
| **Geometric augmentation** | spatial rescale [0.8,1.2] | 无 | 需同步更新intrinsics和pose，实现复杂度高 |

### 4.3 低优先级（论文完整对齐）

| 项目 | 论文要求 | 当前状态 | 未实现原因 |
|------|----------|----------|------------|
| **Aspect-ratio sampling** | ratio∈[0.33,1.0] | 无 | 需实现随机裁剪+resize，同步更新intrinsics |
| **Co-jitter** | 多view共享颜色扰动 | 各view独立jitter | 需在batch级别而非view级别应用相同的color jitter参数 |
| **Dynamic batch packing** | max_images_per_gpu=48 | 无 | 需根据views数动态调整batch_size，保证总images≤48 |
| **Depth有效比例过滤** | valid_ratio≥0.05 | 无 | 数据过滤步骤，影响较小 |
| **Pose有效性检查** | finite check | 仅检查文件存在 | 边界case，影响较小 |

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

> 注：Local Rel Pose显存与window size k成正比，k∈[16,64]

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

**Dynamic batch packing的作用：** 限制 `batch_size × views ≤ 48`，确保在不同views数下显存不超限。当前batch=1，views=320时需约15.5 GB (224分辨率)。

---

## 六、训练命令参考

### Stage1

```bash
# Pilot测试
python try_train/train_replica_v2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --total_iterations 100 --img_size 224 --batch_size 1

# 正式训练
python try_train/train_replica_v2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --total_iterations 160000 --img_size 518 --batch_size 1
```

### Stage2

```bash
# Pilot测试
python try_train/train_replica_stage2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --stage1_checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --total_iterations 100 --views_start 8 --views_end 32 --img_size 224

# 正式训练
python try_train/train_replica_stage2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --stage1_checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --total_iterations 160000 --views_start 24 --views_end 320 --img_size 518
```

---

Claude Code | 2026-05-14 | Stage1 & Stage2 训练总结
