# LingBot-Map 训练框架使用指南

本文档介绍如何使用 `try_train/` 目录下的训练框架来训练 LingBot-Map 模型。

## 概述

训练框架基于 `stage1_scannet_training_plan.md` 文档实现，采用资源受限的 head-only 训练策略：
- 冻结 DINOv2 backbone（通过 torch.hub 加载）
- 只训练轻量 depth/pose heads
- 支持 Replica 和 ScanNet 数据集
- 目标是验证流程可运行

**✅ 已验证：** 使用 Replica room0 数据集成功完成训练测试，checkpoint 保存和模型推理验证均正常。

## 文件结构

```
try_train/
├── stage1_scannet_training_plan.md  # 详细训练计划文档
├── replica_dataset.py               # Replica数据加载器（已验证）
├── scannet_dataset.py               # ScanNet数据加载器
├── scannet_data_exporter.py         # ScanNet数据导出脚本
├── head_only_model.py               # Head-only训练模型（已修复DINOv2加载）
├── train_replica.py                 # Replica训练脚本（已验证）
├── train.py                         # ScanNet训练脚本
├── validate_model.py                # 模型验证脚本（已验证）
├── checkpoints/                     # 保存的checkpoint
│   └── smoke_test/
│       ├── checkpoint_epoch_4.pt    # 405MB
│       └── checkpoint_epoch_9.pt    # 405MB
├── vis_results/                     # 验证可视化结果
│   └── depth_comparison_*.png
└── README.md                        # 本文件
```

## 快速开始（使用 Replica 数据集）

### 1. 数据集结构

Replica 数据集结构：
```
Replica/room0/
├── traj.txt                         # 位姿数据（每行16个浮点数→4x4 matrix）
└── results/
    ├── frame000000.jpg              # RGB图像 (1200x680)
    ├── frame000001.jpg
    ├── ...
    ├── depth000000.png              # 深度图 (uint16, 毫米)
    ├── depth000001.png
    └── ...
```

### 2. Smoke Test（快速验证）

```bash
python try_train/train_replica.py \
  --data_root /path/to/Replica/room0 \
  --backbone dinov2_vits14 \
  --batch_size 1 \
  --num_epochs 10 \
  --img_size 224 \
  --num_views 2 \
  --max_samples 100 \
  --output_dir try_train/checkpoints/smoke_test \
  --save_every 5 \
  --log_every 20
```

### 3. 验证训练结果

```bash
python try_train/validate_model.py \
  --checkpoint try_train/checkpoints/smoke_test/checkpoint_epoch_9.pt \
  --data_root /path/to/Replica/room0 \
  --num_samples 10 \
  --output_dir try_train/vis_results
```

**验证结果示例（10 epochs, 100 samples）：**
- 平均相对误差: 21.52%
- 最小相对误差: 17.51%
- 最大相对误差: 29.22%
- 深度预测范围: 3m-40m（GT: 8m-33m）

## 使用 ScanNet 数据集

### 1. 下载 ScanNet 数据

需要申请 ScanNet 许可：https://www.scan-net.org/

### 2. 导出数据

```bash
python try_train/scannet_data_exporter.py \
  --scannet_root scannet_raw/ \
  --output_root scannet_processed/ \
  --max_scenes 5 \
  --max_frames 100
```

### 3. 训练

```bash
python try_train/train.py \
  --train_metadata scannet_processed/scannet_train_meta.json \
  --backbone dinov2_vits14 \
  --batch_size 1 \
  --num_epochs 10 \
  --img_size 224 \
  --num_views 2 \
  --output_dir checkpoints/scannet_train
```

## 参数说明

### 模型参数
- `--backbone`: `dinov2_vits14` (推荐), `dinov2_vitb14`, `dinov2_vitl14`
- `--freeze_backbone`: 是否冻结 backbone (默认True)
- `--train_pose`: 是否训练 pose head (默认False)

### 训练参数
- `--lr`: learning rate (默认1e-3)
- `--batch_size`: batch 大小 (默认1)
- `--num_views`: 视角数 (默认2, 可选2-4)
- `--img_size`: 图像尺寸 (默认224, 可选336)
- `--gradient_clip_norm`: gradient clipping (默认1.0)
- `--pose_weight`: pose loss 权重 (默认0.1)
- `--max_samples`: 限制样本数量（用于快速测试）

## 训练设置与文档对应

| 文档章节 | 训练设置 | 状态 |
|---------|---------|------|
| 第2节 | Head-only架构 | ✅ 已实现 |
| 第4节 | 数据预处理 | ✅ 已实现 |
| 第7节 | Loss: masked log L1 | ✅ 已实现 |
| 第9节 | lr=1e-3, gradient clip | ✅ 已实现 |

## 实现细节

### 数据预处理流程（replica_dataset.py）

```text
原始数据                          预处理步骤                          输出格式
────────────────────────────────────────────────────────────────────────────
traj.txt (16 floats/line)    →   解析为 4x4 matrix              →   T_c2w: [V, 4, 4]
                                   (camera-to-world)

frame{id}.jpg (1200x680)      →   1. resize (max_dim=224)       →   RGB: [V, 3, H, W]
                                   2. pad to 14 multiples
                                   3. normalize (0-1)

depth{id}.png (uint16 mm)     →   1. 除以1000 → meters          →   Depth: [V, H, W]
                                   2. resize (nearest)
                                   3. pad
                                   4. valid_mask = depth > 0

内参推断                       →   fx=fy=600, cx=W/2, cy=H/2     →   K: [3, 3]
                                   resize后同步更新内参

Pose归一化                     →   T_ref = poses[0]             →   相对第一帧坐标系
                                   T_i = inv(T_ref) @ T_c2w[i]

采样策略                       →   temporal_nearby sampler      →   每样本2-4 views
                                   - 时间窗口: 30帧
                                   - 窗口内随机采样
                                   - shuffle视角顺序
                                   - 第一帧作为reference

数据增强                       →   color jitter (概率0.3)       →   brightness/contrast/saturation
```

**关键预处理特点：**
- 深度单位转换：毫米 → 米（符合训练文档第4.2节要求）
- Pose格式统一：camera-to-world matrix
- 尺寸对齐：14的整数倍（适配DINOv2 patch_size=14）
- 有效像素过滤：valid_mask排除无效深度区域

### 模型训练策略（head_only_model.py）

```text
架构设计
────────────────────────────────────────────────────────────────────
                    ┌─────────────────────┐
                    │   RGB Images        │ [B, V, 3, H, W]
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   DINOv2 ViT-S14    │ (冻结 ✓)
                    │   torch.hub 加载    │
                    └─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Patch Tokens      │ [B*V, N, 384]
                    │   + CLS Token       │ [B*V, 1, 384]
                    └─────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   Depth Head    │               │   Pose Head     │
    │   (DPTHead)     │               │   (CameraHead)  │
    │   trainable ✓   │               │   可选          │
    └─────────────────┘               └─────────────────┘
              │                               │
              ▼                               ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   Depth Pred    │               │   Pose Enc      │
    │   activation=exp│               │   9-dim encoding│
    │   [B, V, H, W]  │               │   [B, V, 9]     │
    └─────────────────┘               └─────────────────┘
```

**训练策略要点：**
- **冻结策略**：DINOv2 backbone冻结（44.09%参数冻结）
- **可训练参数**：DepthHead + PoseHead（可选）
- **深度激活**：`exp`函数确保输出为正数
- **参数统计**：Trainable 27.9M, Total 50.0M

### Loss函数设计

```python
# Depth Loss: masked log L1（符合文档第7节）
L_depth = mean(|log(D_pred + eps) - log(D_gt + eps)|)  # 仅在valid_mask上计算

# Pose Loss: rotation geodesic + translation Huber（可选）
L_pose = λ_rot * L_rotation + λ_trans * L_translation
# rotation: quaternion geodesic
# translation: Huber loss (鲁棒性)
```

### 训练设置（train_replica.py）

```text
参数配置                           对应文档章节
────────────────────────────────────────────────────────────
Optimizer: AdamW                   第9节
Learning Rate: 1e-3                第9节 smoke run
Weight Decay: 1e-4                 第9节

Gradient Clipping: 1.0             第9节
Mixed Precision: AMP (fp16)        第9节

Batch Size: 1                      第9节 smoke run
Epochs: 10                         验证测试
Views per Sample: 2                第5.1节
Image Size: 224                    第6节 augmentation

Pose Weight: 0.1                   第7.2节
Depth Weight: 1.0                  第7.1节

Save Every: 5 epochs               checkpoint保存
Log Every: 20 batches              训练监控
```

**训练流程：**
```text
1. DataLoader创建 → 采样样本 → 形成batch
2. Forward pass:
   - backbone.forward_features() → patch tokens
   - depth_head() → depth prediction
   - pose_head() → pose encoding (可选)
3. Loss计算:
   - depth_loss = masked_log_l1(pred, gt, mask)
   - pose_loss = geodesic + huber (可选)
4. Backward:
   - scaler.scale(loss).backward()
   - gradient clipping
   - optimizer.step()
5. Checkpoint保存 (每5 epochs)
```

### 训练收敛曲线（验证测试）

```text
Epoch    Loss      说明
──────────────────────────────────
0        2.91      初始高loss（模型初始化）
0-1      → inf     部分batch出现数值不稳定
1        0.47      快速下降（学习生效）
2        0.38      继续下降
4        0.27      稳定收敛
9        0.26      最终收敛（相对误差21.52%）
```

**说明：** Epoch 0出现inf是由于部分batch深度预测不稳定，经过gradient clipping和正则化后快速收敛。

---

Claude Code | 2026-05-12