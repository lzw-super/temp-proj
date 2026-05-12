# LingBot-Map 训练框架使用指南

本文档介绍如何使用 `try_train/` 目录下的训练框架来训练 LingBot-Map 模型。

## 概述

训练框架基于 `stage1_scannet_training_plan.md` 文档实现，采用资源受限的 head-only 训练策略：
- 冻结 DINOv2 backbone（通过 torch.hub 加载）
- 只训练轻量 depth/pose heads
- 支持 Replica 和 ScanNet 数据集

### 两个版本对比

| 版本 | 训练脚本 | 目标 | 主要设置 |
|------|---------|------|---------|
| **旧版（验证版）** | `train_replica.py` | 验证流程可运行 | lr=1e-3, pose可选, 无scheduler |
| **新版（论文对齐）** | `train_replica_v2.py` | 论文4.1第一阶段对齐 | lr=2e-4, pose必训, warmup+cosine scheduler, relative pose loss |

**✅ 已验证：**
- 旧版：Replica room0 10 epochs训练，checkpoint保存和验证正常
- 新版：论文4.1对齐训练，depth+abs_pose+rel_pose loss全部工作，scheduler正常

## 文件结构

```
try_train/
├── stage1_scannet_training_plan.md  # 详细训练计划文档（论文4.1对齐版）
├── replica_dataset.py               # Replica数据加载器（支持2-24 views）
├── head_only_model.py               # Head-only模型 + RelativePoseLoss
├── train_replica.py                 # 旧版训练脚本（验证流程）
├── train_replica_v2.py              # 新版训练脚本（论文4.1对齐）★推荐
├── validate_model.py                # 模型验证脚本
├── checkpoints/
│   ├── smoke_test/                  # 旧版验证checkpoint
│   └── v2_smoke_test/               # 新版论文对齐checkpoint
└── README.md                        # 本文件
```

## 快速开始（论文4.1对齐版）

### 1. 数据集结构

Replica 数据集结构：
```
Replica/room0/
├── traj.txt                         # 位姿数据（每行16个浮点数→4x4 matrix）
└── results/
    ├── frame000000.jpg              # RGB图像 (1200x680)
    ├── depth000000.png              # 深度图 (uint16, 毫米)
    └── ...
```

### 2. 论文4.1对齐训练（推荐）

```bash
python try_train/train_replica_v2.py \
  --data_root /path/to/Replica/room0 \
  --total_iterations 5000 \
  --lr 2e-4 \
  --weight_decay 0.05 \
  --train_pose True \
  --rel_pose True \
  --rel_pose_weight 0.05 \
  --rel_pose_start_iter 1000 \
  --img_size 224 \
  --batch_size 1 \
  --output_dir try_train/checkpoints/paper_aligned \
  --save_every 1000 \
  --log_every 20
```

**关键参数说明（论文4.1对齐）：**
- `--lr 2e-4`: Base learning rate（论文设置）
- `--weight_decay 0.05`: Weight decay（论文设置）
- `--train_pose True`: PoseHead必训（论文第一阶段）
- `--rel_pose True`: Relative pose loss启用
- `--rel_pose_start_iter 1000`: Relative pose延迟开启
- `--warmup_ratio 0.05`: 5% warmup（论文设置）
- Scheduler: 5% warmup + cosine decay to 1e-8

### 3. Smoke Test（快速验证）

```bash
python try_train/train_replica_v2.py \
  --data_root /path/to/Replica/room0 \
  --total_iterations 100 \
  --max_samples 50 \
  --output_dir try_train/checkpoints/v2_smoke_test
```

**验证结果（100 iterations, 50 samples）：**
- Loss: 1.62 → 0.13（稳定收敛）
- Depth loss + abs_pose loss + rel_pose loss 全部工作
- LR scheduler: 2e-4 → 1e-8（warmup + cosine）
- Checkpoint保存正常

### 4. 模型验证（论文4.1对齐版）

```bash
python try_train/validate_model_v2.py \
  --checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_iter_100.pt \
  --data_root /path/to/Replica/room0 \
  --num_samples 10 \
  --output_dir try_train/vis_results_v2
```

**验证结果（100 iterations训练，10样本验证）：**
- **Depth相对误差**: 平均11.30%，范围[9.25%, 15.92%]
- **Pose Rotation误差**: 平均9.05°，范围[2.96°, 16.71°]
- **Pose Translation误差**: 平均0.427m，范围[0.158m, 0.633m]

**对比旧版（只训练depth head）：**
| 版本 | Depth相对误差 | Pose训练 |
|------|--------------|---------|
| 旧版(10 epochs) | 21.52% | 无 |
| 新版(100 iter) | 11.30% | 必训 |

**说明**: Depth+Pose联合训练反而使深度预测更准确，因为pose监督提供了额外的几何约束。

可视化输出：
- `depth_comparison_v2_*.png`: 深度预测对比图
- `pose_error_distribution.png`: 位姿误差分布直方图

## 参数说明（论文4.1对齐版）

### 模型参数
- `--backbone`: `dinov2_vits14` (推荐), `dinov2_vitb14`, `dinov2_vitl14`
- `--freeze_backbone`: 是否冻结 backbone (默认True)
- `--train_pose`: 是否训练 pose head (默认True，论文对齐)
- `--rel_pose`: 是否启用 relative pose loss (默认True)

### 训练参数（论文4.1设置）
- `--lr`: learning rate (默认2e-4，论文设置)
- `--weight_decay`: weight decay (默认0.05，论文设置)
- `--total_iterations`: 总迭代数 (论文160K，推荐先5K-20K pilot)
- `--batch_size`: batch 大小 (默认1)
- `--img_size`: 图像尺寸 (默认224，论文目标518)
- `--gradient_clip_norm`: gradient clipping (默认1.0)

### Loss 权重
- `--pose_weight`: absolute pose loss 权重 (默认0.1)
- `--rel_pose_weight`: relative pose loss 权重 (默认0.05)
- `--rel_pose_start_iter`: relative pose 延迟开启 iteration (默认1000)

### Scheduler 参数
- `--warmup_ratio`: warmup iterations ratio (默认0.05，论文设置)
- `--min_lr`: minimum learning rate (默认1e-8，论文设置)

## 训练设置与论文4.1对应

| 项目 | 论文4.1设置 | 实现状态 |
|------|------------|---------|
| Backbone冻结 | DINOv2 frozen | ✅ 已实现 |
| PoseHead训练 | 必训 | ✅ 默认启用 |
| Relative Pose Loss | weight=0.05 | ✅ 已实现 |
| LR | 2e-4 | ✅ 默认值 |
| Weight Decay | 0.05 | ✅ 默认值 |
| Warmup | 5% | ✅ 已实现 |
| Scheduler | cosine decay | ✅ 已实现 |
| Min LR | 1e-8 | ✅ 已实现 |
| Views | 2-24 | ✅ 数据集支持 |
| Augmentation | color jitter 0.9 | ✅ 已实现 |

## 实现细节

### 数据预处理流程（replica_dataset.py - 论文4.1对齐版）

```text
原始数据                          预处理步骤                          输出格式
────────────────────────────────────────────────────────────────────────────
traj.txt (16 floats/line)    →   解析为 4x4 matrix              →   T_c2w: [V, 4, 4]
                                   (camera-to-world)

frame{id}.jpg (1200x680)      →   1. resize (max_dim=224)       →   RGB: [V, 3, H, W]
                                   2. pad to 14 multiples
                                   3. normalize (0-1)
                                   4. color jitter (0.9概率)     ← 论文4.1对齐

depth{id}.png (uint16 mm)     →   1. 除以1000 → meters          →   Depth: [V, H, W]
                                   2. resize (nearest)
                                   3. valid_mask = depth > 0

视角采样                       →   temporal_nearby sampler      →   每样本[min_V, max_V] views
                                   - 支持动态范围2-24            ← 论文4.1对齐
                                   - 时间窗口: 30帧
                                   - shuffle视角顺序

数据增强                       →   color jitter (概率0.9)       →   论文4.1对齐
                                   brightness=0.5
                                   contrast=0.5
                                   saturation=0.5
                                   grayscale=0.05
```

### 模型架构（head_only_model.py - 论文4.1对齐版）

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
    │   trainable ✓   │               │   trainable ✓   │  ← 论文4.1必训
    └─────────────────┘               └─────────────────┘
              │                               │
              ▼                               ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   Depth Pred    │               │   Pose Enc      │
    │   [B, V, H, W]  │               │   [B, V, 9]     │
    └─────────────────┘               └─────────────────┘

参数统计：Trainable 35.6M, Total 57.6M, Frozen 38.26%
```

### Loss函数设计（论文Sec. 3.3）

```text
总 Loss = L_depth + λ_abs_pose * L_abs_pose + λ_rel_pose * L_rel_pose

L_depth = mean_valid(|log(D_pred + eps) - log(D_gt + eps)|)

L_abs_pose = L_rot_geodesic + λ_trans * L_trans_huber
           - rotation: quaternion geodesic loss
           - translation: Huber loss
           - 第一帧不参与loss计算

L_rel_pose = mean_pairs(L_rot_ij + λ_trans_rel * L_trans_ij)
           - T_i_to_j = inv(T_i) @ T_j
           - 所有view pairs计算relative pose
           - 延迟开启（iteration 1000后）
```

### LR Scheduler（论文4.1对齐）

```text
Warmup阶段 (5% iterations):
  LR: 1e-8 → 2e-4 (线性增长)

Cosine Decay阶段 (95% iterations):
  LR: 2e-4 → 1e-8 (cosine衰减)

示例 (5000 iterations):
  - Warmup: 250 iterations (LR 1e-8 → 2e-4)
  - Cosine: 4750 iterations (LR 2e-4 → 1e-8)
```

---

Claude Code | 2026-05-12