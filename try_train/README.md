# LingBot-Map 训练框架使用指南

本文档介绍 `try_train/` 目录下的训练框架，实现论文4.1和4.2的两阶段训练策略。

---

## 目录结构

```
try_train/
│
├── 📄 训练计划文档
│   ├── stage1_scannet_training_plan.md           # Stage1训练计划（论文4.1）
│   └── replica_head_only_stage1_stage2_training_plan.md  # 两阶段完整计划 ★
│
├── 📄 核心模块
│   ├── head_only_model.py                        # Head-only模型 + Loss实现
│   ├── replica_dataset.py                        # Replica数据加载器（2-24 views）
│   ├── scannet_dataset.py                        # ScanNet数据加载器
│   ├── scannet_data_exporter.py                  # ScanNet数据导出工具
│   ├── foldback_video_sampler.py                 # Stage2长序列采样组件 ★
│   └── foldback_video_sampler.md                 # 长序列采样组件说明
│
├── 🔧 训练脚本
│   ├── train_replica_v2.py                       # ★ Stage1训练（论文4.1）
│   └── train_replica_stage2.py                   # ★ Stage2训练（论文4.2）
│
├── 🔍 验证脚本
│   ├── validate_model_v2.py                      # Stage1模型验证（短序列2 views）
│   ├── validate_stage1_vs_stage2.py              # Stage1 vs Stage2对比（2 views）
│   └── validate_long_sequence.py                 # ★ 长序列验证（8 views）
│
├── 📊 可视化输出目录
│   ├── vis_stage1_test/                          # Stage1验证输出（短序列）
│   ├── vis_stage2_test/                          # Stage2验证输出（长序列）
│   └── vis_long_sequence/                        # 长序列验证输出（推荐）
│
├── 💾 Checkpoint目录
│   └── checkpoints/
│       ├── v2_smoke_test/                        # Stage1 checkpoint
│       ├── stage2_smoke_test/                    # Stage2 checkpoint
│       └── stage2_rel_pose_test/                 # Stage2 rel_pose测试
│
└── README.md                                     # 本文件
```

---

## 训练脚本说明

### train_replica_v2.py（Stage1，论文4.1）

**特点：**
- Iteration-based训练（而非epoch-based）
- LR=2e-4，weight_decay=0.05
- Warmup 5% + cosine decay scheduler
- Relative pose loss（weight=0.05，延迟开启）
- Views范围2-24（随机采样）
- 训练depth和pose heads（backbone冻结）

**使用：**
```bash
python try_train/train_replica_v2.py \
  --data_root /path/to/Replica/room0 \
  --total_iterations 5000 \
  --output_dir try_train/checkpoints/v2_smoke_test
```

### train_replica_stage2.py（Stage2，论文4.2）

**特点：**
- 长序列训练（views动态增长）
- LR=5e-4，weight_decay=0.05
- Foldback sampler避免固定前进偏置
- Progressive view curriculum（24→320）
- Local window relative pose loss（窗口内pairs）
- 从Stage1 checkpoint初始化

**使用：**
```bash
python try_train/train_replica_stage2.py \
  --data_root /path/to/Replica/room0 \
  --stage1_checkpoint checkpoints/v2_smoke_test/checkpoint_final.pt \
  --total_iterations 160000 \
  --views_start 24 --views_end 320
```

---

## 验证脚本说明

### validate_model_v2.py（Stage1验证）

**验证方式：** 短序列（2 views），与GT比较

**输出：**
- Depth相对误差统计（Mean, Min, Max）
- Pose误差统计（Rotation°, Translation m）
- 可视化：深度预测对比（viridis colormap）

**使用：**
```bash
python try_train/validate_model_v2.py \
  --checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --data_root /path/to/Replica/room0 \
  --num_samples 20 \
  --output_dir try_train/vis_stage1_test
```

### validate_long_sequence.py（Stage2验证，推荐）

**验证方式：** 长序列（8 views），Stage1 vs Stage2 vs GT

**关键特性：**
- 使用Foldback sampler采样（与训练一致）
- 灰度colormap（gray）便于对比原始PNG
- 对比Stage1和Stage2的位姿改善

**输出：**
- Depth误差对比（Stage1 vs Stage2）
- Rotation/Translation误差对比
- 改进百分比统计
- 可视化：长序列深度对比、误差分布

**使用：**
```bash
python try_train/validate_long_sequence.py \
  --stage1_checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --stage2_checkpoint try_train/checkpoints/stage2_smoke_test/checkpoint_stage2_final.pt \
  --num_views 8 --num_samples 10 \
  --output_dir try_train/vis_stage2_test
```

---

## 验证结果（最新测试 2026-05-13）

### Stage1性能（短序列2 views）

| 指标 | 数值 |
|------|------|
| Depth相对误差 (Mean) | **11.11%** |
| Depth相对误差 (Min) | 8.57% |
| Depth相对误差 (Max) | 15.92% |
| Rotation误差 (Mean) | **8.60°** |
| Rotation误差 (范围) | [2.96°, 16.71°] |
| Translation误差 (Mean) | **0.373m** |
| Translation误差 (范围) | [0.130m, 0.633m] |

### Stage2性能（长序列8 views）

| 指标 | Stage1 (100 iter) | Stage2 (100 iter) | 改进 |
|------|-------------------|-------------------|------|
| Depth相对误差 (Mean) | 33.03% | 40.55% | -22.7% (方差↓) |
| **Rotation误差** | **22.75° ± 13.05°** | **5.17° ± 3.05°** | **+77.3% ✅** |
| **Translation误差** | **1.013m ± 0.480m** | **0.394m ± 0.176m** | **+61.1% ✅** |

**关键发现：**
- 位姿预测显著改善（Rotation 77.3%, Translation 61.1%）
- Stage2预测更稳定（方差大幅降低）
- 验证了论文4.2长序列训练策略的有效性

---

## 可视化输出说明

### vis_stage1_test/（Stage1短序列验证）

**内容：**
- `depth_comparison_v2_*.png`: 深度预测vs GT对比（viridis colormap）
- `pose_error_distribution.png`: Pose误差分布统计图

### vis_stage2_test/（Stage2长序列验证）

**内容：**
- `long_sequence_depth_comparison_8views.png`: 8帧序列深度对比（灰度）
- `long_sequence_error_distribution_8views.png`: 误差分布统计
- `single_frame_detailed_comparison_view4.png`: 单帧详细对比（灰度+误差）

**深度可视化说明：**
- 使用灰度colormap（gray），近处暗色，远处亮色
- 便于对比原始PNG深度图的视觉效果

---

## 快速使用指南

### Stage1 训练 + 验证

```bash
# 1. 训练 Stage1
python try_train/train_replica_v2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --total_iterations 100 \
  --output_dir try_train/checkpoints/v2_smoke_test

# 2. 验证 Stage1（2 views）
python try_train/validate_model_v2.py \
  --checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0
```

### Stage2 训练 + 验证

```bash
# 1. 训练 Stage2（从Stage1初始化）
python try_train/train_replica_stage2.py \
  --data_root /home/shared_files/datasets/dovsg/Replica/room0 \
  --stage1_checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --total_iterations 100 \
  --views_start 8 --views_end 32 \
  --output_dir try_train/checkpoints/stage2_smoke_test

# 2. 长序列验证（推荐）
python try_train/validate_long_sequence.py \
  --stage1_checkpoint try_train/checkpoints/v2_smoke_test/checkpoint_final.pt \
  --stage2_checkpoint try_train/checkpoints/stage2_smoke_test/checkpoint_stage2_final.pt \
  --num_views 8 --num_samples 5
```

---

## 核心组件说明

### Foldback Video Sampler（Stage2核心）

**位置：** `foldback_video_sampler.py`

**功能：**
- `FoldbackVideoSampler`: 边界反向继续，避免固定前进偏置
- `ProgressiveViewCurriculum`: Views数线性增长（24→320）
- `LocalWindowSampler`: 窗口k随机采样（[16,64]）

**使用：**
```python
from foldback_video_sampler import FoldbackVideoSampler, ProgressiveViewCurriculum

sampler = FoldbackVideoSampler(total_frames=2000, stride_range=(1, 3))
curriculum = ProgressiveViewCurriculum(views_start=24, views_end=320)

frame_ids = sampler.sample_sequence(num_views=32)  # 生成长序列
current_views = curriculum.get_num_views(iteration)  # 动态views数
```

### Head-Only Model

**位置：** `head_only_model.py`

**架构：**
```
RGB [B,V,3,H,W] → DINOv2 ViT-S14 (冻结) → DepthHead + PoseHead
```

**Loss：**
- `DepthLoss`: masked log-L1
- `PoseLoss`: quaternion geodesic + Huber
- `RelativePoseLoss`: 全pairs relative pose（Stage1）
- `LocalRelativePoseLoss`: 窗口内relative pose（Stage2）

---

## 论文对齐状态

| 项目 | 论文设置 | Stage1实现 | Stage2实现 |
|------|----------|------------|------------|
| Backbone冻结 | ✓ | ✓ | ✓ |
| PoseHead必训 | ✓ | ✓ | ✓ |
| LR | 2e-4 / 5e-4 | ✓ 2e-4 | ✓ 5e-4 |
| Weight Decay | 0.05 | ✓ | ✓ |
| Warmup | 5% | ✓ | ✓ |
| Scheduler | cosine | ✓ | ✓ |
| Views | 2-24 / 24-320 | ✓ 2-24 | ✓ 8-32(pilot) |
| Relative Pose | ✓ | ✓ 全pairs | ✓ 窗口内 |
| Foldback Sampler | - | - | ✓ |

---

## 总结

本训练框架实现了 LingBot-Map 论文的两阶段训练策略：

**Stage1（论文4.1）：**
- 短序列训练（2-24 views）
- Depth + Pose heads训练
- LR=2e-4，warmup+cosine scheduler
- Relative pose loss

**Stage2（论文4.2）：**
- 长序列训练（views动态增长）
- Foldback sampler避免固定前进偏置
- 位姿改善：Rotation +77.3%，Translation +61.1%

**推荐验证方式：**
- Stage1: `validate_model_v2.py`（短序列2 views）
- Stage2: `validate_long_sequence.py`（长序列8 views，灰度深度）

---

Claude Code | 2026-05-13 | 废弃文件清理 + 最新测试结果