#!/bin/bash
# GCTStream Stage2 (论文4.2对齐版) 训练脚本
# 在 Stage1 GCT checkpoint 基础上做 Stage2 长序列训练：
#   - foldback video sampler
#   - progressive view curriculum (4 -> 8)
#   - local window relative pose loss (k=[2,4])
#   - lr=5e-4, warmup+cosine
#
# 用法:
#   bash quick-op/train_gct_stage2.sh           # 默认5000 iters
#   bash quick-op/train_gct_stage2.sh 1000      # 自定义iters

set -e

# ============== 配置 ==============
DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
STAGE1_CKPT="try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt"
OUTPUT_DIR="try_train/checkpoints/gct_stage2_5k"
TOTAL_ITERS=${1:-5000}
LOG_EVERY=50
SAVE_EVERY=1000

# Stage2 论文4.2参数（基于 518 分辨率 + RTX 3090 显存做了缩放）
VIEWS_START=4
VIEWS_END=8
WARMUP_ITERS=1000
K_MIN=2
K_MAX=4
LR=5e-4
# ================================

echo "============================================================"
echo "GCTStream Stage2 Head-Only Training (论文 4.2 对齐版)"
echo "============================================================"
echo "  Stage1 Checkpoint: ${STAGE1_CKPT}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Iterations:        ${TOTAL_ITERS}"
echo "  Views curriculum:  ${VIEWS_START} -> ${VIEWS_END}"
echo "  Local window k:    [${K_MIN}, ${K_MAX}]"
echo "  Base LR:           ${LR}"
echo "============================================================"

conda run -n lingbot-map --no-capture-output python try_train/train_replica_gct_stage2.py \
  --data_root "${DATA_ROOT}" \
  --stage1_checkpoint "${STAGE1_CKPT}" \
  --output_dir "${OUTPUT_DIR}" \
  --total_iterations "${TOTAL_ITERS}" \
  --log_every "${LOG_EVERY}" \
  --save_every "${SAVE_EVERY}" \
  --views_start "${VIEWS_START}" \
  --views_end "${VIEWS_END}" \
  --warmup_iterations "${WARMUP_ITERS}" \
  --k_min "${K_MIN}" \
  --k_max "${K_MAX}" \
  --lr "${LR}" \
  --use_sdpa
