#!/bin/bash
# GCTStream Stage2 (论文4.2对齐版) 训练脚本
# 在 Stage1 GCT checkpoint 基础上做 Stage2 长序列训练：
#   - foldback video sampler
#   - progressive view curriculum (4 -> 8)
#   - local window relative pose loss (k=[2,4])
#   - lr=1e-4, warmup+cosine
#
# 用法:
#   bash quick-op/train_gct_stage2.sh           # 默认5000 iters
#   bash quick-op/train_gct_stage2.sh 1000      # 自定义iters
#   bash quick-op/train_gct_stage2.sh 5000 try_train/checkpoints/gct_stage2_5k/checkpoint_iter_3000.pt
#   RESUME=auto bash quick-op/train_gct_stage2.sh 5000  # 自动续训最新 checkpoint_iter_*.pt

set -e

# ============== 配置 ==============
DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
STAGE1_CKPT="try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt"
OUTPUT_DIR="try_train/checkpoints/gct_stage2_5k"
TOTAL_ITERS=${1:-5000}
RESUME_CKPT=${2:-${RESUME:-}}
LOG_EVERY=50
SAVE_EVERY=1000

# Stage2 论文4.2参数（基于 518 分辨率 + RTX 3090 显存做了缩放）
VIEWS_START=4
VIEWS_END=8
WARMUP_ITERS=1000
K_MIN=2
K_MAX=4
LR=1e-4
# ================================

RESUME_ARGS=()
if [[ "${RESUME_CKPT}" == "auto" ]]; then
  RESUME_CKPT=$(find "${OUTPUT_DIR}" -maxdepth 1 -name 'checkpoint_iter_*.pt' -printf '%f\n' 2>/dev/null \
    | sort -V \
    | tail -1)
  if [[ -z "${RESUME_CKPT}" ]]; then
    echo "[ERROR] RESUME=auto but no checkpoint_iter_*.pt found in ${OUTPUT_DIR}" >&2
    exit 1
  fi
  RESUME_CKPT="${OUTPUT_DIR}/${RESUME_CKPT}"
fi

if [[ -n "${RESUME_CKPT}" ]]; then
  if [[ ! -f "${RESUME_CKPT}" ]]; then
    echo "[ERROR] Resume checkpoint not found: ${RESUME_CKPT}" >&2
    exit 1
  fi
  RESUME_ARGS=(--resume "${RESUME_CKPT}")
fi

echo "============================================================"
echo "GCTStream Stage2 Head-Only Training (论文 4.2 对齐版)"
echo "============================================================"
echo "  Stage1 Checkpoint: ${STAGE1_CKPT}"
echo "  Output:            ${OUTPUT_DIR}"
echo "  Iterations:        ${TOTAL_ITERS}"
echo "  Views curriculum:  ${VIEWS_START} -> ${VIEWS_END}"
echo "  Local window k:    [${K_MIN}, ${K_MAX}]"
echo "  Base LR:           ${LR}"
if [[ -n "${RESUME_CKPT}" ]]; then
  echo "  Resume:            ${RESUME_CKPT}"
else
  echo "  Resume:            disabled (from Stage1 checkpoint)"
fi
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
  --use_sdpa \
  "${RESUME_ARGS[@]}"
