#!/bin/bash
# 3-way 验证: Original / Stage1 (global attn) / Stage2 (GCA streaming)
#
# 调用 try_train/validate_gct_stage2.py 同时加载三个模型并对比性能 + 可视化。
# Stage2 评估时使用 GCA streaming forward（与训练设置一致）。
#
# 用法:
#   bash quick-op/validate_gct_stage2.sh           # 默认 20 个样本
#   bash quick-op/validate_gct_stage2.sh 50        # 自定义样本数

set -e

DATA_ROOT="/home/shared_files/datasets/dovsg/Replica/room0"
ORIGINAL_MODEL="/home/shared_files/model_weights/linbo_map/lingbot-map.pt"
STAGE1_CKPT="try_train/checkpoints/gct_stage1_5k/checkpoint_final.pt"
STAGE2_CKPT="try_train/checkpoints/gct_stage2_5k/checkpoint_final.pt"
NUM_SAMPLES=${1:-20}
NUM_VIS=3
NUM_VIEWS=4
STAGE2_K=4

echo "============================================================"
echo "GCTStream 3-way Validation (Original / Stage1 / Stage2)"
echo "============================================================"
echo "  Original:  ${ORIGINAL_MODEL}"
echo "  Stage1:    ${STAGE1_CKPT}"
echo "  Stage2:    ${STAGE2_CKPT}"
echo "  Samples:   ${NUM_SAMPLES}  Vis: ${NUM_VIS}  Views/sample: ${NUM_VIEWS}"
echo "  Stage2 forward: GCA streaming, k=${STAGE2_K}"
echo "============================================================"

conda run -n lingbot-map --no-capture-output python try_train/validate_gct_stage2.py \
  --data_root "${DATA_ROOT}" \
  --original_model "${ORIGINAL_MODEL}" \
  --stage1_checkpoint "${STAGE1_CKPT}" \
  --stage2_checkpoint "${STAGE2_CKPT}" \
  --num_samples "${NUM_SAMPLES}" \
  --num_vis "${NUM_VIS}" \
  --num_views "${NUM_VIEWS}" \
  --stage2_sliding_window "${STAGE2_K}"
