#!/bin/bash
# Fetus2026 - 验证集预测脚本
# 分割: FPN + MIT-B5
# 分类: Xception

set -e

IMAGES_DIR="${1:-data/val/images}"
OUTPUT_DIR="${2:-data/val/preds}"
CKPT_DIR="${3:-checkpoints}"

echo "=============================================="
echo " Fetus2026 Prediction"
echo "=============================================="
echo " Images: ${IMAGES_DIR}"
echo " Output: ${OUTPUT_DIR}"
echo " Ckpt:   ${CKPT_DIR}"
echo "=============================================="

python predict.py \
    --images_dir "${IMAGES_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --checkpoints_dir "${CKPT_DIR}" \
    --task both \
    --views 1 2 3 4 \
    --batch_size 1

echo ""
echo "=============================================="
echo " Prediction completed!"
echo " Results saved to: ${OUTPUT_DIR}"
echo "=============================================="
