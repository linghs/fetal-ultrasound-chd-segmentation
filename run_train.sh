#!/bin/bash
# Fetus2026 - 训练脚本
# 分割: FPN + MIT-B5
# 分类: Xception

set -e

DATA_DIR="/root/autodl-tmp/_FETUS_data/train"
IMAGES_DIR="${DATA_DIR}/images"
LABELS_DIR="${DATA_DIR}/labels"
CKPT_DIR="checkpoints"
EPOCHS=100

echo "=============================================="
echo " Fetus2026 Training Pipeline"
echo "=============================================="
echo " Data:   ${DATA_DIR}"
echo " Ckpt:   ${CKPT_DIR}"
echo " Epochs: ${EPOCHS}"
echo "=============================================="

# ---- 1. 分割训练 (FPN + MIT-B5) ----
echo ""
echo "[1/2] Segmentation Training (FPN + MIT-B5)"
echo "----------------------------------------------"
python train_segmentation.py \
    --images_dir "${IMAGES_DIR}" \
    --labels_dir "${LABELS_DIR}" \
    --save_dir "${CKPT_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size 8 \
    --lr 5e-5 \
    --views 1 2 3 4

# ---- 2. 分类训练 (Xception) ----
echo ""
echo "[2/2] Classification Training (Xception)"
echo "----------------------------------------------"
python train_classification.py \
    --images_dir "${IMAGES_DIR}" \
    --labels_dir "${LABELS_DIR}" \
    --save_dir "${CKPT_DIR}" \
    --epochs ${EPOCHS} \
    --batch_size 16 \
    --lr 1e-4 \
    --views 1 2 3 4

echo ""
echo "=============================================="
echo " All training completed!"
echo " Checkpoints saved to: ${CKPT_DIR}"
echo "=============================================="
