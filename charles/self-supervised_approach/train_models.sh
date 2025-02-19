#!/bin/bash

# -----------------------------------
# workspace directories for cloud GPU
# -----------------------------------
TRAIN_DATA="/root/data/train.h5"
VAL_DATA="/root/data/val.h5"

SIMCLR_WEIGHTS_DIR="/root/experiments/SimCLR"
SIMCLR_LOG_DIR="/root/experiments/SimCLR"

BYOL_WEIGHTS_DIR="/root/experiments/BYOL"
BYOL_LOG_DIR="/root/experiments/BYOL"

# -------------------------------
# shared parameters
# -------------------------------
EPOCHS=100
LR=3e-4
BATCH_SIMCLR=256
BATCH_BYOL=128
PROJ_DIM=128
TEMPERATURE=0.5
MOVING_AVG_DECAY=0.99
PATIENCE=5

# ----------------------------------
# augmentation params
# ----------------------------------
NOISE_STRENGTH=0.02
SCALE_MIN=0.75
SCALE_MAX=1.25
TIME_CROP_MIN=0.6
TIME_CROP_MAX=1.0
MIN_VALID_RATIO=0.2

SWAP_INTENSITY=0.1
MASK_INTENSITY=0.1

TIME_CROP_PROB=1.0
ADD_NOISE_PROB=1.0
SCALE_DATA_PROB=1.0
SWAP_ADJ_PROB=0.4
MASK_DATA_PROB=0.0

# ------------------------------
# SIMCLR TRAINING
# -----------------------------

python3 train.py \
    --model_type SimCLR \
    --train_h5_file_path "$TRAIN_DATA" \
    --val_h5_file_path "$VAL_DATA" \
    --weights_dir "$SIMCLR_WEIGHTS_DIR" \
    --log_dir "$SIMCLR_LOG_DIR" \
    --batch_size "$BATCH_SIMCLR" \
    --num_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --projection_dim "$PROJ_DIM" \
    --temperature "$TEMPERATURE" \
    --patience "$PATIENCE" \
    --noise_strength "$NOISE_STRENGTH" \
    --scale_min "$SCALE_MIN" \
    --scale_max "$SCALE_MAX" \
    --time_crop_min "$TIME_CROP_MIN" \
    --time_crop_max "$TIME_CROP_MAX" \
    --min_valid_ratio "$MIN_VALID_RATIO" \
    --swap_intensity "$SWAP_INTENSITY" \
    --mask_intensity "$MASK_INTENSITY" \
    --time_crop_prob "$TIME_CROP_PROB" \
    --add_noise_prob "$ADD_NOISE_PROB" \
    --scale_data_prob "$SCALE_DATA_PROB" \
    --swap_adj_prob "$SWAP_ADJ_PROB" \
    --mask_data_prob "$MASK_DATA_PROB"

echo "SimCLR training finished."

# ----------------------
# BYOL TRAINING
# ----------------------
echo "======================"
echo "Starting BYOL training"
echo "======================"

python3 train.py \
    --model_type BYOL \
    --train_h5_file_path "$TRAIN_DATA" \
    --val_h5_file_path "$VAL_DATA" \
    --weights_dir "$BYOL_WEIGHTS_DIR" \
    --log_dir "$BYOL_LOG_DIR" \
    --batch_size "$BATCH_BYOL" \
    --num_epochs "$EPOCHS" \
    --learning_rate "$LR" \
    --projection_dim "$PROJ_DIM" \
    --moving_avg_decay "$MOVING_AVG_DECAY" \
    --patience "$PATIENCE" \
    --noise_strength "$NOISE_STRENGTH" \
    --scale_min "$SCALE_MIN" \
    --scale_max "$SCALE_MAX" \
    --time_crop_min "$TIME_CROP_MIN" \
    --time_crop_max "$TIME_CROP_MAX" \
    --min_valid_ratio "$MIN_VALID_RATIO" \
    --swap_intensity "$SWAP_INTENSITY" \
    --mask_intensity "$MASK_INTENSITY" \
    --time_crop_prob "$TIME_CROP_PROB" \
    --add_noise_prob "$ADD_NOISE_PROB" \
    --scale_data_prob "$SCALE_DATA_PROB" \
    --swap_adj_prob "$SWAP_ADJ_PROB" \
    --mask_data_prob "$MASK_DATA_PROB"

echo "BYOL training finished."

echo "Training complete for both SimCLR and BYOL."

