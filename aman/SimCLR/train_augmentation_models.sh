#!/bin/bash

BASE_DIR="C:\Users\aman\Desktop\MPhys Data\Data\experiments"

NUM_MODELS=10
NUM_EPOCHS=50
LEARNING_RATE=0.001
BATCH_SIZE=256

NOISE_MAX=0.2
NOISE_INCREMENT=$(echo "$NOISE_MAX / $NUM_MODELS" | bc -l)

SCALE_MIN_START=0.8
SCALE_MIN_MAX=1.0
SCALE_MIN_INCREMENT=$(echo "($SCALE_MIN_MAX - $SCALE_MIN_START) / $NUM_MODELS" | bc -l)

SCALE_MAX_START=1.0
SCALE_MAX_MAX=1.2
SCALE_MAX_INCREMENT=$(echo "($SCALE_MAX_MAX - $SCALE_MAX_START) / $NUM_MODELS" | bc -l)

MAX_SHIFT_MAX=5
MAX_SHIFT_INCREMENT=$(echo "$MAX_SHIFT_MAX / $NUM_MODELS" | bc -l)

SWAP_PROB_MAX=0.5
SWAP_PROB_INCREMENT=$(echo "$SWAP_PROB_MAX / $NUM_MODELS" | bc -l)

MASK_PROB_MAX=0.5
MASK_PROB_INCREMENT=$(echo "$MASK_PROB_MAX / $NUM_MODELS" | bc -l)

for MODEL_IDX in $(seq 1 $NUM_MODELS); do
  NOISE=$(echo "$NOISE_INCREMENT * $MODEL_IDX" | bc -l)
  SCALE_MIN=$(echo "$SCALE_MIN_START + $SCALE_MIN_INCREMENT * $MODEL_IDX" | bc -l)
  SCALE_MAX=$(echo "$SCALE_MAX_START + $SCALE_MAX_INCREMENT * $MODEL_IDX" | bc -l)
  MAX_SHIFT=$(echo "$MAX_SHIFT_INCREMENT * $MODEL_IDX" | bc -l)
  SWAP_PROB=$(echo "$SWAP_PROB_INCREMENT * $MODEL_IDX" | bc -l)
  MASK_PROB=$(echo "$MASK_PROB_INCREMENT * $MODEL_IDX" | bc -l)

  MAX_SHIFT=$(printf "%.0f" $MAX_SHIFT)

  EXPERIMENT_NAME="SimCLR_Model_${MODEL_IDX}_noise${NOISE}_scale${SCALE_MIN}-${SCALE_MAX}_shift${MAX_SHIFT}_swap${SWAP_PROB}_mask${MASK_PROB}_$(date +%Y%m%d_%H%M%S)"
  EXPERIMENT_DIR="${BASE_DIR}/${EXPERIMENT_NAME}"

  python self-supervised_approach/train.py \
    --model_type "SimCLR" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --checkpoint_dir $EXPERIMENT_DIR \
    --noise_strength $NOISE \
    --scale_min $SCALE_MIN \
    --scale_max $SCALE_MAX \
    --max_shift $MAX_SHIFT \
    --swap_prob $SWAP_PROB \
    --mask_prob $MASK_PROB

  sleep 5
done
