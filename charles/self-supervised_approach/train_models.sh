#!/bin/bash

# params
SIMCLR_BATCH_SIZE=256
BYOL_BATCH_SIZE=128
NUM_EPOCHS=100
LEARNING_RATE=3e-4
TEMPERATURE=0.5
PROJECTION_DIM=128
MOVING_AVG_DECAY=0.99
PATIENCE=5

# log directories
LOG_DIR="/data/cs7n21/charles/logs"
mkdir -p $LOG_DIR
SIMCLR_LOG="$LOG_DIR/simclr_training.log"
BYOL_LOG="$LOG_DIR/byol_training.log"

echo "Starting SimCLR training..." | tee -a $SIMCLR_LOG
python3 train.py --model_type SimCLR \
    --batch_size $SIMCLR_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --temperature $TEMPERATURE \
    --projection_dim $PROJECTION_DIM \
    --patience $PATIENCE | tee -a $SIMCLR_LOG

echo "SimCLR training finished. Now starting BYOL training..." | tee -a $BYOL_LOG
python3 train.py --model_type BYOL \
    --batch_size $BYOL_BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --projection_dim $PROJECTION_DIM \
    --moving_avg_decay $MOVING_AVG_DECAY \
    --patience $PATIENCE | tee -a $BYOL_LOG

echo "Training complete for both SimCLR and BYOL."
