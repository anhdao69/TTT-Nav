#!/bin/bash
# Spatial-TTT ACTION-ISOLATED training — ROUND 2 (continue training).
#
# Why: teacher-forced diagnosis of the round-1 checkpoint showed turn recall
# 0.23 / stop recall ~0 under argmax (forward 0.98) -> closed-loop collapses
# to 95% forward. Fix: class-balanced action loss + fresh cosine cycle at a
# higher LR, RESUMING from the round-1 isolated checkpoint (the readout
# scaffold prior is already unlearned, so no CE~24 warm-up phase).

export TORCHCODEC_FFMPEG_LOG_LEVEL=QUIET

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spatial_ttt_isolated_r2_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to: $LOG_FILE"; echo "Started at: $(date)"; echo "Host: $(hostname)"

NPROC_PER_NODE=4
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

# ======================
# Paths — MODEL_PATH is the ROUND-1 ISOLATED CHECKPOINT (config.json +
# model.safetensors incl. all LaCT keys; train_spatial_ttt.py loads the full
# state dict from MODEL_PATH/model.safetensors after wrapping).
# ======================
MODEL_PATH="./checkpoints/spatial_ttt_nav_train_episode_isolated"
OUTPUT_DIR="./checkpoints/spatial_ttt_nav_train_episode_isolated_r2"
CACHE_DIR="/mnt/data/vmo-ai-task/anhdh35/.cache"

DATASET="train_r2r_rxr_episode_isolated"
VIDEO_MAX_FRAMES=128
RESIZE_HEIGHT=352
RESIZE_WIDTH=480

# ======================
# Training — fresh cosine cycle, higher LR than round 1 (round-1 plateaued at
# ~0.95/decision while LR decayed to 1e-7).
# ======================
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-6
NUM_EPOCHS=2
MAX_LENGTH=65536

# Class-balanced action loss (targets the measured failure: turn recall 0.23,
# stop recall ~0 under argmax). Takes precedence over --stop_loss_weight.
ACTION_LOSS_WEIGHTS="forward:1.0,left:2.5,right:2.5,stop:6.0"

LACT_CHUNK_SIZE=1024
WINDOW_SIZE=2048

export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

torchrun --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    qwenvl/train/train_spatial_ttt.py \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --dataset_use $DATASET \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --model_max_length $MAX_LENGTH \
    --bf16 True \
    --gradient_checkpointing True \
    --optim adamw_torch \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --min_lr_rate 0.1 \
    --weight_decay 0.01 \
    --dataloader_num_workers 0 \
    --video_fps 30 \
    --video_max_frames $VIDEO_MAX_FRAMES \
    --resize_height $RESIZE_HEIGHT \
    --resize_width $RESIZE_WIDTH \
    --video_min_frames 1 \
    --video_min_pixels 0 \
    --video_max_pixels 0 \
    --max_pixels 168960 \
    --min_pixels 50176 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 3 \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm True \
    --data_flatten False \
    --data_packing False \
    --deepspeed "scripts/zero2.json" \
    --ddp_find_unused_parameters True \
    --lora_enable False \
    --lact_enable True \
    --num_lact_heads 4 \
    --lact_chunk_size $LACT_CHUNK_SIZE \
    --window_size $WINDOW_SIZE \
    --window_decay False \
    --use_muon True \
    --use_momentum True \
    --use_conv_layer False \
    --w0_w2_low_rank 0 \
    --learnable_ttt_scale True \
    --lact_lr 1e-5 \
    --lact_layers "0/1/2/4/5/6/8/9/10/12/13/14/16/17/18/20/21/22/24/25/26" \
    --use_fused_kernel False \
    --action_loss_weights "$ACTION_LOSS_WEIGHTS" \
    --seed 42
