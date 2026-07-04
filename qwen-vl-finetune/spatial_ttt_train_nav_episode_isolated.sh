#!/bin/bash
# Spatial-TTT training — ACTION-ISOLATED episode packing.
# Actions are removed from the token stream (no leakage into history);
# each step is supervised via a single-token readout label
# (see create_data/build_episode_isolated.py).

export TORCHCODEC_FFMPEG_LOG_LEVEL=QUIET

# ======================
# Logging
# ======================
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/spatial_ttt_isolated_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging to: $LOG_FILE"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "PWD: $(pwd)"

# ======================
# Distributed
# ======================
NPROC_PER_NODE=4
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)

# ======================
# Paths (set these for your environment)
# ======================
MODEL_PATH="/mnt/data/vmo-ai-task/anhdh35/Spatial-TTT/qwen-vl-finetune/checkpoints/Qwen3-VL-2B-Instruct"
OUTPUT_DIR="./checkpoints/spatial_ttt_nav_train_episode_isolated"
CACHE_DIR="/mnt/data/vmo-ai-task/anhdh35/.cache"

# ======================
# Data (action-isolated episodes, built by create_data/build_episode_isolated.py)
# ======================
DATASET="train_r2r_rxr_episode_isolated"
VIDEO_MAX_FRAMES=128
RESIZE_HEIGHT=352
RESIZE_WIDTH=480

# ======================
# Training
# ======================
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=1e-6
NUM_EPOCHS=1
MAX_LENGTH=65536

# STOP is ~1.2% of action labels; up-weight it so the policy learns to stop.
# Set to 1.0 to disable.
STOP_LOSS_WEIGHT=5.0

# ======================
# TTT / LaCT
# ======================
LACT_CHUNK_SIZE=1024
WINDOW_SIZE=2048

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
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
    --stop_loss_weight $STOP_LOSS_WEIGHT \
    --stop_action_token "stop" \
    --seed 42
