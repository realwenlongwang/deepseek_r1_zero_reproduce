#!/bin/bash

# Example script to resume training from the latest checkpoint
# This demonstrates how to use the new resume functionality
set -e  # Exit on any error

echo "Setting up environment variables for GRPO training..."

# Distributed training environment variables
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12356

# CUDA/GPU environment variables
export CUDA_VISIBLE_DEVICES=1  # Adjust as needed for your GPU setup

# Python/UV environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# TRL/Transformers environment variables
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings

# Optional: Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

export VLLM_LOGGING_LEVEL=WARNING

echo "Environment variables set:"
echo "  RANK=$RANK"
echo "  WORLD_SIZE=$WORLD_SIZE" 
echo "  LOCAL_RANK=$LOCAL_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM"
echo ""
# Set the checkpoint path (adjust to your specific checkpoint)
CHECKPOINT_PATH="/home/wenlong.wang/src/deepseek_r1_zero_reproduce/saved_models/qwen2.5-3b_format-equation_20250720_123256/checkpoint-15400"

echo "🔄 Resuming GRPO training from checkpoint: $CHECKPOINT_PATH"
echo "📝 This will create a new wandb run named: qwen2.5-3b_format-equation_resumed_from_vfq6army_step15400"
echo ""

# Resume training using the new system.resume_from_checkpoint parameter
uv run train_grpo.py \
    --system.resume_from_checkpoint "$CHECKPOINT_PATH" \
    --monitoring.wandb.enabled true

echo ""
echo "✅ Resume command completed!"