#!/bin/bash

# DeepSeek R1 Zero GRPO Training Environment Setup and Execution Script
# This script exports all necessary environment variables and runs training in one command

set -e  # Exit on any error

echo "Setting up environment variables for GRPO training..."

# Distributed training environment variables
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=12354

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

# Run training with UV
uv run train_grpo.py --model.unsloth.enabled true

echo "Training completed successfully!"