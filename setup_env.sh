#!/bin/bash
# Setup script for DeepSeek R1 Zero Reproduction Project
# Run with: source setup_env.sh

# CUDA Library Path - Required for unsloth/triton CUDA compilation
export LIBRARY_PATH="/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH"

# Optional: Disable CUDA cache warnings
export CUDA_LAUNCH_BLOCKING=0

echo "✓ Environment variables set for CUDA/Unsloth compatibility"
echo "✓ You can now run: uv run train_grpo.py --no_wandb --reward_funcs accuracy format reasoning_steps"