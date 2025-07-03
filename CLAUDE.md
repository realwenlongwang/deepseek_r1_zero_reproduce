# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DeepSeek R1 Zero reproduction project implementing GRPO (Gradient Reward Policy Optimization) training using the TRL library. The project follows the exact specifications from the DeepSeek R1 training tutorial and uses Qwen2.5 models (0.5B or 7B variants) as the base model.

## Key Commands

### Running Training
```bash
# Basic training with UV package manager
uv run train_grpo.py --no_wandb --reward_funcs accuracy format reasoning_steps

# Full training with all reward functions
uv run train_grpo.py \
    --model_name "./models/Qwen2.5-0.5B" \
    --reward_funcs accuracy format reasoning_steps cosine repetition_penalty \
    --per_device_train_batch_size 4 \
    --logging_steps 10

# Test training functionality
uv run train_grpo.py --no_wandb --reward_funcs accuracy format reasoning_steps
```

### Testing
```bash
# Run comprehensive tests
uv run python tests/test_tutorial_rewards_comprehensive.py
uv run python tests/test_callbacks.py
uv run python tests/test_grpo_integration.py

# Run specific test
uv run python -m pytest tests/test_tutorial_rewards_comprehensive.py::TestTutorialRewards::test_accuracy_reward -v
```

### Model Management
```bash
# Download models locally (already done)
huggingface-cli download Qwen/Qwen2.5-0.5B --local-dir ./models/Qwen2.5-0.5B
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

## Architecture

### Core Components

1. **train_grpo.py**: Main training script using TRL's GRPOTrainer
   - Implements complete GRPO training pipeline
   - Uses comprehensive logging and callbacks
   - Supports both local and HuggingFace models

2. **src/rewards/**: Reward function system
   - `tutorial_rewards.py`: 5 core reward functions following tutorial specs
   - `trl_reward_functions.py`: TRL integration wrapper functions
   - Supports accuracy, format, reasoning_steps, cosine, repetition_penalty rewards

3. **src/config/grpo_config.py**: Configuration management
   - GRPOScriptArguments for command-line arguments
   - ModelConfig for model settings
   - Comprehensive logging callbacks

4. **src/data/dataset.py**: Dataset processing
   - Supports NuminaMath-TIR and custom datasets
   - Converts to conversation format required by TRL

### Critical Implementation Details

#### GRPO Batch Size Constraint
- The effective batch size must be divisible by `num_generations_per_prompt` (default: 8)
- Use `--per_device_train_batch_size 4` to get effective batch size of 8
- Formula: `effective_batch_size = per_device_train_batch_size * num_gpus * gradient_accumulation_steps`

#### Reward Function Integration
- Tutorial reward functions expect `List[List[Dict]]` format (conversation format)
- TRL passes various formats including strings, lists, and tensors
- The `trl_reward_functions.py` contains wrapper functions that handle format conversion

#### Model Selection
- Qwen2.5-0.5B (494M parameters): Recommended for development/testing
- Qwen2.5-7B-Instruct (7.6B parameters): Full-scale training but requires more GPU memory
- Both models are stored locally in `./models/` directory

### Expected Output Format
The model should generate responses in this structure:
```
<think>
Step-by-step reasoning here...
</think>

<answer>
Final answer here
</answer>
```

## Common Issues and Solutions

### Memory Issues

**GPU Memory Requirements:**
- **Qwen2.5-0.5B**: ~12-15GB (recommended for development)
- **Qwen2.5-3B**: ~38-42GB (requires high-end GPU)
- **Qwen2.5-7B**: ~60-70GB (requires multiple GPUs or A100)

**If you encounter GPU memory issues, try these solutions in order:**

1. **Enable gradient checkpointing**: Set `gradient_checkpointing=True` in `src/config/grpo_config.py`
   - Trades computation for memory (saves ~2-4GB)
   - May show "Caching is incompatible with gradient checkpointing" warning (this is normal)

2. **Reduce batch size**: `--per_device_train_batch_size 4` or `--per_device_train_batch_size 2`
   - Effective batch size must be divisible by 8 for GRPO

3. **Switch to smaller model**: `--model_name "./models/Qwen2.5-0.5B"`
   - Much lower memory requirements while maintaining training functionality

4. **Use DeepSpeed ZeRO**: Enable DeepSpeed for memory optimization (advanced)

**Current default**: `gradient_checkpointing=False` for better performance when sufficient memory is available.

### Import/Dependency Issues
- The project uses fallback implementations for `latex2sympy2` and `math_verify`
- All reward functions have robust error handling
- Missing dependencies won't break training

### Batch Size Errors
- Ensure effective batch size is divisible by 8
- Common working configurations:
  - `--per_device_train_batch_size 4` (effective: 8)
  - `--per_device_train_batch_size 8` (effective: 16)

## Development Workflow

1. **Make changes** to reward functions in `src/rewards/tutorial_rewards.py`
2. **Test reward functions** with `uv run python tests/test_tutorial_rewards_comprehensive.py`
3. **Test integration** with `uv run python tests/test_grpo_integration.py`
4. **Run training** with `uv run train_grpo.py --no_wandb --reward_funcs accuracy format reasoning_steps`
5. **Monitor logs** in `training.log` and console output

## Key Files to Understand

- **train_grpo.py**: Entry point and main training loop
- **src/rewards/tutorial_rewards.py**: Core reward function implementations
- **src/rewards/trl_reward_functions.py**: TRL integration layer
- **src/config/grpo_config.py**: Configuration and callbacks
- **pyproject.toml**: Dependencies managed by UV

## Package Management

This project uses UV for dependency management. Always prefix Python commands with `uv run` to ensure proper environment isolation and dependency resolution.