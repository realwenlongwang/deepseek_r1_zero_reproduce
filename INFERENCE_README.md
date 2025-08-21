# DeepSeek R1 Zero Inference System

Comprehensive inference system for DeepSeek R1 Zero trained models with automatic checkpoint detection, Unsloth optimization, and reward-based evaluation.

## Features

- 🔍 **Automatic Checkpoint Detection**: Supports both full models and LoRA adapters with Unsloth optimization
- 🎯 **Training Reward Function Integration**: Uses actual format_reward and equation_reward functions from training
- 🤖 **Multiple Inference Modes**: Interactive chat, single prompts, batch processing, dataset evaluation
- 📊 **Real-time Reward Evaluation**: Shows format and equation scores for each response
- ⚙️ **Generation Presets**: Creative, balanced, precise, deterministic, and reasoning modes
- 🚀 **Unsloth Optimization**: Automatic fast inference with LoRA adapter support
- 💾 **Comprehensive Output**: JSON results with detailed evaluation metrics

## Quick Start

### Interactive Chat
```bash
# Use latest checkpoint automatically
python inference.py --interactive

# Use specific checkpoint
python inference.py --checkpoint saved_models/permanent_checkpoints/checkpoint-20000 --interactive
```

### Single Prompt
```bash
python inference.py --prompt "Solve: x^2 + 5x + 6 = 0" --preset reasoning
```

### Dataset Evaluation with Reward Functions
```bash
# Evaluate on countdown dataset with reward scoring
python inference.py --eval_dataset countdown --samples 50 --output results.json

# Evaluate on numina math dataset
python inference.py --eval_dataset numina --samples 100 --output math_results.json

# Batch processing from file
python inference.py --batch_file prompts.txt --output responses.json
```

### List Available Checkpoints
```bash
python inference.py --list_checkpoints
```

## Architecture

### Core Components

1. **AutoCheckpointLoader** (`src/inference/checkpoint_loader.py`)
   - Detects checkpoint types (full model vs LoRA adapter)
   - Handles base model resolution for LoRA adapters
   - Applies appropriate Qwen2.5 chat templates

2. **InferenceEngine** (`src/inference/generators.py`)
   - Unified generation interface with preset support
   - Batch processing capabilities
   - Interactive chat mode

3. **Reward Function Integration** (`src/rewards/openr1_rewards.py`)
   - Real-time format_reward scoring during evaluation
   - equation_reward validation for countdown problems
   - Same reward functions used during GRPO training

4. **InteractiveSession** (`src/inference/interactive.py`)
   - Enhanced chat interface with conversation management
   - Command system (/help, /clear, /save, etc.)
   - Session persistence

## Checkpoint Types Supported

### LoRA Adapters (Recommended)
- **Location**: `saved_models/permanent_checkpoints/checkpoint-*`
- **Files**: `adapter_config.json`, `adapter_model.safetensors`
- **Base Model**: Auto-resolved from `adapter_config.json` (typically `unsloth/qwen2.5-7b-unsloth-bnb-4bit`)
- **Optimization**: Unsloth fast inference with automatic LoRA loading
- **Template**: Training-compatible ChatML

### Full Models  
- **Location**: `grpo_output/checkpoint-*`
- **Files**: `model.safetensors`, `config.json`
- **Template**: Official Qwen2.5 ChatML

## Chat Template Optimization

The system automatically applies the correct chat template based on checkpoint type:

### For LoRA Adapters (Training-Compatible)
```jinja
{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```

### For Full Models (Official Qwen2.5)
- Uses official Qwen2.5 ChatML template with proper system message
- Includes tool calling support and multi-turn conversation handling
- Default system message: "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

## Generation Presets

| Preset | Temperature | Top-p | Use Case |
|--------|-------------|--------|----------|
| **Creative** | 1.0 | 0.95 | Creative writing, brainstorming |
| **Balanced** | 0.7 | 0.9 | General conversation |
| **Precise** | 0.3 | 0.8 | Factual/mathematical content |
| **Deterministic** | 0.0 | - | Reproducible outputs |
| **Reasoning** | 0.4 | 0.85 | Step-by-step problem solving |

## Reward-Based Evaluation

The system uses the exact same reward functions from training for evaluation:

### Format Reward (format_reward)
- **Score**: 1.0 for perfect `<think>...</think><answer>...</answer>` structure, 0.0 otherwise
- **Display**: `📊 Format: ✅ (1.0)` or `📊 Format: ❌ (0.0)`
- **Validation**: Uses regex pattern matching for exact format compliance

### Equation Reward (equation_reward) - Countdown Dataset Only
- **Score**: 1.0 for mathematically correct equation using all numbers exactly once, 0.0 otherwise  
- **Display**: `📊 Format: ✅ (1.0) | Equation: ✅ (1.0)`
- **Validation**: Checks equation syntax, number usage, and mathematical correctness

### Sample Output
```
📊 Format: ✅ (1.0) | Equation: ✅ (1.0)  # Perfect response
📊 Format: ✅ (1.0) | Equation: ❌ (0.0)  # Good format, wrong math
📊 Format: ❌ (0.0) | Equation: ❌ (0.0)  # Poor format and math
```

## Examples

### Advanced Usage
```bash
# Reasoning preset with reward evaluation  
python inference.py \
    --checkpoint saved_models/permanent_checkpoints/checkpoint-20000 \
    --prompt "Find all solutions to x³ - 6x² + 11x - 6 = 0" \
    --preset reasoning \
    --max_new_tokens 1024

# Countdown evaluation with detailed reward scoring
python inference.py \
    --eval_dataset countdown \
    --samples 50 \
    --temperature 0.4 \
    --top_p 0.85 \
    --output countdown_results.json

# Interactive with custom system message
python inference.py \
    --interactive \
    --system_message "You are a math tutor. Always show step-by-step solutions." \
    --max_new_tokens 1024
```

### Command Line Options

#### Checkpoint Options
- `--checkpoint PATH`: Specific checkpoint path
- `--list_checkpoints`: List all available checkpoints

#### Inference Modes
- `--interactive`: Start interactive chat
- `--prompt TEXT`: Single prompt generation
- `--batch_file FILE`: Process prompts from file
- `--eval_dataset {countdown,numina}`: Dataset evaluation

#### Generation Parameters
- `--preset {creative,balanced,precise,deterministic,reasoning}`: Use preset
- `--temperature FLOAT`: Sampling temperature (default: 0.7)
- `--top_p FLOAT`: Top-p sampling (default: 0.9)
- `--top_k INT`: Top-k sampling (default: 50)
- `--repetition_penalty FLOAT`: Repetition penalty (default: 1.1)
- `--max_new_tokens INT`: Max tokens to generate (default: 1024)
- `--system_message TEXT`: Custom system message
- `--no_system_message`: Disable default system message

#### Output Options
- `--output FILE`: Save results to JSON file
- `--samples INT`: Number of samples to evaluate (default: 10)
- `--sample_offset INT`: Starting sample index (default: 0) 
- `--verbose`: Show detailed information
- `--show_metadata`: Show model metadata
- `--device {auto,cpu,cuda,cuda:0}`: Device for inference (default: auto)

## Testing

Run comprehensive tests to verify all components:

```bash
python test_comprehensive_inference.py
```

Tests cover:
- Checkpoint discovery and type detection
- Model loading for both checkpoint types
- Chat template configuration
- Generation across different presets
- Response validation and format checking
- Batch processing capabilities
- Error handling

## Integration with Training

The inference system is tightly integrated with the GRPO training pipeline:

1. **Direct Reward Function Usage**: Uses identical `format_reward` and `equation_reward` functions from training
2. **Checkpoint Compatibility**: Seamlessly handles LoRA adapters saved by `train_grpo.py`
3. **Template Consistency**: Maintains exact same ChatML templates used during training
4. **Unsloth Optimization**: Leverages same Unsloth optimizations used in training for fast inference
5. **Real-time Evaluation**: Provides immediate feedback using training metrics

## Performance Tips

1. **Use LoRA Checkpoints**: LoRA adapters load faster and use less memory than full models
2. **Unsloth Optimization**: System automatically applies Unsloth fast inference for LoRA adapters
3. **Token Limits**: Increased default from 512 to 1024 tokens to prevent truncation
4. **Device Selection**: Use `--device cuda:0` for predictable single GPU inference
5. **Temperature Settings**: Lower temperatures (0.1-0.4) for mathematical/countdown tasks

## Troubleshooting

### Common Issues

1. **"No checkpoints found"**
   - Run training first: `uv run train_grpo.py --reward_funcs accuracy format reasoning_steps`
   - Check `saved_models/` and `grpo_output/` directories

2. **"Unknown checkpoint format"**
   - Ensure checkpoint directory contains required files
   - LoRA: `adapter_config.json`, `adapter_model.safetensors`
   - Full model: `config.json`, `model.safetensors`

3. **Memory errors**
   - Use smaller models (`Qwen2.5-0.5B` vs `Qwen2.5-7B`)
   - Reduce `--max_new_tokens`
   - Use single GPU: `--device cuda:0`

4. **Poor reward scores**
   - Check if model was trained with format and equation rewards
   - Ensure proper `<think>...</think><answer>...</answer>` format in responses
   - For countdown problems, verify equation uses all numbers exactly once
   - Try `--preset reasoning` for better structured responses

5. **LoRA loading issues**
   - System includes automatic fallback mechanisms
   - Check base model path in `adapter_config.json`
   - Unsloth integration has automatic error handling

For detailed diagnostics, run with `--verbose --show_metadata` flags.