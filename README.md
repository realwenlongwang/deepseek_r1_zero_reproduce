# DeepSeek R1 Zero Reproduction

A comprehensive reproduction of the DeepSeek R1 Zero training methodology using GRPO (Gradient Reward Policy Optimization) with **Qwen/Qwen2.5-7B-Instruct** as the base model.

This implementation follows the exact specifications from the [DeepSeek R1 training tutorial](https://github.com/FareedKhan-dev/train-deepseek-r1) with comprehensive logging and monitoring capabilities.

## 🚀 Features

- **📊 Comprehensive GRPO Training**: Full implementation with 5 reward functions
- **🎯 Multi-Reward System**: Accuracy, format, reasoning steps, cosine scaling, repetition penalty
- **📈 Advanced Logging**: Detailed training metrics, reward breakdowns, generation quality analysis
- **📚 Multi-Dataset Support**: Countdown-Tasks, NuminaMath-TIR and Bespoke-Stratos-17k datasets
- **🤔 Structured Reasoning**: `<think>` and `<answer>` tags for step-by-step reasoning
- **⚡ Performance Optimized**: Flash Attention 2, gradient checkpointing, mixed precision
- **🔧 Flexible Configuration**: Command-line arguments and tutorial-exact configurations

## 📋 Quick Start

### 1. Setup Environment
```bash
# Install dependencies using UV package manager
uv sync

# Note: Flash Attention may need to be installed separately first
uv pip install "flash-attn>=2.3.0"
```

### 2. Run GRPO Training
```bash
# Basic training with comprehensive logging
uv run train_grpo.py --no_wandb --reward_funcs accuracy format reasoning_steps

# Full training with all reward functions
uv run train_grpo.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --dataset_name "Jiayi-Pan/Countdown-Tasks-3to4" \
    --reward_funcs accuracy format reasoning_steps cosine repetition_penalty \
    --learning_rate 5e-5 \
    --per_device_train_batch_size 8 \
    --logging_steps 10
```

### 3. Monitor Training
Training logs include:
- **Basic metrics**: Loss, learning rate, epoch progress
- **GRPO metrics**: Policy loss, value loss, entropy loss
- **Reward breakdown**: Statistics for each reward function
- **Generation quality**: Format compliance, reasoning indicators
- **Performance**: GPU/RAM usage, training speed

## 🎯 Reward Functions

Our implementation includes 5 reward functions following the tutorial exactly:

| Reward Function | Purpose | Range |
|----------------|---------|-------|
| **Accuracy** | Mathematical correctness | 0.0 - 1.0 |
| **Format** | `<think>/<answer>` structure | 0.0 - 1.0 |
| **Reasoning Steps** | Step-by-step reasoning quality | 0.0 - 1.0 |
| **Cosine Scaling** | Length-based reward scaling | -0.5 - 1.0 |
| **Repetition Penalty** | Penalize repetitive text | -0.1 - 0.0 |

## 📊 Logging System

### Comprehensive Metrics
- **Training Progress**: Loss curves, learning rate schedules
- **Reward Analysis**: Mean, std, min/max for each reward function
- **Generation Quality**: Format compliance rates, response lengths
- **Performance Monitoring**: GPU memory, CPU usage, training speed
- **Trend Detection**: Automatic alerts for reward improvements/degradations

### Sample Output
```
Step   10 | Loss: 2.3450 | LR: 5.00e-05 | Epoch: 0.50
         GRPO | Policy: 1.2340 | Value: 0.5670 | Entropy: 0.0890

Reward Breakdown:
  accuracy       :  0.750 ± 0.433 [ 0.000,  1.000]
  format         :  0.875 ± 0.331 [ 0.000,  1.000] 
  reasoning_steps:  0.625 ± 0.217 [ 0.330,  1.000]
  total          :  2.250 (combined)

Generation Quality:
  Format compliance:  87.5% (7/8)
  Avg response length:  145.3 chars
  Reasoning indicators:  2.14 per response

Performance:
  Step time:      3.24s
  Samples/sec:    2.5
  GPU memory:     4.32GB
```

## 🗂️ Project Structure

```
deepseek_r1_zero_repro/
├── train_grpo.py                 # Main GRPO training script
├── requirements.txt              # Dependencies
├── train-deepseek-tutorial.md    # Original tutorial reference
├── LOGGING_SYSTEM.md            # Detailed logging documentation
├── CONFIG_INTEGRATION.md        # Configuration system docs
├── src/
│   ├── config/
│   │   ├── grpo_config.py       # Tutorial-exact GRPO configuration
│   │   └── config.py            # Legacy configuration system
│   ├── data/
│   │   └── dataset.py           # Dataset processing
│   ├── rewards/
│   │   ├── tutorial_rewards.py  # 5 reward functions (tutorial-exact)
│   │   └── reward_functions.py  # Legacy reward system
│   ├── models/
│   │   └── reasoning_model.py   # Model utilities
│   └── training/
│       ├── grpo.py              # GRPO implementation
│       └── trainer.py           # Training utilities
├── tests/
│   ├── test_callbacks.py        # Comprehensive callback tests
│   ├── test_tutorial_rewards.py # Reward function tests
│   └── test_config_integration.py # Configuration tests
└── configs/
    └── default_config.yaml      # Default configuration
```

## ⚙️ Configuration Options

### Model Configuration
```bash
--model_name "Qwen/Qwen2.5-7B-Instruct"  # Base model
--torch_dtype "bfloat16"                   # Mixed precision
--attn_implementation "flash_attention_2"  # Attention optimization
```

### Training Configuration
```bash
--learning_rate 5e-5                      # Learning rate
--per_device_train_batch_size 8           # Batch size per GPU
--gradient_accumulation_steps 2           # Gradient accumulation
--num_train_epochs 1                      # Training epochs
--logging_steps 10                        # Logging frequency
```

### Reward Function Configuration
```bash
--reward_funcs accuracy format reasoning_steps cosine repetition_penalty
--cosine_min_value_wrong -0.5             # Cosine scaling parameters
--cosine_max_value_correct 1.0
--repetition_n_grams 3                    # Repetition penalty n-grams
```

## 📚 Dataset Support

### Supported Datasets
- **Countdown-Tasks-3to4**: Number puzzle games requiring arithmetic operations (default)
- **NuminaMath-TIR**: Mathematical reasoning problems
- **Bespoke-Stratos-17k**: Structured reasoning datasets
- **Custom datasets**: Any dataset with `problem` field

### Data Format
Input data is automatically converted to conversation format:
```python
{
    "prompt": [
        {"role": "system", "content": "You are a helpful assistant..."},
        {"role": "user", "content": "Solve: 2 + 2 = ?"},
    ],
}
```

### Expected Output Format
```
<think>
Step 1: I need to solve 2 + 2
Step 2: Adding 2 and 2 gives me 4
Step 3: Let me verify: 2 + 2 = 4 ✓
</think>

<answer>
4
</answer>
```

## 🧪 Testing

Run comprehensive tests to verify the implementation:

```bash
# Test all components
uv run python tests/test_callbacks.py
uv run python tests/test_tutorial_rewards.py
uv run python tests/test_config_integration.py

# Expected output: All tests should pass ✅
```

## 📈 Advanced Usage

### Custom Reward Functions
Extend the reward system by modifying `src/rewards/tutorial_rewards.py`:

```python
def custom_reward(completion, ground_truth=None):
    """Custom reward function."""
    # Your custom logic here
    return reward_score
```

### Integration with Weights & Biases
```bash
# Enable W&B logging
uv run train_grpo.py \
    --wandb_project "my-deepseek-r1" \
    --wandb_run_name "experiment-1"
```

### Memory Optimization
```bash
# For limited GPU memory
uv run train_grpo.py \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --torch_dtype "bfloat16"
```

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Error**: Reduce `per_device_train_batch_size` or enable gradient checkpointing
2. **Slow Training**: Check data loading, enable Flash Attention 2
3. **Reward Stagnation**: Adjust learning rate or reward function weights
4. **Format Compliance Issues**: Increase format reward weight

### Performance Optimization
- Use Flash Attention 2 for faster training
- Enable gradient checkpointing for memory efficiency
- Use mixed precision (bfloat16) for speed
- Optimize batch size for your hardware

## 📖 Documentation

- **[LOGGING_SYSTEM.md](LOGGING_SYSTEM.md)**: Comprehensive logging system documentation
- **[CONFIG_INTEGRATION.md](CONFIG_INTEGRATION.md)**: Configuration system details
- **[train-deepseek-tutorial.md](train-deepseek-tutorial.md)**: Original tutorial reference

## 🏆 Results

The system provides:
- **Comprehensive monitoring** of training progress
- **Detailed reward analysis** for each function
- **Generation quality metrics** for model improvement
- **Performance optimization** insights
- **Trend detection** for training issues

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project follows the same license as the original DeepSeek R1 tutorial.

## 🙏 Acknowledgments

- **DeepSeek Team**: For the original R1 methodology
- **FareedKhan-dev**: For the comprehensive training tutorial
- **Qwen Team**: For the excellent base model
- **Hugging Face**: For the transformers library

---

**Status**: ✅ **COMPREHENSIVE IMPLEMENTATION READY**

This reproduction includes all tutorial components with extensive logging and monitoring capabilities. Ready for production GRPO training!