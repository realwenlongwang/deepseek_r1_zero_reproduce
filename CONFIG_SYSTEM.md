# Configuration System Documentation

This document describes the new centralized YAML configuration system for the DeepSeek R1 Zero GRPO training project.

## Features

- **Centralized Configuration**: All settings in one hierarchical YAML file
- **CLI Overrides**: Override any configuration value from command line
- **Array Syntax**: Support for bracket notation: `[item1,item2,item3]`
- **Type Validation**: Comprehensive type checking and business logic validation
- **Backwards Compatibility**: Legacy CLI arguments still work
- **Environment Variables**: Support for environment variable overrides
- **Configuration Profiles**: Pre-configured profiles for different environments

## Quick Start

### Basic Usage

```bash
# Use default configuration
python train_grpo.py

# Use development profile
python train_grpo.py --profile dev

# Override specific settings
python train_grpo.py --model.name "Qwen/Qwen2.5-7B" --training.epochs 2.0
```

### Array Syntax

```bash
# Set reward functions using bracket notation
python train_grpo.py --rewards.functions [accuracy,format,reasoning_steps]

# Alternative syntax
python train_grpo.py --rewards.functions=[accuracy,format,reasoning_steps]

# With spaces
python train_grpo.py --rewards.functions [accuracy, format, reasoning_steps]
```

### Legacy Compatibility

```bash
# Old arguments still work
python train_grpo.py --model_name "Qwen/Qwen2.5-7B" --learning_rate 1e-4 --no_wandb
```

## Configuration Structure

The configuration is organized hierarchically:

```yaml
project:          # Project metadata
model:            # Model configuration
training:         # Training settings
  batch_size:     # Batch size configuration
  optimization:   # Optimizer settings
  precision:      # Mixed precision settings
  scheduling:     # Training schedule
  dataloader:     # Data loading settings
dataset:          # Dataset configuration
  split:          # Train/test split settings
  processing:     # Data processing settings
grpo:             # GRPO-specific settings
  vllm:           # vLLM configuration
rewards:          # Reward function configuration
  cosine:         # Cosine reward settings
  repetition:     # Repetition penalty settings
  code:           # Code reward settings
  soft_punish:    # Soft punishment settings
system:           # System configuration
monitoring:       # Monitoring and logging
  wandb:          # Weights & Biases settings
  logging:        # Logging configuration
callbacks:        # Callback configuration
  comprehensive_logging:    # Detailed logging
  reward_trend:            # Reward trend analysis
  checkpoint_preservation: # Checkpoint preservation
```

## Configuration Profiles

### Available Profiles

1. **default** - Standard training configuration
2. **dev** - Development profile (faster, less logging)
3. **prod** - Production profile (full training, comprehensive logging)
4. **test** - Testing profile (minimal resources)
5. **profile** - Profiling profile (detailed performance analysis)

### Using Profiles

```bash
# Use development profile
python train_grpo.py --profile dev

# Use production profile with overrides
python train_grpo.py --profile prod --training.epochs 5.0
```

## CLI Override Syntax

### Dot Notation for Nested Values

```bash
# Set nested configuration values
python train_grpo.py --training.optimization.learning_rate 1e-4
python train_grpo.py --model.name "Qwen/Qwen2.5-7B"
python train_grpo.py --dataset.split.test_size 0.2
```

### Array Values

```bash
# Bracket notation (recommended)
python train_grpo.py --rewards.functions [accuracy,format,reasoning_steps]

# Equal sign syntax
python train_grpo.py --rewards.functions=[accuracy,format,reasoning_steps]

# With spaces
python train_grpo.py --rewards.functions [accuracy, format, reasoning_steps]
```

### Boolean Values

```bash
# Boolean flags
python train_grpo.py --monitoring.wandb.enabled false
python train_grpo.py --training.precision.gradient_checkpointing true
```

### Complex Example

```bash
python train_grpo.py \
    --profile prod \
    --model.name "Qwen/Qwen2.5-7B" \
    --training.epochs 3.0 \
    --training.optimization.learning_rate 3e-5 \
    --training.batch_size.per_device_train 4 \
    --rewards.functions [accuracy,format,reasoning_steps,cosine] \
    --monitoring.wandb.enabled true \
    --monitoring.wandb.run_name "production-run-v1" \
    --callbacks.checkpoint_preservation.every_n_steps 1000
```

## Environment Variables

Set environment variables with `GRPO_` prefix:

```bash
export GRPO_TRAINING_LEARNING_RATE=1e-4
export GRPO_MODEL_NAME="Qwen/Qwen2.5-7B"
export GRPO_MONITORING_WANDB_ENABLED=false

python train_grpo.py
```

## Legacy CLI Arguments

The system maintains backwards compatibility with legacy arguments:

| Legacy Argument | New Configuration Path |
|----------------|------------------------|
| `--model_name` | `--model.name` |
| `--learning_rate` | `--training.optimization.learning_rate` |
| `--per_device_train_batch_size` | `--training.batch_size.per_device_train` |
| `--reward_funcs` | `--rewards.functions` |
| `--no_wandb` | `--monitoring.wandb.enabled false` |

## Validation

The system provides comprehensive validation:

### Type Validation
- Ensures correct data types for all fields
- Validates array contents
- Checks boolean values

### Business Logic Validation
- Batch size must be divisible by 8 for GRPO
- Learning rate must be positive
- Test split size must be between 0 and 1
- Reward functions must be valid

### Hardware Validation
- Checks GPU availability for GPU-specific settings
- Validates memory requirements
- Warns about potential memory issues

### Example Validation Errors

```bash
# Invalid batch size (not divisible by 8)
python train_grpo.py --training.batch_size.per_device_train 7
# Error: Effective batch size (7) must be divisible by 8 for GRPO

# Invalid learning rate
python train_grpo.py --training.optimization.learning_rate -1
# Error: Learning rate must be positive

# Invalid reward function
python train_grpo.py --rewards.functions [invalid_function]
# Error: Invalid reward function 'invalid_function'
```

## Advanced Features

### Configuration Inheritance

```yaml
# config.yaml
_profiles:
  dev:
    training.epochs: 0.1
    monitoring.wandb.enabled: false
  prod:
    training.epochs: 3.0
    monitoring.wandb.enabled: true
```

### Profile Overrides

Profiles can override any configuration value:

```yaml
_profiles:
  gpu_optimized:
    training.precision.bf16: true
    training.precision.tf32: true
    training.dataloader.pin_memory: true
    grpo.vllm.enabled: true
    grpo.vllm.gpu_memory_utilization: 0.4
```

## Migration Guide

### From Legacy to New System

1. **Update CLI arguments**:
   ```bash
   # Old
   python train_grpo.py --model_name "Qwen/Qwen2.5-7B" --learning_rate 1e-4
   
   # New
   python train_grpo.py --model.name "Qwen/Qwen2.5-7B" --training.optimization.learning_rate 1e-4
   ```

2. **Use array syntax**:
   ```bash
   # Old
   python train_grpo.py --reward_funcs accuracy,format,reasoning_steps
   
   # New
   python train_grpo.py --rewards.functions [accuracy,format,reasoning_steps]
   ```

3. **Leverage profiles**:
   ```bash
   # Instead of many arguments
   python train_grpo.py --profile dev --training.epochs 0.5
   ```

### Configuration File Creation

Create a custom configuration file:

```bash
# Copy default configuration
cp config.yaml my_config.yaml

# Edit as needed
vim my_config.yaml

# Use custom configuration
python train_grpo.py --config my_config.yaml
```

## Troubleshooting

### Common Issues

1. **Validation Errors**: Check that batch sizes are divisible by 8
2. **Array Parsing**: Use brackets for arrays: `[item1,item2,item3]`
3. **Boolean Values**: Use lowercase: `true`/`false`
4. **Profile Not Found**: Check profile name in `_profiles` section

### Debug Mode

Enable debug logging:

```bash
python train_grpo.py --monitoring.logging.level DEBUG
```

### Configuration Validation

Test configuration without training:

```bash
python test_config.py
```

## API Reference

### ConfigManager Class

```python
from src.config import ConfigManager

# Create manager
manager = ConfigManager(
    config_file="config.yaml",
    profile="dev",
    enable_legacy_compatibility=True
)

# Load configuration
config = manager.load_config(cli_args=sys.argv[1:])

# Access configuration
print(config.model.name)
print(config.training.optimization.learning_rate)
```

### Configuration Schema

All configuration options are defined in `src/config/schemas.py` with comprehensive type annotations and default values.

## Contributing

When adding new configuration options:

1. Add to appropriate dataclass in `src/config/schemas.py`
2. Update field type mapping in `get_config_field_types()`
3. Add validation logic in `src/config/validator.py`
4. Update this documentation
5. Add tests in `test_config.py`

## Examples

### Development Training

```bash
python train_grpo.py \
    --profile dev \
    --rewards.functions [format,accuracy] \
    --training.epochs 0.1
```

### Production Training

```bash
python train_grpo.py \
    --profile prod \
    --model.name "Qwen/Qwen2.5-7B" \
    --training.epochs 3.0 \
    --training.optimization.learning_rate 3e-5 \
    --monitoring.wandb.run_name "production-run-$(date +%Y%m%d)"
```

### Quick Testing

```bash
python train_grpo.py \
    --profile test \
    --rewards.functions [format] \
    --training.epochs 0.01
```

### Performance Profiling

```bash
python train_grpo.py \
    --profile profile \
    --callbacks.comprehensive_logging.enabled true \
    --monitoring.logging.profiling_mode true
```