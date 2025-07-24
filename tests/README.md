# Tests Directory

This directory contains test files for the DeepSeek R1 Zero GRPO training project.

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_tutorial_rewards_comprehensive.py
pytest tests/test_grpo_integration.py
pytest tests/test_config_profiles.py
```

## Test Categories

### Core Tests
- `test_tutorial_rewards_comprehensive.py` - Comprehensive reward function tests
- `test_grpo_integration.py` - GRPO trainer integration tests
- `test_dataset_consistency.py` - Dataset loading and consistency tests
- `test_train_test_split.py` - Dataset splitting tests

### Configuration Tests
- `test_config.py` - Configuration system tests
- `test_config_profiles.py` - Configuration profiles tests
- `test_config_combinations.py` - Configuration combinations tests
- `test_tutorial_config.py` - Tutorial-specific configuration tests

### GPU/Performance Tests
- `test_gpu_grpo.py` - GPU-specific GRPO tests
- `test_h200_optimization.py` - H200 optimization tests
- `test_device_placement.py` - Device placement tests

### Dataset Tests
- `test_real_dataset_download.py` - Real dataset download tests
- `test_real_dataset_exact_pattern.py` - Dataset pattern matching tests
- `test_countdown_*.py` - Countdown dataset specific tests
- `test_train_test_split_simple.py` - Simple dataset split tests

### Training Tests
- `test_training_integration.py` - Training integration tests
- `test_without_vllm.py` - Tests without vLLM
- `test_train_grpo_new.py` - New GRPO training tests
- `test_grpo_simple.py` - Simple GRPO tests
- `test_grpo_eval_minimal.py` - Minimal GRPO evaluation tests

### Callback Tests
- `test_callbacks.py` - Callback system tests
- `test_checkpoint_naming.py` - Checkpoint naming tests

### Utility Tests
- `test_reward_functions.py` - Reward function tests
- `test_inference.py` - Inference tests
- `test_profiles.py` - Profile system tests

### Issue-Specific Tests
- `test_padding_issue.py` - Padding-related tests
- `test_qwen_padding.py` - Qwen model padding tests
- `test_dataloader_bottleneck.py` - Dataloader performance tests
- `test_evaluation_freeze.py` - Evaluation freeze issue tests

## Test Organization

### Main Tests Directory
Contains all production-ready tests organized by functionality.

### Subdirectories
- `debug/` - Debug scripts and diagnostic tools
- `experimental/` - Experimental test files and performance tests

## Test File Structure

### Core Test Files
- **Configuration Tests**: `test_config*.py` - Test configuration system and profiles
- **Training Tests**: `test_*training*.py`, `test_grpo*.py` - Test training functionality
- **Dataset Tests**: `test_*dataset*.py`, `test_countdown*.py` - Test dataset handling
- **Reward Tests**: `test_reward*.py`, `test_tutorial_rewards*.py` - Test reward functions
- **Integration Tests**: `test_*integration*.py` - Test system integration

### Experimental Files
- `experimental/heavy_processing_test.py` - Heavy processing performance tests
- `experimental/quick_dataloader_test.py` - Quick dataloader tests
- `experimental/simple_worker_test.py` - Simple worker tests

### Debug Files
- `debug/debug_rewards.py` - Reward function debugging
- `debug/debug_tokenizer_padding.py` - Tokenizer padding debugging
- `debug/quick_freeze_diagnostic.py` - Freeze issue diagnostics

## Adding New Tests

When adding new tests:
1. Follow the naming convention `test_<component>_<description>.py`
2. Include comprehensive docstrings
3. Use pytest fixtures for common setup
4. Group related tests into classes
5. Use appropriate test markers for slow or GPU tests
6. Place experimental tests in `experimental/` subdirectory
7. Place debug scripts in `debug/` subdirectory

## Test Execution

### Run all tests:
```bash
pytest tests/
```

### Run specific categories:
```bash
pytest tests/test_config*.py          # Configuration tests
pytest tests/test_grpo*.py            # GRPO tests
pytest tests/test_*training*.py       # Training tests
pytest tests/test_*dataset*.py        # Dataset tests
```

### Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```