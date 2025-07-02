# Tests Directory

This directory contains test files for the DeepSeek R1 Zero reproduction project.

## Test Files

### `test_tutorial_rewards.py`
- **Purpose**: Comprehensive testing of tutorial-based reward functions
- **Features**:
  - Individual reward function testing
  - Complete reward system validation
  - Format reward comparison
  - Edge case handling
  - Configuration analysis

**Usage:**
```bash
python tests/test_tutorial_rewards.py
```

## Test Categories

### 1. Individual Reward Function Tests
- Format Reward (binary scoring)
- Reasoning Steps Reward (step counting)
- Repetition Penalty (n-gram analysis)
- Accuracy Reward (answer matching)
- Cosine Scaling (length-based adjustment)

### 2. Integration Tests
- Complete reward system with all functions
- Different configuration combinations
- Reward weight analysis

### 3. Edge Case Tests
- Empty completions
- Very long text
- Special characters
- Multiple answer tags
- Malformed input

## Sample Test Results

**Good Completion** (follows format, correct answer):
- Format: 1.0
- Accuracy: 1.0  
- Reasoning: 0.333
- Total: ~3.3

**Poor Completion** (no format, wrong answer):
- Format: 0.0
- Accuracy: 0.0
- Total: ~-0.5

## Future Test Ideas

- Performance benchmarking
- Comparison with original reward functions
- Dataset-specific validation
- GRPO integration testing
- Model generation quality assessment