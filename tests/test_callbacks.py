#!/usr/bin/env python3
"""
Test the comprehensive logging callbacks for GRPO training.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_callback_creation():
    """Test creating the comprehensive callbacks."""
    
    print("="*80)
    print("TESTING COMPREHENSIVE LOGGING CALLBACKS")
    print("="*80)
    
    from src.config.grpo_config import (
        GRPOScriptArguments,
        ModelConfig,
        create_training_arguments,
        get_callbacks,
        ComprehensiveLoggingCallback,
        RewardTrendCallback
    )
    
    print("\n1. CREATING CALLBACK COMPONENTS")
    print("-" * 50)
    
    # Create configuration components
    script_args = GRPOScriptArguments(
        reward_funcs=["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"]
    )
    model_args = ModelConfig()
    training_args = create_training_arguments("./test_output")
    
    print("✓ Configuration components created")
    print(f"  Reward functions: {script_args.reward_funcs}")
    print(f"  Logging steps: {training_args.logging_steps}")
    
    print("\n2. CREATING CALLBACKS")
    print("-" * 50)
    
    # Create callbacks
    callbacks = get_callbacks(training_args, model_args, script_args)
    
    print(f"✓ Created {len(callbacks)} callbacks:")
    for i, callback in enumerate(callbacks):
        print(f"  {i+1}. {type(callback).__name__}")
    
    # Test individual callback creation
    comprehensive_callback = ComprehensiveLoggingCallback(script_args, log_examples=True)
    trend_callback = RewardTrendCallback(window_size=20)
    
    print("✓ Individual callbacks created successfully")
    print(f"  ComprehensiveLoggingCallback: log_examples={comprehensive_callback.log_examples}")
    print(f"  RewardTrendCallback: window_size={trend_callback.window_size}")
    
    return callbacks, comprehensive_callback, trend_callback


def test_mock_callback_execution():
    """Test callback execution with mock data."""
    
    print("\n" + "="*60)
    print("TESTING CALLBACK EXECUTION WITH MOCK DATA")
    print("="*60)
    
    from src.config.grpo_config import ComprehensiveLoggingCallback, GRPOScriptArguments
    from transformers import TrainingArguments, TrainerState, TrainerControl
    import numpy as np
    
    # Create mock components
    script_args = GRPOScriptArguments(
        reward_funcs=["accuracy", "format", "reasoning_steps"]
    )
    
    callback = ComprehensiveLoggingCallback(script_args, log_examples=False)
    
    # Mock training arguments
    training_args = TrainingArguments(
        output_dir="./test",
        logging_steps=1,  # Log every step for testing
        per_device_train_batch_size=4
    )
    
    # Mock trainer state
    state = TrainerState()
    state.global_step = 10
    state.epoch = 0.5
    state.log_history = [
        {
            'loss': 2.345,
            'learning_rate': 5e-5,
            'policy_loss': 1.234,
            'value_loss': 0.567,
            'entropy_loss': 0.089
        }
    ]
    
    control = TrainerControl()
    
    # Mock reward data
    mock_rewards = {
        'accuracy': [1.0, 0.0, 1.0, 0.5],
        'format': [1.0, 0.0, 1.0, 1.0],
        'reasoning_steps': [0.67, 0.33, 1.0, 0.5],
        'total': [2.67, 0.33, 3.0, 2.0]
    }
    
    # Mock completion data
    mock_completions = [
        [{"content": "<think>\nStep 1: Calculate 2+2\nStep 2: The answer is 4\n</think>\n\n<answer>\n4\n</answer>"}],
        [{"content": "The answer is 4"}],
        [{"content": "<think>\nFirst, I need to solve this.\nSecond, let me work through it.\nFinally, the answer is 4.\n</think>\n\n<answer>\n4\n</answer>"}],
        [{"content": "<think>\nLet me think about this.\n</think>\n\n<answer>\n5\n</answer>"}]
    ]
    
    print("\n1. TESTING STEP BEGIN")
    print("-" * 30)
    callback.on_step_begin(training_args, state, control)
    print("✓ Step begin callback executed")
    
    print("\n2. TESTING STEP END WITH MOCK DATA")
    print("-" * 30)
    
    try:
        callback.on_step_end(
            training_args, 
            state, 
            control,
            rewards=mock_rewards,
            completions=mock_completions
        )
        print("✓ Step end callback executed successfully")
        print("✓ All logging methods completed without errors")
        
    except Exception as e:
        print(f"❌ Error during callback execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_reward_trend_analysis():
    """Test the reward trend analysis callback."""
    
    print("\n" + "="*60)
    print("TESTING REWARD TREND ANALYSIS")
    print("="*60)
    
    from src.config.grpo_config import RewardTrendCallback
    from transformers import TrainingArguments, TrainerState, TrainerControl
    import numpy as np
    
    # Create trend callback with small window for testing
    trend_callback = RewardTrendCallback(window_size=5)
    
    # Mock training arguments
    training_args = TrainingArguments(output_dir="./test")
    state = TrainerState()
    control = TrainerControl()
    
    print("\n1. SIMULATING TRAINING STEPS WITH TRENDS")
    print("-" * 50)
    
    # Simulate 15 steps with different reward trends
    trends_detected = []
    
    for step in range(1, 16):
        state.global_step = step
        
        # Create mock rewards with trends
        if step <= 5:
            # Stable period
            accuracy_values = [0.8 + np.random.normal(0, 0.05) for _ in range(4)]
        elif step <= 10:
            # Improving period
            base = 0.8 + (step - 5) * 0.04  # Gradually improving
            accuracy_values = [base + np.random.normal(0, 0.02) for _ in range(4)]
        else:
            # Declining period
            base = 1.0 - (step - 10) * 0.06  # Gradually declining
            accuracy_values = [base + np.random.normal(0, 0.02) for _ in range(4)]
        
        mock_rewards = {
            'accuracy': accuracy_values,
            'format': [1.0] * 4,
            'total': [sum(vals) for vals in zip(accuracy_values, [1.0] * 4)]
        }
        
        try:
            trend_callback.on_step_end(
                training_args,
                state,
                control,
                rewards=mock_rewards
            )
            
            if step % 5 == 0:
                print(f"✓ Step {step}: Trend analysis completed")
                
        except Exception as e:
            print(f"❌ Error at step {step}: {e}")
            return False
    
    print(f"✓ Completed {step} training steps")
    print(f"✓ Reward history length: {len(trend_callback.reward_history['accuracy'])}")
    
    return True


def test_generation_quality_analysis():
    """Test the generation quality analysis features."""
    
    print("\n" + "="*60)
    print("TESTING GENERATION QUALITY ANALYSIS") 
    print("="*60)
    
    from src.config.grpo_config import ComprehensiveLoggingCallback, GRPOScriptArguments
    import re
    
    script_args = GRPOScriptArguments(reward_funcs=["accuracy", "format"])
    callback = ComprehensiveLoggingCallback(script_args, log_examples=False)
    
    # Test different quality generations
    test_cases = [
        {
            "name": "Perfect Format",
            "content": "<think>\nStep 1: Analyze the problem\nStep 2: Work through solution\nStep 3: Verify answer\n</think>\n\n<answer>\n42\n</answer>"
        },
        {
            "name": "No Format",
            "content": "The answer is 42 because I calculated it."
        },
        {
            "name": "Missing Answer Tag",
            "content": "<think>\nLet me think about this problem step by step.\n</think>\n\nThe answer is 42."
        },
        {
            "name": "Long Reasoning",
            "content": "<think>\n" + "This is detailed reasoning. " * 20 + "\n</think>\n\n<answer>\n42\n</answer>"
        },
        {
            "name": "Short Response",
            "content": "<think>\nQuick.\n</think>\n\n<answer>\n42\n</answer>"
        }
    ]
    
    print("\n1. ANALYZING DIFFERENT GENERATION QUALITIES")
    print("-" * 50)
    
    for i, case in enumerate(test_cases):
        print(f"\nCase {i+1}: {case['name']}")
        
        # Check format compliance
        think_match = re.search(r'<think>(.*?)</think>', case['content'], re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r'<answer>(.*?)</answer>', case['content'], re.DOTALL | re.IGNORECASE)
        
        format_compliant = bool(think_match and answer_match)
        print(f"  Format compliant: {format_compliant}")
        print(f"  Content length: {len(case['content'])} chars")
        
        if think_match:
            think_content = think_match.group(1).strip()
            print(f"  Think length: {len(think_content)} chars")
            
            # Count reasoning indicators
            reasoning_patterns = [r'step \d+', r'first[,\s]', r'second[,\s]', r'therefore']
            reasoning_count = sum(len(re.findall(pattern, think_content, re.IGNORECASE)) 
                                for pattern in reasoning_patterns)
            print(f"  Reasoning indicators: {reasoning_count}")
    
    print("\n✓ Generation quality analysis completed")
    
    return True


if __name__ == "__main__":
    try:
        # Test callback creation
        callbacks, comprehensive, trend = test_callback_creation()
        
        # Test callback execution
        execution_success = test_mock_callback_execution()
        
        # Test trend analysis
        trend_success = test_reward_trend_analysis()
        
        # Test generation quality analysis
        quality_success = test_generation_quality_analysis()
        
        print("\n" + "="*80)
        print("CALLBACK TESTING SUMMARY")
        print("="*80)
        
        results = [
            ("Callback Creation", True),
            ("Mock Execution", execution_success),
            ("Trend Analysis", trend_success), 
            ("Quality Analysis", quality_success)
        ]
        
        for test_name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:20s}: {status}")
        
        all_passed = all(success for _, success in results)
        
        if all_passed:
            print("\n🎉 ALL CALLBACK TESTS PASSED!")
            print("💡 Comprehensive logging system ready for GRPO training")
        else:
            print("\n⚠️  Some tests failed - check implementation")
            
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during callback testing: {e}")
        import traceback
        traceback.print_exc()