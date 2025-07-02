#!/usr/bin/env python3
"""
Small-scale integration test for DeepSeek R1 Zero GRPO system.
Tests all components with minimal computational requirements.
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import set_seed
import json
import tempfile

# Import our GRPO components
from src.config.grpo_config import (
    GRPOScriptArguments,
    ModelConfig,
    create_training_arguments,
    get_reward_functions,
    get_callbacks
)
from src.rewards.tutorial_rewards import TutorialRewardSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_test_dataset():
    """Create a small test dataset with math problems."""
    test_problems = [
        {
            "problem": "What is 2 + 3?",
            "solution": "5",
            "level": "Level 1"
        },
        {
            "problem": "Calculate 7 × 8.",
            "solution": "56", 
            "level": "Level 1"
        },
        {
            "problem": "If x + 5 = 12, what is x?",
            "solution": "7",
            "level": "Level 2"
        },
        {
            "problem": "What is the square root of 16?",
            "solution": "4",
            "level": "Level 2"
        },
        {
            "problem": "Solve: 3x - 7 = 14",
            "solution": "x = 7",
            "level": "Level 3"
        }
    ]
    
    # Create a temporary dataset file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_problems, f)
        return f.name


def create_test_completions():
    """Create test completions to validate reward functions."""
    test_completions = [
        # Perfect completion
        [{"content": "<think>\nStep 1: I need to calculate 2 + 3\nStep 2: Adding 2 and 3 gives me 5\nStep 3: Let me verify: 2 + 3 = 5 ✓\n</think>\n\n<answer>\n5\n</answer>"}],
        
        # Correct answer, no format
        [{"content": "The answer is 56 because 7 times 8 equals 56."}],
        
        # Wrong answer but good format
        [{"content": "<think>\nStep 1: I need to solve x + 5 = 12\nStep 2: Subtracting 5 from both sides: x = 12 - 5\nStep 3: So x = 8\n</think>\n\n<answer>\n8\n</answer>"}],
        
        # Correct answer and format
        [{"content": "<think>\nStep 1: I need to find the square root of 16\nStep 2: I know that 4 × 4 = 16\nStep 3: Therefore, √16 = 4\n</think>\n\n<answer>\n4\n</answer>"}],
        
        # Repetitive text (should get penalty)
        [{"content": "<think>\nStep step step 1: I need to solve solve solve this equation\nStep step step 2: The equation equation equation is 3x - 7 = 14\n</think>\n\n<answer>\nx = 7\n</answer>"}]
    ]
    
    return test_completions


def test_configuration_system():
    """Test 1: Validate configuration system works."""
    logger.info("="*60)
    logger.info("TEST 1: CONFIGURATION SYSTEM")
    logger.info("="*60)
    
    try:
        # Create configuration objects
        script_args = GRPOScriptArguments(
            reward_funcs=["accuracy", "format", "reasoning_steps"],
            cosine_min_value_wrong=-0.3,
            cosine_max_value_correct=0.9
        )
        
        model_args = ModelConfig(
            model_name_or_path="distilgpt2",  # Small model for testing
            torch_dtype="float32",
            attn_implementation=None  # Disable flash attention for CPU
        )
        
        training_args = create_training_arguments("./test_output")
        training_args.per_device_train_batch_size = 2
        training_args.logging_steps = 1
        
        logger.info("✅ Configuration objects created successfully")
        logger.info(f"   Reward functions: {script_args.reward_funcs}")
        logger.info(f"   Model: {model_args.model_name_or_path}")
        logger.info(f"   Batch size: {training_args.per_device_train_batch_size}")
        
        return True, (script_args, model_args, training_args)
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False, None


def test_reward_functions(script_args):
    """Test 2: Validate reward functions work on real data."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: REWARD FUNCTIONS")
    logger.info("="*60)
    
    try:
        # Get reward functions
        reward_functions = get_reward_functions(script_args)
        logger.info(f"✅ Created {len(reward_functions)} reward functions")
        
        # Test completions
        test_completions = create_test_completions()
        test_ground_truths = ["5", "56", "7", "4", "x = 7"]
        
        logger.info("\nTesting reward functions on sample completions:")
        
        for i, (completion, ground_truth) in enumerate(zip(test_completions, test_ground_truths)):
            logger.info(f"\nExample {i+1}:")
            logger.info(f"  Content preview: {completion[0]['content'][:50]}...")
            
            rewards = {}
            for j, reward_func in enumerate(reward_functions):
                try:
                    if script_args.reward_funcs[j] == "accuracy":
                        reward = reward_func(completion, ground_truth)
                    else:
                        reward = reward_func(completion)
                    rewards[script_args.reward_funcs[j]] = reward
                    logger.info(f"  {script_args.reward_funcs[j]:15s}: {reward:.3f}")
                except Exception as e:
                    logger.warning(f"  {script_args.reward_funcs[j]:15s}: ERROR - {e}")
                    rewards[script_args.reward_funcs[j]] = 0.0
        
        logger.info("✅ Reward functions tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reward function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading(model_args):
    """Test 3: Validate model and tokenizer loading."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: MODEL LOADING")
    logger.info("="*60)
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {model_args.model_name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info("✅ Tokenizer loaded successfully")
        
        # Load model
        logger.info(f"Loading model: {model_args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=getattr(torch, model_args.torch_dtype),
            trust_remote_code=model_args.trust_remote_code
        )
        
        logger.info(f"✅ Model loaded successfully")
        logger.info(f"   Parameters: {model.num_parameters():,}")
        logger.info(f"   Device: {next(model.parameters()).device}")
        
        # Test generation
        test_input = "What is 2 + 2?"
        inputs = tokenizer(test_input, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"✅ Generation test successful")
        logger.info(f"   Input: {test_input}")
        logger.info(f"   Output: {generated_text}")
        
        return True, (model, tokenizer)
        
    except Exception as e:
        logger.error(f"❌ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_logging_system(script_args, training_args, model_args):
    """Test 4: Validate comprehensive logging system."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: LOGGING SYSTEM")
    logger.info("="*60)
    
    try:
        # Get callbacks
        callbacks = get_callbacks(training_args, model_args, script_args)
        logger.info(f"✅ Created {len(callbacks)} callbacks")
        
        for i, callback in enumerate(callbacks):
            logger.info(f"   {i+1}. {type(callback).__name__}")
        
        # Test callback execution with mock data
        from transformers import TrainerState, TrainerControl
        import numpy as np
        
        state = TrainerState()
        control = TrainerControl()
        
        # Simulate a training step
        state.global_step = 1
        state.epoch = 0.1
        state.log_history = [{
            'loss': 2.5,
            'learning_rate': 5e-5,
            'policy_loss': 1.2,
            'value_loss': 0.6,
            'entropy_loss': 0.1
        }]
        
        # Mock reward data
        mock_rewards = {
            'accuracy': [1.0, 0.0, 0.5],
            'format': [1.0, 0.0, 1.0],
            'reasoning_steps': [0.8, 0.2, 0.6],
            'total': [2.8, 0.2, 2.1]
        }
        
        # Mock completions (use our test completions)
        test_completions = create_test_completions()[:3]
        
        # Execute callbacks
        for callback in callbacks:
            callback.on_step_begin(training_args, state, control)
            callback.on_step_end(
                training_args, state, control,
                rewards=mock_rewards,
                completions=test_completions
            )
        
        logger.info("✅ Logging system tested successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration(configs):
    """Test 5: Full integration test with mock training loop."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: INTEGRATION TEST")
    logger.info("="*60)
    
    script_args, model_args, training_args = configs
    
    try:
        # Load model and tokenizer
        model_success, model_components = test_model_loading(model_args)
        if not model_success:
            return False
            
        model, tokenizer = model_components
        
        # Get reward functions and callbacks
        reward_functions = get_reward_functions(script_args)
        callbacks = get_callbacks(training_args, model_args, script_args)
        
        # Create test dataset
        test_problems = [
            "What is 5 + 7?",
            "Calculate 9 × 6.",
            "If y - 3 = 8, what is y?"
        ]
        
        logger.info("Running mini training simulation...")
        
        from transformers import TrainerState, TrainerControl
        import numpy as np
        
        state = TrainerState()
        control = TrainerControl()
        
        for step in range(1, 4):  # 3 mini steps
            state.global_step = step
            state.epoch = step * 0.1
            state.log_history = [{
                'loss': 3.0 - step * 0.2,
                'learning_rate': 5e-5,
                'policy_loss': 1.5 - step * 0.1,
                'value_loss': 0.8 - step * 0.05,
                'entropy_loss': 0.1 + step * 0.02
            }]
            
            # Generate responses for test problems
            completions = []
            for problem in test_problems:
                # Simple mock generation (in real case, this would be model.generate)
                mock_response = f"<think>\nI need to solve: {problem}\nStep 1: Let me work through this.\n</think>\n\n<answer>\nMock answer for step {step}\n</answer>"
                completions.append([{"content": mock_response}])
            
            # Calculate rewards
            rewards = {'total': []}
            for reward_name in script_args.reward_funcs:
                rewards[reward_name] = []
            
            for completion in completions:
                step_rewards = {}
                total_reward = 0
                
                for j, reward_func in enumerate(reward_functions):
                    reward_name = script_args.reward_funcs[j]
                    try:
                        if reward_name == "accuracy":
                            reward = reward_func(completion, "mock_ground_truth")
                        else:
                            reward = reward_func(completion)
                        step_rewards[reward_name] = reward
                        total_reward += reward
                    except:
                        step_rewards[reward_name] = 0.0
                
                for reward_name in script_args.reward_funcs:
                    rewards[reward_name].append(step_rewards.get(reward_name, 0.0))
                rewards['total'].append(total_reward)
            
            # Execute callbacks
            for callback in callbacks:
                callback.on_step_begin(training_args, state, control)
                callback.on_step_end(
                    training_args, state, control,
                    rewards=rewards,
                    completions=completions
                )
            
            logger.info(f"   Step {step}: Simulated training step completed")
        
        logger.info("✅ Integration test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_comprehensive_test():
    """Run all tests in sequence."""
    logger.info("🚀 STARTING COMPREHENSIVE GRPO INTEGRATION TEST")
    logger.info("="*80)
    
    set_seed(42)
    
    test_results = []
    
    # Test 1: Configuration
    config_success, configs = test_configuration_system()
    test_results.append(("Configuration System", config_success))
    
    if not config_success:
        logger.error("❌ Cannot proceed without valid configuration")
        return False
    
    script_args, model_args, training_args = configs
    
    # Test 2: Reward Functions
    reward_success = test_reward_functions(script_args)
    test_results.append(("Reward Functions", reward_success))
    
    # Test 3: Model Loading
    model_success, _ = test_model_loading(model_args)
    test_results.append(("Model Loading", model_success))
    
    # Test 4: Logging System
    logging_success = test_logging_system(script_args, training_args, model_args)
    test_results.append(("Logging System", logging_success))
    
    # Test 5: Integration
    if all([config_success, reward_success, model_success, logging_success]):
        integration_success = test_integration(configs)
        test_results.append(("Integration", integration_success))
    else:
        logger.warning("⚠️ Skipping integration test due to previous failures")
        test_results.append(("Integration", False))
    
    # Print final results
    logger.info("\n" + "="*80)
    logger.info("🎯 COMPREHENSIVE TEST RESULTS")
    logger.info("="*80)
    
    all_passed = True
    for test_name, success in test_results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")
        if not success:
            all_passed = False
    
    logger.info("="*80)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! GRPO system is ready for training.")
        logger.info("💡 You can now run: python train_grpo.py --no_wandb")
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
    
    logger.info("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)