#!/usr/bin/env python3
"""
Minimal test to reproduce the GRPO evaluation freeze issue.
This script systematically tests the evaluation pipeline components.
"""

import os
import sys
import time
import logging
import warnings
import torch
import signal
from contextlib import contextmanager
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds):
    """Context manager to timeout operations."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def test_basic_components():
    """Test basic components step by step."""
    logger.info("="*60)
    logger.info("🧪 Testing Basic Components")
    logger.info("="*60)
    
    # Test 1: Import dependencies
    logger.info("1. Testing imports...")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import GRPOConfig, GRPOTrainer
        from src.config import Config, ConfigManager
        from src.data.dataset import create_train_test_datasets
        from src.rewards.openr1_rewards import get_reward_funcs
        from src.config.grpo_config import GRPOScriptArguments
        logger.info("✅ All imports successful")
    except Exception as e:
        logger.error(f"❌ Import error: {e}")
        return False
    
    # Test 2: GPU availability
    logger.info("2. Testing GPU availability...")
    if torch.cuda.is_available():
        logger.info(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("⚠️ No GPU available, using CPU")
    
    # Test 3: Model loading
    logger.info("3. Testing model loading...")
    try:
        with timeout(120):
            model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else None,
                use_cache=False
            )
            logger.info(f"✅ Model loaded: {model.config.model_type}")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"❌ Model loading error: {e}")
        return False
    
    # Test 4: Dataset loading
    logger.info("4. Testing dataset loading...")
    try:
        with timeout(120):
            train_dataset, test_dataset = create_train_test_datasets(
                dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
                test_size=0.05,  # Smaller for testing
                split_seed=42
            )
            logger.info(f"✅ Dataset loaded: Train={len(train_dataset)}, Test={len(test_dataset)}")
    except Exception as e:
        logger.error(f"❌ Dataset loading error: {e}")
        return False
    
    # Test 5: Reward functions
    logger.info("5. Testing reward functions...")
    try:
        script_args = GRPOScriptArguments(
            reward_funcs=["format", "equation"],
            cosine_min_value_wrong=-0.5,
            cosine_max_value_wrong=-0.1,
            cosine_min_value_correct=0.8,
            cosine_max_value_correct=1.0,
            cosine_max_len=1000,
            repetition_n_grams=3,
            repetition_max_penalty=-0.1,
            code_language="python",
            max_completion_len=512,
            soft_punish_cache=50,
        )
        
        reward_functions = get_reward_funcs(script_args)
        logger.info(f"✅ Reward functions loaded: {len(reward_functions)}")
    except Exception as e:
        logger.error(f"❌ Reward functions error: {e}")
        return False
    
    logger.info("✅ All basic components working")
    return True


def test_grpo_trainer_creation():
    """Test GRPOTrainer creation with different configurations."""
    logger.info("="*60)
    logger.info("🧪 Testing GRPOTrainer Creation")
    logger.info("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import GRPOConfig, GRPOTrainer
    from src.rewards.openr1_rewards import get_reward_funcs
    from src.config.grpo_config import GRPOScriptArguments
    
    # Load model and tokenizer
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            use_cache=False
        )
        
        # Create minimal dataset
        mock_dataset = [
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
            {"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "6"}]},
            {"messages": [{"role": "user", "content": "What is 4+4?"}, {"role": "assistant", "content": "8"}]},
        ]
        
        # Create reward functions
        script_args = GRPOScriptArguments(reward_funcs=["format", "equation"])
        reward_functions = get_reward_funcs(script_args)
        
        # Test different configurations
        configs = [
            {"vllm": False, "eval_freq": 50, "workers": 0},
            {"vllm": False, "eval_freq": 10, "workers": 0},
            {"vllm": True, "eval_freq": 50, "workers": 0},
        ]
        
        for i, config in enumerate(configs):
            logger.info(f"Testing config {i+1}: vLLM={config['vllm']}, eval_freq={config['eval_freq']}")
            
            try:
                with timeout(180):
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Create training arguments
                        training_args = TrainingArguments(
                            output_dir=temp_dir,
                            per_device_train_batch_size=2,
                            per_device_eval_batch_size=2,
                            gradient_accumulation_steps=1,
                            logging_steps=5,
                            eval_strategy="steps",
                            eval_steps=config['eval_freq'],
                            save_strategy="steps",
                            save_steps=config['eval_freq'],
                            save_total_limit=1,
                            num_train_epochs=0.01,
                            dataloader_num_workers=config['workers'],
                            dataloader_pin_memory=False,
                            dataloader_persistent_workers=False,
                            remove_unused_columns=False,
                            report_to="none",
                            bf16=torch.cuda.is_available(),
                        )
                        
                        # Create GRPO config
                        grpo_config = GRPOConfig(
                            **training_args.to_dict(),
                            max_completion_length=512,
                            num_generations=4,
                            use_vllm=config['vllm'],
                            generation_batch_size=8,
                        )
                        
                        # Create trainer
                        trainer = GRPOTrainer(
                            model=model,
                            reward_funcs=reward_functions,
                            args=grpo_config,
                            train_dataset=mock_dataset,
                            eval_dataset=mock_dataset,
                            processing_class=tokenizer,
                        )
                        
                        logger.info(f"✅ Config {i+1} successful")
                        
                        # Clean up
                        del trainer
                        torch.cuda.empty_cache()
                        
            except TimeoutError:
                logger.error(f"❌ Config {i+1} timed out during creation")
                return False
            except Exception as e:
                logger.error(f"❌ Config {i+1} failed: {e}")
                return False
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        logger.info("✅ All GRPOTrainer configurations successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ GRPOTrainer creation failed: {e}")
        return False


def test_evaluation_step():
    """Test a single evaluation step to isolate the freeze."""
    logger.info("="*60)
    logger.info("🧪 Testing Evaluation Step")
    logger.info("="*60)
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from trl import GRPOConfig, GRPOTrainer
    from src.rewards.openr1_rewards import get_reward_funcs
    from src.config.grpo_config import GRPOScriptArguments
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        logger.info("Setting up evaluation test...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            use_cache=False
        )
        
        # Create small dataset
        mock_dataset = [
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
            {"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "6"}]},
        ]
        
        # Create simple reward functions
        script_args = GRPOScriptArguments(reward_funcs=["format"])
        reward_functions = get_reward_funcs(script_args)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create training arguments with minimal evaluation
            training_args = TrainingArguments(
                output_dir=temp_dir,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                gradient_accumulation_steps=1,
                logging_steps=2,
                eval_strategy="steps",
                eval_steps=4,  # Evaluate after 4 steps
                save_strategy="steps",
                save_steps=4,
                save_total_limit=1,
                num_train_epochs=0.1,
                dataloader_num_workers=0,  # No multiprocessing
                dataloader_pin_memory=False,
                dataloader_persistent_workers=False,
                remove_unused_columns=False,
                report_to="none",
                bf16=torch.cuda.is_available(),
                max_steps=10,  # Limit total steps
            )
            
            # Create GRPO config without vLLM
            grpo_config = GRPOConfig(
                **training_args.to_dict(),
                max_completion_length=256,
                num_generations=2,  # Minimal generations
                use_vllm=False,
                generation_batch_size=4,
            )
            
            # Create trainer
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=reward_functions,
                args=grpo_config,
                train_dataset=mock_dataset,
                eval_dataset=mock_dataset,
                processing_class=tokenizer,
            )
            
            logger.info("Starting training with evaluation...")
            
            # Train with timeout to catch freeze
            try:
                with timeout(300):  # 5 minute timeout
                    trainer.train()
                    logger.info("✅ Training completed successfully")
                    return True
            except TimeoutError:
                logger.error("❌ Training timed out - likely frozen during evaluation")
                return False
            except Exception as e:
                logger.error(f"❌ Training failed: {e}")
                return False
    
    except Exception as e:
        logger.error(f"❌ Evaluation test setup failed: {e}")
        return False


def test_reward_function_isolation():
    """Test reward functions in isolation to identify slow ones."""
    logger.info("="*60)
    logger.info("🧪 Testing Reward Function Isolation")
    logger.info("="*60)
    
    from src.rewards.openr1_rewards import get_reward_funcs
    from src.config.grpo_config import GRPOScriptArguments
    
    # Mock completions for testing
    mock_completions = [
        [{"content": "<think>\nStep 1: Calculate 2+2\nStep 2: The answer is 4\n</think>\n<answer>4</answer>"}],
        [{"content": "<think>\nFirst, I need to solve this.\nSecond, the answer is 6.\n</think>\n<answer>6</answer>"}],
        [{"content": "Invalid format without tags"}],
    ]
    
    mock_solution = ["4", "6", "8"]
    mock_target = ["4", "6", "8"]
    mock_nums = [[2, 2], [3, 3], [4, 4]]
    
    reward_functions_to_test = [
        "format",
        "equation",
        "accuracy",
        "reasoning_steps",
    ]
    
    for func_name in reward_functions_to_test:
        logger.info(f"Testing {func_name} reward function...")
        
        try:
            script_args = GRPOScriptArguments(
                reward_funcs=[func_name],
                cosine_min_value_wrong=-0.5,
                cosine_max_value_wrong=-0.1,
                cosine_min_value_correct=0.8,
                cosine_max_value_correct=1.0,
                cosine_max_len=1000,
                repetition_n_grams=3,
                repetition_max_penalty=-0.1,
                code_language="python",
                max_completion_len=512,
                soft_punish_cache=50,
            )
            
            reward_functions = get_reward_funcs(script_args)
            reward_func = reward_functions[0]
            
            # Test with timeout
            start_time = time.time()
            try:
                with timeout(30):
                    results = reward_func(
                        mock_completions,
                        solution=mock_solution,
                        target=mock_target,
                        nums=mock_nums
                    )
                    elapsed = time.time() - start_time
                    logger.info(f"✅ {func_name}: {elapsed:.2f}s - Results: {results}")
            except TimeoutError:
                logger.error(f"❌ {func_name}: Timed out after 30s")
                return False
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {func_name}: {elapsed:.2f}s - Error: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {func_name}: Setup failed: {e}")
            return False
    
    logger.info("✅ All reward functions tested successfully")
    return True


def main():
    """Main test runner."""
    logger.info("🚀 Starting GRPO Evaluation Freeze Minimal Test")
    logger.info("="*60)
    
    # Run tests in order
    tests = [
        ("Basic Components", test_basic_components),
        ("Reward Function Isolation", test_reward_function_isolation),
        ("GRPOTrainer Creation", test_grpo_trainer_creation),
        ("Evaluation Step", test_evaluation_step),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                logger.info(f"✅ {test_name} PASSED")
                passed += 1
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")
    
    logger.info("="*60)
    logger.info(f"🏁 FINAL RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("✅ All tests passed - issue may be configuration-specific")
    else:
        logger.error(f"❌ {total - passed} tests failed - issue likely in failed components")
    
    return passed == total


if __name__ == "__main__":
    main()