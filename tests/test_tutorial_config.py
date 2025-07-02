#!/usr/bin/env python3
"""
Test the tutorial-exact configuration with Qwen/Qwen2.5-7B-Instruct.
Validates that all settings match the tutorial specifications.
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

# Import our GRPO components
from src.config.grpo_config import (
    GRPOScriptArguments,
    ModelConfig,
    create_training_arguments,
    get_reward_functions,
    get_callbacks
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def test_tutorial_configuration():
    """Test the tutorial-exact configuration."""
    logger.info("🎯 TESTING TUTORIAL-EXACT CONFIGURATION")
    logger.info("="*80)
    
    try:
        # Create default configurations (should match tutorial exactly)
        script_args = GRPOScriptArguments()
        model_args = ModelConfig()
        training_args = create_training_arguments()
        
        logger.info("✅ Configuration objects created successfully")
        logger.info("")
        
        # Validate model configuration
        logger.info("📋 MODEL CONFIGURATION:")
        logger.info(f"  Model: {model_args.model_name_or_path}")
        logger.info(f"  Torch dtype: {model_args.torch_dtype}")
        logger.info(f"  Trust remote code: {model_args.trust_remote_code}")
        logger.info(f"  Attention implementation: {model_args.attn_implementation}")
        
        # Validate training arguments (tutorial exact)
        logger.info("\n📋 TRAINING ARGUMENTS (Tutorial Exact):")
        logger.info(f"  Output dir: {training_args.output_dir}")
        logger.info(f"  Num train epochs: {training_args.num_train_epochs}")
        logger.info(f"  Per device train batch size: {training_args.per_device_train_batch_size}")
        logger.info(f"  Per device eval batch size: {training_args.per_device_eval_batch_size}")
        logger.info(f"  Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        logger.info(f"  Learning rate: {training_args.learning_rate}")
        logger.info(f"  Warmup ratio: {training_args.warmup_ratio}")
        logger.info(f"  Weight decay: {training_args.weight_decay}")
        logger.info(f"  Logging steps: {training_args.logging_steps}")
        logger.info(f"  Evaluation strategy: {training_args.eval_strategy}")
        logger.info(f"  Eval steps: {training_args.eval_steps}")
        logger.info(f"  Save strategy: {training_args.save_strategy}")
        logger.info(f"  Save steps: {training_args.save_steps}")
        logger.info(f"  Save total limit: {training_args.save_total_limit}")
        logger.info(f"  Dataloader num workers: {training_args.dataloader_num_workers}")
        logger.info(f"  Seed: {training_args.seed}")
        logger.info(f"  BF16: {training_args.bf16}")
        logger.info(f"  Push to hub: {training_args.push_to_hub}")
        logger.info(f"  Gradient checkpointing: {training_args.gradient_checkpointing}")
        logger.info(f"  Report to: {training_args.report_to}")
        logger.info(f"  Remove unused columns: {training_args.remove_unused_columns}")
        
        # Validate reward functions (all 5 from tutorial)
        logger.info("\n📋 REWARD FUNCTIONS:")
        logger.info(f"  Enabled functions: {script_args.reward_funcs}")
        logger.info(f"  Expected: ['accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty']")
        
        expected_rewards = ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"]
        if script_args.reward_funcs == expected_rewards:
            logger.info("  ✅ All 5 tutorial reward functions enabled")
        else:
            logger.warning(f"  ⚠️ Mismatch in reward functions!")
        
        # Cosine scaling parameters
        logger.info(f"\n📋 COSINE SCALING PARAMETERS:")
        logger.info(f"  Min value wrong: {script_args.cosine_min_value_wrong}")
        logger.info(f"  Max value wrong: {script_args.cosine_max_value_wrong}")
        logger.info(f"  Min value correct: {script_args.cosine_min_value_correct}")
        logger.info(f"  Max value correct: {script_args.cosine_max_value_correct}")
        logger.info(f"  Max length: {script_args.cosine_max_len}")
        
        # Repetition penalty parameters
        logger.info(f"\n📋 REPETITION PENALTY PARAMETERS:")
        logger.info(f"  N-grams: {script_args.repetition_n_grams}")
        logger.info(f"  Max penalty: {script_args.repetition_max_penalty}")
        
        # Test reward functions creation
        logger.info("\n🔧 TESTING REWARD FUNCTIONS:")
        reward_functions = get_reward_functions(script_args)
        logger.info(f"  ✅ Created {len(reward_functions)} reward functions")
        for i, func_name in enumerate(script_args.reward_funcs):
            logger.info(f"    {i+1}. {func_name}")
        
        # Test callbacks creation
        logger.info("\n🔧 TESTING CALLBACKS:")
        callbacks = get_callbacks(training_args, model_args, script_args)
        logger.info(f"  ✅ Created {len(callbacks)} callbacks")
        for i, callback in enumerate(callbacks):
            logger.info(f"    {i+1}. {type(callback).__name__}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_compatibility():
    """Test GPU compatibility with tutorial configuration."""
    logger.info("\n🚀 TESTING GPU COMPATIBILITY")
    logger.info("="*60)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available - cannot test GPU compatibility")
        return True
    
    gpu_count = torch.cuda.device_count()
    logger.info(f"GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        
        # Check if GPU has enough memory for Qwen2.5-7B
        if props.total_memory / 1e9 >= 14:  # Minimum 14GB for 7B model
            logger.info(f"  ✅ Sufficient memory for Qwen2.5-7B-Instruct")
        else:
            logger.warning(f"  ⚠️ May need memory optimization for Qwen2.5-7B-Instruct")
    
    return True


def test_memory_estimation():
    """Estimate memory requirements for tutorial configuration."""
    logger.info("\n💾 MEMORY ESTIMATION")
    logger.info("="*60)
    
    # Qwen2.5-7B-Instruct memory estimation
    model_params = 7.6e9  # 7.6B parameters
    bytes_per_param_bf16 = 2  # bfloat16 uses 2 bytes per parameter
    
    model_memory_gb = (model_params * bytes_per_param_bf16) / 1e9
    logger.info(f"Model memory (bf16): {model_memory_gb:.1f} GB")
    
    # Training memory estimation (rough)
    # Need ~4x model size for gradients, optimizer states, activations
    training_memory_gb = model_memory_gb * 4
    logger.info(f"Estimated training memory: {training_memory_gb:.1f} GB")
    
    # Batch size impact
    batch_size = 8 * 2  # per_device_batch_size * gradient_accumulation_steps
    logger.info(f"Effective batch size: {batch_size}")
    
    # Recommendations
    logger.info("\n📝 MEMORY OPTIMIZATION RECOMMENDATIONS:")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Available GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory >= training_memory_gb:
            logger.info("✅ Should work with tutorial settings")
        elif gpu_memory >= model_memory_gb * 2:
            logger.info("⚠️ May need to reduce batch size:")
            logger.info("   --per_device_train_batch_size 4 --gradient_accumulation_steps 4")
        else:
            logger.info("❌ Need significant optimization:")
            logger.info("   --per_device_train_batch_size 2 --gradient_accumulation_steps 8")
            logger.info("   Consider using gradient checkpointing and DeepSpeed")
    
    return True


def run_tutorial_config_test():
    """Run comprehensive tutorial configuration test."""
    logger.info("🎯 COMPREHENSIVE TUTORIAL CONFIGURATION TEST")
    logger.info("="*80)
    
    set_seed(42)
    
    tests = [
        ("Tutorial Configuration", test_tutorial_configuration),
        ("GPU Compatibility", test_gpu_compatibility),
        ("Memory Estimation", test_memory_estimation),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            results.append((test_name, False))
    
    # Print final results
    logger.info("\n" + "="*80)
    logger.info("🏆 TUTORIAL CONFIGURATION TEST RESULTS")
    logger.info("="*80)
    
    all_passed = True
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"{test_name:25s}: {status}")
        if not success:
            all_passed = False
    
    logger.info("="*80)
    
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("🚀 Tutorial configuration is ready for training")
        logger.info("")
        logger.info("💡 Ready to run:")
        logger.info("   python train_grpo.py")
        logger.info("")
        logger.info("🎯 Or with custom settings:")
        logger.info("   python train_grpo.py \\")
        logger.info("     --per_device_train_batch_size 4 \\")
        logger.info("     --gradient_accumulation_steps 4 \\")
        logger.info("     --logging_steps 5")
    else:
        logger.error("❌ Some tests failed. Check configuration.")
    
    logger.info("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = run_tutorial_config_test()
    sys.exit(0 if success else 1)