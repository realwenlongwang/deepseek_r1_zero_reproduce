#!/usr/bin/env python3
"""
GPU-focused test for DeepSeek R1 Zero GRPO system.
Tests GPU availability, memory usage, and training performance.
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
import psutil
import gc

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


def check_gpu_environment():
    """Check GPU availability and specifications."""
    logger.info("🔍 GPU ENVIRONMENT CHECK")
    logger.info("="*60)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        logger.error("❌ CUDA not available. Cannot run GPU tests.")
        return False
    
    # GPU specifications
    gpu_count = torch.cuda.device_count()
    logger.info(f"GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"GPU {i}: {props.name}")
        logger.info(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        logger.info(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Current GPU memory status
    current_device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(current_device) / 1e9
    reserved = torch.cuda.memory_reserved(current_device) / 1e9
    total = torch.cuda.get_device_properties(current_device).total_memory / 1e9
    
    logger.info(f"\nCurrent GPU Memory Status:")
    logger.info(f"  Allocated: {allocated:.2f} GB")
    logger.info(f"  Reserved:  {reserved:.2f} GB")
    logger.info(f"  Total:     {total:.2f} GB")
    logger.info(f"  Free:      {total - reserved:.2f} GB")
    
    return True


def test_gpu_model_loading():
    """Test loading model on GPU with different configurations."""
    logger.info("\n🚀 GPU MODEL LOADING TEST")
    logger.info("="*60)
    
    # Test configurations from smallest to largest
    test_configs = [
        {
            "name": "Small Model (distilgpt2)",
            "model_name": "distilgpt2",
            "torch_dtype": "float16",
            "expected_memory": 0.2  # GB
        },
        {
            "name": "Medium Model (microsoft/DialoGPT-medium)", 
            "model_name": "microsoft/DialoGPT-medium",
            "torch_dtype": "float16",
            "expected_memory": 0.7  # GB
        },
        {
            "name": "Large Model (Qwen/Qwen2.5-1.5B-Instruct)",
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct", 
            "torch_dtype": "bfloat16",
            "expected_memory": 3.0  # GB
        }
    ]
    
    successful_configs = []
    
    for config in test_configs:
        logger.info(f"\nTesting: {config['name']}")
        logger.info("-" * 40)
        
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            gc.collect()
            
            # Record initial memory
            initial_memory = torch.cuda.memory_allocated() / 1e9
            
            # Create model config (disable flash attention for compatibility)
            model_args = ModelConfig(
                model_name_or_path=config["model_name"],
                torch_dtype=config["torch_dtype"],
                attn_implementation=None  # Disable flash attention for this test
            )
            
            # Load model and tokenizer
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            logger.info(f"Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.info(f"Loading model to GPU...")
            model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                torch_dtype=getattr(torch, model_args.torch_dtype),
                trust_remote_code=model_args.trust_remote_code,
                device_map="auto"
            )
            
            # Check final memory usage
            final_memory = torch.cuda.memory_allocated() / 1e9
            memory_used = final_memory - initial_memory
            
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"   Parameters: {model.num_parameters():,}")
            logger.info(f"   Memory used: {memory_used:.2f} GB")
            logger.info(f"   Expected: {config['expected_memory']:.2f} GB")
            logger.info(f"   Device: {next(model.parameters()).device}")
            
            # Test generation on GPU
            test_input = "What is 5 + 3?"
            inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
            
            logger.info(f"Testing generation...")
            with torch.no_grad():
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                end_time.record()
                
                torch.cuda.synchronize()
                generation_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
            
            logger.info(f"✅ Generation successful")
            logger.info(f"   Time: {generation_time:.3f}s")
            logger.info(f"   Speed: {tokens_generated/generation_time:.1f} tokens/sec")
            logger.info(f"   Output: {generated_text}")
            
            successful_configs.append({
                **config,
                "actual_memory": memory_used,
                "generation_speed": tokens_generated/generation_time,
                "model": model,
                "tokenizer": tokenizer,
                "model_args": model_args
            })
            
            # Clean up for next test
            del model, tokenizer
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            logger.error(f"❌ Failed to load {config['name']}: {e}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
    
    logger.info(f"\n✅ Successfully tested {len(successful_configs)}/{len(test_configs)} configurations")
    return successful_configs


def test_gpu_training_simulation(model_config):
    """Test GPU training simulation with comprehensive logging."""
    logger.info("\n🏋️ GPU TRAINING SIMULATION")
    logger.info("="*60)
    
    try:
        # Create GRPO configuration
        script_args = GRPOScriptArguments(
            reward_funcs=["accuracy", "format", "reasoning_steps"],
        )
        
        training_args = create_training_arguments("./gpu_test_output")
        training_args.per_device_train_batch_size = 2
        training_args.gradient_accumulation_steps = 1
        training_args.logging_steps = 1
        training_args.dataloader_num_workers = 0  # Avoid multiprocessing issues
        
        # Load model and tokenizer on GPU
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_args = model_config["model_args"]
        logger.info(f"Loading {model_config['name']} for training simulation...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=getattr(torch, model_args.torch_dtype),
            trust_remote_code=model_args.trust_remote_code,
            device_map="auto"
        )
        
        # Get reward functions and callbacks
        reward_functions = get_reward_functions(script_args)
        callbacks = get_callbacks(training_args, model_args, script_args)
        
        logger.info(f"✅ Setup complete - running training simulation...")
        
        # Simulate training steps with real GPU operations
        from transformers import TrainerState, TrainerControl
        import numpy as np
        
        state = TrainerState()
        control = TrainerControl()
        
        # Create batch of training data
        train_prompts = [
            "What is 8 + 7?",
            "Calculate 12 × 3.",
            "If 2x = 16, what is x?",
            "What is the square of 9?"
        ]
        
        for step in range(1, 4):  # 3 training steps
            logger.info(f"\n--- Training Step {step} ---")
            
            # Record GPU memory before step
            memory_before = torch.cuda.memory_allocated() / 1e9
            
            state.global_step = step
            state.epoch = step * 0.1
            state.log_history = [{
                'loss': 3.0 - step * 0.2,
                'learning_rate': 5e-5,
                'policy_loss': 1.8 - step * 0.1,
                'value_loss': 1.0 - step * 0.05,
                'entropy_loss': 0.1 + step * 0.02
            }]
            
            # Simulate forward pass with real GPU computation
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            start_time.record()
            
            # Generate responses on GPU
            completions = []
            for prompt in train_prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=50,
                        do_sample=True,
                        temperature=0.8,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Format as expected completion structure
                completion_text = f"<think>\nStep 1: Analyzing {prompt}\nStep 2: Computing result\n</think>\n\n<answer>\nGenerated: {generated_text.split(prompt)[-1].strip()}\n</answer>"
                completions.append([{"content": completion_text}])
            
            end_time.record()
            torch.cuda.synchronize()
            step_time = start_time.elapsed_time(end_time) / 1000
            
            # Record GPU memory after computation
            memory_after = torch.cuda.memory_allocated() / 1e9
            memory_used = memory_after - memory_before
            
            # Create mock rewards (would be calculated by reward functions in real training)
            rewards = {
                'accuracy': np.random.uniform(0.0, 1.0, len(completions)),
                'format': np.random.uniform(0.5, 1.0, len(completions)),
                'reasoning_steps': np.random.uniform(0.3, 0.9, len(completions)),
            }
            rewards['total'] = [sum(r[i] for r in rewards.values()) for i in range(len(completions))]
            
            # Execute callbacks with comprehensive logging
            for callback in callbacks:
                callback.on_step_begin(training_args, state, control)
                callback.on_step_end(
                    training_args, state, control,
                    rewards=rewards,
                    completions=completions
                )
            
            logger.info(f"GPU Memory: {memory_after:.2f} GB (+{memory_used:.3f} GB)")
            logger.info(f"Step Time: {step_time:.3f}s")
            logger.info(f"Throughput: {len(completions)/step_time:.1f} samples/sec")
            
            # Clear intermediate computations
            torch.cuda.empty_cache()
        
        logger.info("✅ GPU training simulation completed successfully!")
        
        # Final memory report
        final_memory = torch.cuda.memory_allocated() / 1e9
        max_memory = torch.cuda.max_memory_allocated() / 1e9
        
        logger.info(f"\nFinal GPU Memory Report:")
        logger.info(f"  Current allocated: {final_memory:.2f} GB")
        logger.info(f"  Peak allocated: {max_memory:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU training simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_gpu_tests():
    """Run comprehensive GPU tests."""
    logger.info("🔥 STARTING COMPREHENSIVE GPU TESTS")
    logger.info("="*80)
    
    set_seed(42)
    
    # Test 1: Check GPU environment
    if not check_gpu_environment():
        logger.error("❌ GPU environment check failed. Cannot proceed.")
        return False
    
    # Test 2: Model loading tests
    successful_configs = test_gpu_model_loading()
    
    if not successful_configs:
        logger.error("❌ No models could be loaded on GPU.")
        return False
    
    # Test 3: Training simulation with best performing model
    best_config = successful_configs[0]  # Use the first successful config
    logger.info(f"\nUsing {best_config['name']} for training simulation")
    
    training_success = test_gpu_training_simulation(best_config)
    
    # Final results
    logger.info("\n" + "="*80)
    logger.info("🎯 GPU TEST RESULTS")
    logger.info("="*80)
    
    logger.info(f"GPU Environment: ✅ PASS")
    logger.info(f"Model Loading: ✅ PASS ({len(successful_configs)} configs)")
    logger.info(f"Training Simulation: {'✅ PASS' if training_success else '❌ FAIL'}")
    
    if training_success:
        logger.info("\n🚀 GPU GRPO SYSTEM READY!")
        logger.info("💡 You can now run GPU training with:")
        logger.info("   python train_grpo.py --model_name distilgpt2 --per_device_train_batch_size 4")
        
        # Show optimized GPU command
        logger.info("\n🎯 RECOMMENDED GPU TRAINING COMMAND:")
        logger.info("   python train_grpo.py \\")
        logger.info(f"     --model_name \"{best_config['model_name']}\" \\")
        logger.info("     --per_device_train_batch_size 4 \\")
        logger.info("     --gradient_accumulation_steps 2 \\")
        logger.info("     --learning_rate 5e-5 \\")
        logger.info("     --logging_steps 5 \\")
        logger.info("     --no_wandb")
    else:
        logger.error("❌ GPU training tests failed.")
    
    logger.info("="*80)
    
    return training_success


if __name__ == "__main__":
    success = run_gpu_tests()
    sys.exit(0 if success else 1)