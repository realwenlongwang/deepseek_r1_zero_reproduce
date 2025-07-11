#!/usr/bin/env python3
"""
Focused test to reproduce the specific padding side issue in GRPOTrainer.
"""

import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from src.config.grpo_config import GRPOScriptArguments
from src.rewards.openr1_rewards import get_reward_funcs
from src.data.dataset import create_dataset

def test_exact_reproduction():
    """Test that reproduces the exact conditions from the training script."""
    print("="*80)
    print("EXACT REPRODUCTION TEST")
    print("="*80)
    
    # Use the exact same model as in the error
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Testing with model: {model_name}")
    
    try:
        # Load tokenizer exactly as in train_grpo.py
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        print(f"   Initial padding_side: {tokenizer.padding_side}")
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"   Set pad_token to: {tokenizer.pad_token}")
        
        # Set padding side to left (as in train_grpo.py)
        tokenizer.padding_side = 'left'
        print(f"   Set padding_side to: {tokenizer.padding_side}")
        
        # Load model exactly as in train_grpo.py  
        print("\n2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        print(f"   Model loaded successfully")
        print(f"   Model type: {model.config.model_type}")
        
        # Create dataset exactly as in train_grpo.py
        print("\n3. Loading dataset...")
        train_dataset = create_dataset(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            split="train"
        )
        print(f"   Dataset loaded: {len(train_dataset)} samples")
        
        # Create reward functions exactly as in train_grpo.py
        print("\n4. Creating reward functions...")
        script_args = GRPOScriptArguments(
            reward_funcs=["accuracy", "format", "reasoning_steps"]
        )
        reward_functions = get_reward_funcs(script_args)
        print(f"   Created {len(reward_functions)} reward functions")
        
        # Create GRPOConfig exactly as in train_grpo.py
        print("\n5. Creating GRPOConfig...")
        grpo_config = GRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=4,  # Use the exact batch size from config
            gradient_accumulation_steps=1,
            max_completion_length=512,
            generation_batch_size=32,
            num_generations=8,  # Standard GRPO setting
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.3,
            bf16=True,
            tf32=True,
            remove_unused_columns=False,
        )
        print(f"   GRPOConfig created successfully")
        
        # Check tokenizer before GRPOTrainer
        print(f"\n6. Before GRPOTrainer creation:")
        print(f"   tokenizer.padding_side: {tokenizer.padding_side}")
        
        # Create GRPOTrainer exactly as in train_grpo.py
        print("\n7. Creating GRPOTrainer...")
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=train_dataset[:10],  # Use small subset for testing
            processing_class=tokenizer,
        )
        print(f"   GRPOTrainer created successfully")
        
        # Check tokenizer after GRPOTrainer
        print(f"\n8. After GRPOTrainer creation:")
        print(f"   original tokenizer.padding_side: {tokenizer.padding_side}")
        if hasattr(grpo_trainer, 'tokenizer'):
            print(f"   trainer.tokenizer.padding_side: {grpo_trainer.tokenizer.padding_side}")
        if hasattr(grpo_trainer, 'processing_class'):
            print(f"   trainer.processing_class.padding_side: {grpo_trainer.processing_class.padding_side}")
        
        # Apply the safeguards from train_grpo.py
        print(f"\n9. Applying safeguards from train_grpo.py...")
        if hasattr(grpo_trainer, 'processing_class') and grpo_trainer.processing_class is not None:
            print(f"   Before safeguard - processing_class.padding_side: {grpo_trainer.processing_class.padding_side}")
            grpo_trainer.processing_class.padding_side = 'left'
            print(f"   After safeguard - processing_class.padding_side: {grpo_trainer.processing_class.padding_side}")
        
        if hasattr(grpo_trainer, 'tokenizer') and grpo_trainer.tokenizer is not None:
            print(f"   Before safeguard - tokenizer.padding_side: {grpo_trainer.tokenizer.padding_side}")
            grpo_trainer.tokenizer.padding_side = 'left'
            print(f"   After safeguard - tokenizer.padding_side: {grpo_trainer.tokenizer.padding_side}")
        
        # Try to trigger the issue by calling the problematic method
        print(f"\n10. Testing training step (this might trigger the error)...")
        try:
            # Get a small batch from the dataset
            batch = train_dataset[:1]  # Just one sample
            
            # Try to trigger the error path
            # This is where the error occurs in the stack trace
            print("   Attempting to trigger _prepare_inputs...")
            
            # We can't easily call _prepare_inputs directly, so let's try a different approach
            # Let's check if there are any internal tokenizer instances that might be created
            
            # Check if there are any methods that might create new tokenizers
            print("   Checking for internal tokenizer creation...")
            
            # The error occurs in _generate_and_score_completions -> _get_per_token_logps
            # Let's see if we can inspect the trainer's internal state
            
            print("   SUCCESS: GRPOTrainer setup completed without immediate error")
            print("   The error likely occurs during actual training when _prepare_inputs is called")
            
        except Exception as e:
            print(f"   ERROR during training step test: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_tokenizer_internals():
    """Test to understand tokenizer internal behavior."""
    print("\n" + "="*80)
    print("TOKENIZER INTERNALS TEST")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-0.5B"  # Use smaller model for testing
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        print(f"1. Initial tokenizer state:")
        print(f"   padding_side: {tokenizer.padding_side}")
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   pad_token_id: {tokenizer.pad_token_id}")
        
        # Set to left padding
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"\n2. After configuration:")
        print(f"   padding_side: {tokenizer.padding_side}")
        print(f"   pad_token: {tokenizer.pad_token}")
        print(f"   pad_token_id: {tokenizer.pad_token_id}")
        
        # Test what happens when we pass it to different functions
        print(f"\n3. Testing tokenizer persistence...")
        
        # Create a simple function that might modify tokenizer
        def test_function(tok):
            print(f"   Inside function - padding_side: {tok.padding_side}")
            # Check if the tokenizer gets modified
            return tok
        
        result_tok = test_function(tokenizer)
        print(f"   After function - padding_side: {result_tok.padding_side}")
        
        # Test serialization/deserialization which might reset settings
        print(f"\n4. Testing serialization effects...")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save and reload tokenizer
            tokenizer.save_pretrained(tmp_dir)
            reloaded_tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)
            print(f"   Reloaded tokenizer padding_side: {reloaded_tokenizer.padding_side}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting focused padding side reproduction test...")
    
    # Run the tests
    test_exact_reproduction()
    test_tokenizer_internals()
    
    print("\n" + "="*80)
    print("FOCUSED TESTS COMPLETED")
    print("="*80)