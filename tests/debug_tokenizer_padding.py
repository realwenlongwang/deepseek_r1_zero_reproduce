#!/usr/bin/env python3
"""
Debug script to test tokenizer padding side configuration and Flash Attention compatibility.
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
from src.config.grpo_config import GRPOScriptArguments, ModelConfig
from src.rewards.openr1_rewards import get_reward_funcs

def test_tokenizer_padding_configuration():
    """Test tokenizer padding configuration at different stages."""
    print("="*80)
    print("TOKENIZER PADDING CONFIGURATION TEST")
    print("="*80)
    
    # Test with a small model first
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Testing with model: {model_name}")
    
    # 1. Load tokenizer and check initial configuration
    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"   Initial padding_side: {tokenizer.padding_side}")
    
    # 2. Set padding side to left (as in our code)
    print("\n2. Setting padding_side to 'left'...")
    tokenizer.padding_side = 'left'
    print(f"   After setting: {tokenizer.padding_side}")
    
    # 3. Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"   Set pad_token to: {tokenizer.pad_token}")
    
    # 4. Test simple tokenization with batching
    print("\n3. Testing tokenization with batching...")
    test_texts = [
        "Hello world",
        "This is a longer test sentence to see padding behavior",
        "Short"
    ]
    
    try:
        # Test with current padding side
        encoded = tokenizer(test_texts, padding=True, return_tensors="pt")
        print(f"   Tokenization successful with padding_side='{tokenizer.padding_side}'")
        print(f"   Input shape: {encoded['input_ids'].shape}")
        print(f"   First few tokens of batch:")
        for i, ids in enumerate(encoded['input_ids'][:2]):
            print(f"     Sample {i}: {ids[:10].tolist()}")
    except Exception as e:
        print(f"   ERROR during tokenization: {e}")
    
    return tokenizer

def test_model_with_flash_attention():
    """Test model loading with Flash Attention."""
    print("\n" + "="*80)
    print("MODEL WITH FLASH ATTENTION TEST")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Testing model: {model_name}")
    
    try:
        # Load model with Flash Attention
        print("\n1. Loading model with flash_attention_2...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        print(f"   Model loaded successfully")
        print(f"   Model type: {model.config.model_type}")
        print(f"   Attention implementation: {getattr(model.config, '_attn_implementation', 'default')}")
        
        return model
    except Exception as e:
        print(f"   ERROR loading model: {e}")
        return None

def test_grpo_trainer_tokenizer_handling():
    """Test how GRPOTrainer handles tokenizer configuration."""
    print("\n" + "="*80)
    print("GRPO TRAINER TOKENIZER HANDLING TEST")
    print("="*80)
    
    # Load tokenizer and model
    tokenizer = test_tokenizer_padding_configuration()
    model = test_model_with_flash_attention()
    
    if model is None:
        print("Skipping GRPOTrainer test due to model loading failure")
        return
    
    # Create minimal dataset
    minimal_dataset = [
        {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
        {"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "6"}]}
    ]
    
    # Create minimal reward functions
    script_args = GRPOScriptArguments(reward_funcs=["format"])
    reward_functions = get_reward_funcs(script_args)
    
    print(f"\n1. Before GRPOTrainer initialization:")
    print(f"   tokenizer.padding_side: {tokenizer.padding_side}")
    
    try:
        # Create GRPOConfig
        grpo_config = GRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_completion_length=128,
            num_generations=2,  # Reduced for testing
            use_vllm=False,  # Disable vLLM for testing
        )
        
        # Create GRPOTrainer 
        print("\n2. Creating GRPOTrainer...")
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=minimal_dataset,
            processing_class=tokenizer,
        )
        
        print(f"   GRPOTrainer created successfully")
        print(f"   Original tokenizer.padding_side: {tokenizer.padding_side}")
        
        # Check if trainer has its own tokenizer
        if hasattr(grpo_trainer, 'tokenizer'):
            print(f"   Trainer tokenizer.padding_side: {grpo_trainer.tokenizer.padding_side}")
        if hasattr(grpo_trainer, 'processing_class'):
            print(f"   Trainer processing_class.padding_side: {grpo_trainer.processing_class.padding_side}")
            
    except Exception as e:
        print(f"   ERROR creating GRPOTrainer: {e}")
        import traceback
        traceback.print_exc()

def test_flash_attention_batching():
    """Test Flash Attention batching with different padding sides."""
    print("\n" + "="*80)
    print("FLASH ATTENTION BATCHING TEST")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        test_prompts = [
            "What is 2+2?",
            "Explain the concept of machine learning in simple terms.",
            "Hello"
        ]
        
        # Test with right padding (should fail with Flash Attention)
        print("\n1. Testing with padding_side='right'...")
        tokenizer.padding_side = 'right'
        try:
            inputs = tokenizer(test_prompts, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Try generation (this should trigger the error)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id
                )
            print("   SUCCESS: Generation completed with right padding")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            if "padding_side='right'" in str(e):
                print("   ✓ Confirmed: Flash Attention requires left padding")
        
        # Test with left padding (should work)
        print("\n2. Testing with padding_side='left'...")
        tokenizer.padding_side = 'left'
        try:
            inputs = tokenizer(test_prompts, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id
                )
            print("   SUCCESS: Generation completed with left padding")
            print("   ✓ Confirmed: Left padding works with Flash Attention")
            
        except Exception as e:
            print(f"   ERROR: {e}")
            
    except Exception as e:
        print(f"Setup error: {e}")

if __name__ == "__main__":
    print("Starting tokenizer padding diagnostic tests...")
    
    # Run all tests
    test_tokenizer_padding_configuration()
    test_model_with_flash_attention()
    test_grpo_trainer_tokenizer_handling()
    test_flash_attention_batching()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC TESTS COMPLETED")
    print("="*80)