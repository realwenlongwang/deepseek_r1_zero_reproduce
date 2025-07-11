#!/usr/bin/env python3
"""
Test GRPOTrainer without vLLM to isolate the padding side issue.
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

def test_grpo_without_vllm():
    """Test GRPOTrainer without vLLM to isolate the tokenizer issue."""
    print("="*80)
    print("GRPO WITHOUT VLLM TEST")
    print("="*80)
    
    # Use smaller model for testing
    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"Testing with model: {model_name}")
    
    try:
        # Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"   Initial padding_side: {tokenizer.padding_side}")
        
        # Configure tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        print(f"   Configured padding_side: {tokenizer.padding_side}")
        
        # Load model
        print("\n2. Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="cuda:0"  # Use single GPU
        )
        print(f"   Model loaded successfully")
        
        # Create simple dataset
        print("\n3. Creating dataset...")
        simple_dataset = [
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "<think>2+2 is basic addition</think><answer>4</answer>"}]},
            {"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "<think>3+3 is basic addition</think><answer>6</answer>"}]},
            {"messages": [{"role": "user", "content": "What is 4+4?"}, {"role": "assistant", "content": "<think>4+4 is basic addition</think><answer>8</answer>"}]},
        ]
        print(f"   Dataset created with {len(simple_dataset)} samples")
        
        # Create reward functions
        print("\n4. Creating reward functions...")
        script_args = GRPOScriptArguments(reward_funcs=["format"])
        reward_functions = get_reward_funcs(script_args)
        print(f"   Created {len(reward_functions)} reward functions")
        
        # Create GRPOConfig WITHOUT vLLM
        print("\n5. Creating GRPOConfig (without vLLM)...")
        grpo_config = GRPOConfig(
            output_dir="./test_output",
            per_device_train_batch_size=8,  # Effective batch size must be divisible by num_generations
            gradient_accumulation_steps=1,
            max_completion_length=256,
            num_generations=8,
            use_vllm=False,  # DISABLE vLLM
            bf16=True,
            remove_unused_columns=False,
            logging_steps=1,
            save_steps=1000,
            eval_steps=1000,
            save_strategy="steps",
            eval_strategy="no",
        )
        print(f"   GRPOConfig created successfully (vLLM disabled)")
        
        # Check tokenizer before GRPOTrainer
        print(f"\n6. Before GRPOTrainer creation:")
        print(f"   tokenizer.padding_side: {tokenizer.padding_side}")
        print(f"   tokenizer.pad_token: {tokenizer.pad_token}")
        print(f"   tokenizer.pad_token_id: {tokenizer.pad_token_id}")
        
        # Create GRPOTrainer
        print("\n7. Creating GRPOTrainer...")
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=simple_dataset,
            processing_class=tokenizer,
        )
        print(f"   GRPOTrainer created successfully")
        
        # Check tokenizer after GRPOTrainer
        print(f"\n8. After GRPOTrainer creation:")
        print(f"   original tokenizer.padding_side: {tokenizer.padding_side}")
        
        # Check trainer's internal tokenizer
        if hasattr(grpo_trainer, 'tokenizer') and grpo_trainer.tokenizer:
            print(f"   trainer.tokenizer.padding_side: {grpo_trainer.tokenizer.padding_side}")
            print(f"   trainer.tokenizer.pad_token: {grpo_trainer.tokenizer.pad_token}")
            print(f"   trainer.tokenizer is same object: {grpo_trainer.tokenizer is tokenizer}")
        
        if hasattr(grpo_trainer, 'processing_class') and grpo_trainer.processing_class:
            print(f"   trainer.processing_class.padding_side: {grpo_trainer.processing_class.padding_side}")
            print(f"   trainer.processing_class.pad_token: {grpo_trainer.processing_class.pad_token}")
            print(f"   trainer.processing_class is same object: {grpo_trainer.processing_class is tokenizer}")
            
        # Check if there are any other tokenizer references
        print(f"\n9. Checking for other tokenizer references...")
        for attr_name in dir(grpo_trainer):
            if 'token' in attr_name.lower() and not attr_name.startswith('_'):
                attr = getattr(grpo_trainer, attr_name)
                if hasattr(attr, 'padding_side'):
                    print(f"   {attr_name}.padding_side: {attr.padding_side}")
        
        # Try to trigger the error by doing a mini training step
        print(f"\n10. Testing mini training step...")
        try:
            # Apply safeguards
            print("   Applying safeguards...")
            if hasattr(grpo_trainer, 'tokenizer') and grpo_trainer.tokenizer:
                grpo_trainer.tokenizer.padding_side = 'left'
                print(f"   Set trainer.tokenizer.padding_side to: {grpo_trainer.tokenizer.padding_side}")
                
            if hasattr(grpo_trainer, 'processing_class') and grpo_trainer.processing_class:
                grpo_trainer.processing_class.padding_side = 'left'
                print(f"   Set trainer.processing_class.padding_side to: {grpo_trainer.processing_class.padding_side}")
            
            # Try to run a single training step
            print("   Attempting single training step...")
            
            # This is the risky part - try to trigger the error
            grpo_trainer.train()
            
        except Exception as e:
            print(f"   ERROR during training: {e}")
            if "padding_side='right'" in str(e):
                print("   ✓ REPRODUCED: Flash Attention padding side error!")
                
                # Debug: check all tokenizer states at error time
                print("   Debugging tokenizer states at error time:")
                print(f"     original tokenizer.padding_side: {tokenizer.padding_side}")
                if hasattr(grpo_trainer, 'tokenizer') and grpo_trainer.tokenizer:
                    print(f"     trainer.tokenizer.padding_side: {grpo_trainer.tokenizer.padding_side}")
                if hasattr(grpo_trainer, 'processing_class') and grpo_trainer.processing_class:
                    print(f"     trainer.processing_class.padding_side: {grpo_trainer.processing_class.padding_side}")
            else:
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

def test_tokenizer_cloning():
    """Test how tokenizer cloning/copying affects padding_side."""
    print("\n" + "="*80)
    print("TOKENIZER CLONING TEST")
    print("="*80)
    
    model_name = "Qwen/Qwen2.5-0.5B"
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.padding_side = 'left'
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print(f"1. Original tokenizer:")
        print(f"   padding_side: {tokenizer.padding_side}")
        print(f"   pad_token: {tokenizer.pad_token}")
        
        # Test different ways of copying/cloning
        print(f"\n2. Testing different copying methods:")
        
        # Method 1: Direct assignment
        tokenizer2 = tokenizer
        print(f"   Direct assignment - padding_side: {tokenizer2.padding_side}")
        print(f"   Is same object: {tokenizer2 is tokenizer}")
        
        # Method 2: Using from_pretrained again
        tokenizer3 = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"   Fresh from_pretrained - padding_side: {tokenizer3.padding_side}")
        print(f"   Is same object: {tokenizer3 is tokenizer}")
        
        # Method 3: Using copy module
        import copy
        tokenizer4 = copy.copy(tokenizer)
        print(f"   copy.copy - padding_side: {tokenizer4.padding_side}")
        print(f"   Is same object: {tokenizer4 is tokenizer}")
        
        tokenizer5 = copy.deepcopy(tokenizer)
        print(f"   copy.deepcopy - padding_side: {tokenizer5.padding_side}")
        print(f"   Is same object: {tokenizer5 is tokenizer}")
        
        # Method 4: Simulating what might happen in TRL
        print(f"\n3. Testing TRL-like operations:")
        
        # Create a function that might modify tokenizer internally
        def simulate_trl_initialization(model, tokenizer):
            # This simulates what might happen in TRL
            
            # Check if TRL creates a new tokenizer from the model
            if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
                # This might be what TRL does internally
                internal_tokenizer = AutoTokenizer.from_pretrained(
                    model.config.name_or_path, 
                    trust_remote_code=True
                )
                print(f"   TRL-style internal tokenizer - padding_side: {internal_tokenizer.padding_side}")
                return internal_tokenizer
            return tokenizer
        
        # Load model for testing
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda:0"
        )
        
        trl_tokenizer = simulate_trl_initialization(model, tokenizer)
        print(f"   After TRL simulation - padding_side: {trl_tokenizer.padding_side}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting GRPOTrainer test without vLLM...")
    
    # Run the tests
    test_grpo_without_vllm()
    test_tokenizer_cloning()
    
    print("\n" + "="*80)
    print("TESTS COMPLETED")
    print("="*80)