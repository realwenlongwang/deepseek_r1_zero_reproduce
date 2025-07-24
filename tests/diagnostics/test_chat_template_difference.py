#!/usr/bin/env python3
"""
Test to verify the chat template difference between standard and unsloth models.
"""

import sys
sys.path.append('src')

from src.config import ConfigManager
from train_grpo import setup_model_and_tokenizer

def test_chat_template_difference():
    """Test chat template differences between standard and unsloth models."""
    print("🔍 Testing Chat Template Difference")
    print("=" * 60)
    
    config_manager = ConfigManager('config.yaml', 'test')
    
    # Test message
    messages = [
        {"role": "system", "content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"},
        {"role": "user", "content": "Use the numbers 80, 88, 37, and 8 with basic arithmetic operations (+, -, ×, ÷) to reach the target number 38. You can use each number at most once. Show your reasoning step by step."},
        {"role": "assistant", "content": ""}
    ]
    
    print("Testing with sample messages:")
    print(f"System: {messages[0]['content'][:50]}...")
    print(f"User: {messages[1]['content'][:50]}...")
    print()
    
    # Test 1: Standard model
    print("1️⃣ Standard Model Chat Template")
    print("-" * 40)
    
    try:
        config_std = config_manager.load_config([
            '--model.unsloth.enabled', 'false',
            '--model.name', 'Qwen/Qwen2.5-0.5B-Instruct'
        ])
        
        model_std, tokenizer_std = setup_model_and_tokenizer(config_std)
        
        print(f"Has chat template: {tokenizer_std.chat_template is not None}")
        if tokenizer_std.chat_template:
            print(f"Chat template preview: {tokenizer_std.chat_template[:100]}...")
        
        # Apply chat template
        prompt_std = tokenizer_std.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        print(f"Generated prompt preview: {prompt_std[:200]}...")
        
        # Check for ChatML tokens
        has_chatml = '<|im_start|>' in prompt_std and '<|im_end|>' in prompt_std
        print(f"Uses ChatML format: {has_chatml}")
        
        del model_std
        
    except Exception as e:
        print(f"Error with standard model: {e}")
    
    print()
    
    # Test 2: Unsloth model
    print("2️⃣ Unsloth Model Chat Template") 
    print("-" * 40)
    
    try:
        config_unsloth = config_manager.load_config([
            '--model.unsloth.enabled', 'true',
            '--model.name', 'Qwen/Qwen2.5-0.5B-Instruct',
            '--model.unsloth.load_in_4bit', 'false'
        ])
        
        model_unsloth, tokenizer_unsloth = setup_model_and_tokenizer(config_unsloth)
        
        print(f"Has chat template: {tokenizer_unsloth.chat_template is not None}")
        if tokenizer_unsloth.chat_template:
            print(f"Chat template preview: {tokenizer_unsloth.chat_template[:100]}...")
        
        # Apply chat template
        prompt_unsloth = tokenizer_unsloth.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        print(f"Generated prompt preview: {prompt_unsloth[:200]}...")
        
        # Check for ChatML tokens
        has_chatml = '<|im_start|>' in prompt_unsloth and '<|im_end|>' in prompt_unsloth
        print(f"Uses ChatML format: {has_chatml}")
        
        del model_unsloth
        
    except Exception as e:
        print(f"Error with unsloth model: {e}")
    
    print()
    print("🎯 Summary:")
    print("Standard models preserve the original ChatML chat template with <|im_start|>/<|im_end|> tokens.")
    print("Unsloth models may fall back to a simpler template without special tokens.")
    print("This explains why the prompt format appears different between the two approaches.")

if __name__ == "__main__":
    test_chat_template_difference()