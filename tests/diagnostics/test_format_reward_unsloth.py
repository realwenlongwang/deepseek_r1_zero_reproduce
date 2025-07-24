#!/usr/bin/env python3
"""
Test script to verify the format reward issue when unsloth is enabled.
This test compares text generation between standard and unsloth models.
"""

import os
import sys
sys.path.append('src')

import torch
from src.config import ConfigManager
from src.rewards.openr1_rewards import format_reward
from train_grpo import setup_model_and_tokenizer

def test_format_reward_with_unsloth():
    """Test format reward with both standard and unsloth models."""
    print("🧪 Testing Format Reward with Unsloth")
    print("=" * 60)
    
    # Sample prompt that should generate the expected format
    test_prompt = "Solve this problem: What is 2 + 2?"
    expected_format_example = "<think>\nI need to add 2 + 2.\n2 + 2 = 4\n</think>\n<answer>4</answer>"
    
    print(f"📝 Test prompt: {test_prompt}")
    print(f"✅ Expected format example: {expected_format_example}")
    print()
    
    # Test 1: Standard model (unsloth disabled)
    print("1️⃣ Testing Standard Model (unsloth disabled)")
    print("-" * 50)
    
    config_manager = ConfigManager('config.yaml', 'test')
    config_standard = config_manager.load_config([
        '--model.unsloth.enabled', 'false',
        '--model.name', 'Qwen/Qwen2.5-0.5B-Instruct',  # Use small model for testing
        '--model.lora.enabled', 'false'
    ])
    
    try:
        model_std, tokenizer_std = setup_model_and_tokenizer(config_standard)
        print(f"   ✅ Standard model loaded: {model_std.__class__.__name__}")
        print(f"   📊 Chat template set: {tokenizer_std.chat_template is not None}")
        
        # Generate sample text
        messages = [{"role": "user", "content": test_prompt}]
        prompt = tokenizer_std.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer_std(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_std.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer_std.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][len(inputs['input_ids'][0]):]
        generated_text = tokenizer_std.decode(new_tokens, skip_special_tokens=True)
        
        print(f"   📄 Generated text: {repr(generated_text)}")
        
        # Test format reward
        completion = [[{"content": generated_text}]]
        reward = format_reward([completion])
        print(f"   🎯 Format reward: {reward[0]}")
        
        # Clean up
        del model_std
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Error with standard model: {e}")
    
    print()
    
    # Test 2: Unsloth model (unsloth enabled)
    print("2️⃣ Testing Unsloth Model (unsloth enabled)")
    print("-" * 50)
    
    config_unsloth = config_manager.load_config([
        '--model.unsloth.enabled', 'true',
        '--model.name', 'Qwen/Qwen2.5-0.5B-Instruct',  # Use small model for testing
        '--model.lora.enabled', 'false',
        '--model.unsloth.load_in_4bit', 'false'  # Disable 4bit for testing
    ])
    
    try:
        model_unsloth, tokenizer_unsloth = setup_model_and_tokenizer(config_unsloth)
        print(f"   ✅ Unsloth model loaded: {model_unsloth.__class__.__name__}")
        print(f"   📊 Chat template set: {tokenizer_unsloth.chat_template is not None}")
        
        # Generate sample text
        messages = [{"role": "user", "content": test_prompt}]
        prompt = tokenizer_unsloth.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer_unsloth(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model_unsloth.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.1,
                pad_token_id=tokenizer_unsloth.eos_token_id
            )
        
        # Decode only the new tokens
        new_tokens = outputs[0][len(inputs['input_ids'][0]):]
        generated_text = tokenizer_unsloth.decode(new_tokens, skip_special_tokens=True)
        
        print(f"   📄 Generated text: {repr(generated_text)}")
        
        # Test format reward
        completion = [[{"content": generated_text}]]
        reward = format_reward([completion])
        print(f"   🎯 Format reward: {reward[0]}")
        
        # Clean up
        del model_unsloth
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"   ❌ Error with unsloth model: {e}")
    
    print()
    
    # Test 3: Format reward function validation
    print("3️⃣ Testing Format Reward Function Validation")
    print("-" * 50)
    
    test_cases = [
        # Valid format
        ("<think>I need to calculate 2+2</think>\n<answer>4</answer>", 1.0),
        # Invalid format - missing think tags
        ("<answer>4</answer>", 0.0),
        # Invalid format - missing answer tags  
        ("<think>I need to calculate 2+2</think>", 0.0),
        # Invalid format - wrong order
        ("<answer>4</answer>\n<think>I need to calculate 2+2</think>", 0.0),
        # Invalid format - extra text before
        ("Some text before\n<think>reasoning</think>\n<answer>4</answer>", 0.0),
        # Invalid format - extra text after
        ("<think>reasoning</think>\n<answer>4</answer>\nExtra text", 0.0),
        # Valid format with complex content
        ("<think>First I calculate 2+2=4, then verify the result</think>\n<answer>The answer is 4</answer>", 1.0),
    ]
    
    for i, (text, expected_reward) in enumerate(test_cases, 1):
        completion = [[{"content": text}]]
        actual_reward = format_reward([completion])[0]
        status = "✅" if actual_reward == expected_reward else "❌"
        print(f"   {status} Test case {i}: Expected {expected_reward}, got {actual_reward}")
        if actual_reward != expected_reward:
            print(f"      Text: {repr(text)}")
    
    print()
    print("🎯 Summary and Recommendations:")
    print("1. Compare generated text format between standard and unsloth models")
    print("2. Check if unsloth models are following the expected <think>...</think><answer>...</answer> format")
    print("3. If unsloth generates different format, adjust training prompts or generation parameters")
    print("4. Consider adding format-specific training examples to improve unsloth model behavior")

def test_tokenizer_differences():
    """Test tokenizer configuration differences between standard and unsloth models."""
    print("\n🔍 Testing Tokenizer Configuration Differences")
    print("=" * 60)
    
    config_manager = ConfigManager('config.yaml', 'test')
    
    # Standard tokenizer
    config_std = config_manager.load_config(['--model.unsloth.enabled', 'false'])
    _, tokenizer_std = setup_model_and_tokenizer(config_std)
    
    # Unsloth tokenizer  
    config_unsloth = config_manager.load_config(['--model.unsloth.enabled', 'true'])
    _, tokenizer_unsloth = setup_model_and_tokenizer(config_unsloth)
    
    print("Comparing tokenizer configurations:")
    print(f"Standard pad_token: {tokenizer_std.pad_token}")
    print(f"Unsloth pad_token: {tokenizer_unsloth.pad_token}")
    print(f"Standard chat_template set: {tokenizer_std.chat_template is not None}")
    print(f"Unsloth chat_template set: {tokenizer_unsloth.chat_template is not None}")
    
    if tokenizer_std.chat_template and tokenizer_unsloth.chat_template:
        print(f"Templates identical: {tokenizer_std.chat_template == tokenizer_unsloth.chat_template}")

if __name__ == "__main__":
    test_format_reward_with_unsloth()
    test_tokenizer_differences()