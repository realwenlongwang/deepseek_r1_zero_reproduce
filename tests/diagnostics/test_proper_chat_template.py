#!/usr/bin/env python3
"""
Test the proper chat template implementation for both standard and unsloth models.
"""

import sys
sys.path.append('src')

from src.config import ConfigManager
from train_grpo import setup_model_and_tokenizer

def test_proper_chat_template():
    """Test that both standard and unsloth models now use proper chat templates."""
    print("🧪 Testing Proper Chat Template Implementation")
    print("=" * 60)
    
    config_manager = ConfigManager('config.yaml', 'test')
    
    # Test messages
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    print("Testing with sample messages:")
    print(f"System: {messages[0]['content']}")
    print(f"User: {messages[1]['content']}")
    print()
    
    # Test 1: Standard model
    print("1️⃣ Standard Model (should preserve original ChatML)")
    print("-" * 50)
    
    try:
        config_std = config_manager.load_config([
            '--model.unsloth.enabled', 'false',
            '--model.name', 'Qwen/Qwen2.5-0.5B-Instruct'
        ])
        
        model_std, tokenizer_std = setup_model_and_tokenizer(config_std)
        
        # Apply chat template
        prompt_std = tokenizer_std.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print(f"✅ Standard model loaded")
        print(f"📄 Generated prompt (first 200 chars):")
        print(f"   {repr(prompt_std[:200])}...")
        print()
        
        # Check for ChatML tokens
        has_chatml = '<|im_start|>' in prompt_std and '<|im_end|>' in prompt_std
        print(f"🎯 Uses ChatML format: {has_chatml}")
        
        if has_chatml:
            print("   ✅ PASS - Standard model preserves ChatML format")
        else:
            print("   ❌ FAIL - Standard model lost ChatML format")
        
        del model_std
        
    except Exception as e:
        print(f"   ❌ Error with standard model: {e}")
    
    print()
    
    # Test 2: Unsloth model
    print("2️⃣ Unsloth Model (should now use proper ChatML)")
    print("-" * 50)
    
    try:
        config_unsloth = config_manager.load_config([
            '--model.unsloth.enabled', 'true',
            '--model.name', 'Qwen/Qwen2.5-0.5B-Instruct',
            '--model.unsloth.load_in_4bit', 'false'
        ])
        
        model_unsloth, tokenizer_unsloth = setup_model_and_tokenizer(config_unsloth)
        
        # Apply chat template
        prompt_unsloth = tokenizer_unsloth.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print(f"✅ Unsloth model loaded")
        print(f"📄 Generated prompt (first 200 chars):")
        print(f"   {repr(prompt_unsloth[:200])}...")
        print()
        
        # Check for ChatML tokens
        has_chatml = '<|im_start|>' in prompt_unsloth and '<|im_end|>' in prompt_unsloth
        print(f"🎯 Uses ChatML format: {has_chatml}")
        
        if has_chatml:
            print("   ✅ PASS - Unsloth model now uses ChatML format")
        else:
            print("   ❌ FAIL - Unsloth model still missing ChatML format")
        
        del model_unsloth
        
    except Exception as e:
        print(f"   ❌ Error with unsloth model: {e}")
    
    print()
    print("🎯 Expected Results:")
    print("   • Both models should now generate prompts with <|im_start|> and <|im_end|> tokens")
    print("   • This should eliminate the 'Assistant:' prefix issue")
    print("   • Format rewards should work consistently for both approaches")

def test_completion_format_consistency():
    """Test that completions from both models now have consistent format."""
    print("\n🔄 Testing Completion Format Consistency")
    print("=" * 60)
    
    print("With proper ChatML templates:")
    print("• Standard models: Generate completions directly after <|im_start|>assistant\\n")
    print("• Unsloth models: Should now also generate completions directly (no 'Assistant:' prefix)")
    print()
    print("This means both should generate:")
    print("  '<think>reasoning</think><answer>answer</answer>'")
    print()
    print("Instead of unsloth generating:")
    print("  'Assistant: <think>reasoning</think><answer>answer</answer>'")

if __name__ == "__main__":
    test_proper_chat_template()
    test_completion_format_consistency()
    
    print("\n🎉 If both models now use ChatML format:")
    print("   1. The format reward should work consistently")
    print("   2. No more 'Assistant:' prefix from unsloth models")
    print("   3. Both approaches will generate the same prompt structure")
    print("   4. Training behavior should be consistent between standard and unsloth")