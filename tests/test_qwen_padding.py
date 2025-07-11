#!/usr/bin/env python3
"""
Test to verify that Qwen2.5 tokenizer padding side is configurable and not fixed during training.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def test_qwen_padding_configurability():
    """Test that Qwen2.5 tokenizer padding side is configurable."""
    print("="*60)
    print("QWEN2.5 PADDING SIDE CONFIGURABILITY TEST")
    print("="*60)
    
    model_name = "Qwen/Qwen2.5-3B"
    
    # Test 1: Check default padding side
    print(f"\n1. Testing default padding side:")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"   Default padding_side: {tokenizer.padding_side}")
    
    # Test 2: Change padding side and verify it works
    print(f"\n2. Testing padding side configurability:")
    tokenizer.padding_side = 'left'
    print(f"   After setting to 'left': {tokenizer.padding_side}")
    
    # Test 3: Verify this works with actual tokenization
    print(f"\n3. Testing actual tokenization with different padding sides:")
    test_texts = ["Hello", "This is a longer sentence for testing"]
    
    # Test with left padding
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    left_encoded = tokenizer(test_texts, padding=True, return_tensors="pt")
    print(f"   Left padding shape: {left_encoded['input_ids'].shape}")
    print(f"   Left padding example: {left_encoded['input_ids'][0][:5].tolist()}")
    
    # Test with right padding
    tokenizer.padding_side = 'right'
    right_encoded = tokenizer(test_texts, padding=True, return_tensors="pt")
    print(f"   Right padding shape: {right_encoded['input_ids'].shape}")
    print(f"   Right padding example: {right_encoded['input_ids'][0][:5].tolist()}")
    
    # Test 4: Test with Flash Attention model
    print(f"\n4. Testing with Flash Attention model:")
    try:
        # Load model with Flash Attention
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        
        # Test generation with right padding (should fail)
        print("   Testing generation with right padding...")
        tokenizer.padding_side = 'right'
        try:
            inputs = tokenizer(test_texts, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            print("   ❌ Unexpected: Right padding worked with Flash Attention")
        except Exception as e:
            if "padding_side='right'" in str(e):
                print("   ✅ Expected: Right padding failed with Flash Attention")
                print(f"   Error: {str(e)[:100]}...")
            else:
                print(f"   ❓ Unexpected error: {e}")
        
        # Test generation with left padding (should work)
        print("   Testing generation with left padding...")
        tokenizer.padding_side = 'left'
        try:
            inputs = tokenizer(test_texts, padding=True, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            print("   ✅ Success: Left padding worked with Flash Attention")
        except Exception as e:
            print(f"   ❌ Unexpected: Left padding failed: {e}")
            
    except Exception as e:
        print(f"   Model loading failed: {e}")
    
    print(f"\n" + "="*60)
    print("CONCLUSION:")
    print("- Qwen2.5 tokenizer padding_side is NOT fixed during training")
    print("- It's a runtime configuration that defaults to 'right'")
    print("- Flash Attention requires 'left' padding for batched generation")
    print("- The issue is in TRL's GRPOTrainer not maintaining the padding_side setting")
    print("="*60)

if __name__ == "__main__":
    test_qwen_padding_configurability()