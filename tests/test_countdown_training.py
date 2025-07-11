#!/usr/bin/env python3

"""
Test script for Countdown-Tasks dataset integration with GRPO training pipeline.
"""

import sys
import torch
from transformers import AutoTokenizer
from src.data.dataset import create_dataset

def test_countdown_with_tokenizer():
    """Test countdown dataset with tokenizer integration."""
    print("Testing Countdown dataset with tokenizer...")
    
    try:
        # Load a small model tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataset with tokenizer
        dataset = create_dataset(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            split="train[:3]",  # Just 3 examples for testing
            max_length=512,
            tokenizer=tokenizer
        )
        
        print(f"✓ Dataset created with tokenizer, {len(dataset)} examples")
        
        # Test tokenized examples
        for i in range(len(dataset)):
            example = dataset[i]
            print(f"\n--- Tokenized Example {i+1} ---")
            print(f"Dataset type: {example.get('dataset_type')}")
            print(f"Target: {example.get('target')}")
            print(f"Numbers: {example.get('nums')}")
            
            # Check tokenized fields
            if 'input_ids' in example:
                print(f"Input IDs shape: {example['input_ids'].shape}")
                print(f"Attention mask shape: {example['attention_mask'].shape}")
                print(f"Text preview: {example.get('text', '')[:100]}...")
            
        print("\n✓ Tokenization working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error testing with tokenizer: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_generation():
    """Test batch generation functionality."""
    print("\nTesting batch generation...")
    
    try:
        dataset = create_dataset(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            split="train[:10]",
            max_length=512,
            tokenizer=None
        )
        
        # Test get_batch method
        batch = dataset.get_batch(batch_size=3)
        print(f"✓ Generated batch with {len(batch)} examples")
        
        for i, example in enumerate(batch):
            print(f"Batch item {i+1}: target={example.get('target')}, nums={example.get('nums')}")
        
        print("✓ Batch generation working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Error testing batch generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Countdown Dataset Training Pipeline Test ===\n")
    
    # Test with tokenizer integration
    success1 = test_countdown_with_tokenizer()
    
    # Test batch generation
    success2 = test_batch_generation()
    
    if success1 and success2:
        print("\n🎉 All training pipeline tests passed!")
        print("The Countdown dataset is ready for GRPO training.")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1)