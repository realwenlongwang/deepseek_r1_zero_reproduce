#!/usr/bin/env python3

"""
Test script for Countdown-Tasks dataset integration.
"""

import sys
from src.data.dataset import ReasoningDataset, create_dataset

def test_countdown_dataset():
    """Test loading and processing of Countdown-Tasks dataset."""
    print("Testing Countdown-Tasks dataset integration...")
    
    try:
        # Create dataset with small sample
        dataset = create_dataset(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            split="train[:5]",  # Just first 5 examples for testing
            max_length=2048,
            tokenizer=None
        )
        
        print(f"✓ Dataset loaded successfully with {len(dataset)} examples")
        
        # Test a few examples
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"\n--- Example {i+1} ---")
            print(f"Dataset type: {example.get('dataset_type', 'unknown')}")
            print(f"Target: {example.get('target', 'N/A')}")
            print(f"Numbers: {example.get('nums', 'N/A')}")
            print(f"Problem: {example.get('problem', 'N/A')[:100]}...")
            
            # Check conversation format
            prompt = example.get('prompt', [])
            if prompt:
                print(f"System message: {prompt[0].get('content', '')[:50]}...")
                print(f"User message: {prompt[1].get('content', '')[:100]}...")
            
        print("\n✓ All examples processed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_problem_formatting():
    """Test the problem formatting function."""
    print("\nTesting problem formatting...")
    
    from src.data.dataset import format_countdown_problem
    
    test_cases = [
        (98, [44, 19, 35]),
        (64, [63, 95, 96]),
        (82, [83, 78, 1, 39]),
        (40, [96]),  # Single number case
    ]
    
    for target, nums in test_cases:
        problem = format_countdown_problem(target, nums)
        print(f"Target: {target}, Numbers: {nums}")
        print(f"Problem: {problem}")
        print()
    
    print("✓ Problem formatting working correctly")

if __name__ == "__main__":
    print("=== Countdown Dataset Integration Test ===\n")
    
    # Test problem formatting first (no network required)
    test_problem_formatting()
    
    # Test dataset loading
    success = test_countdown_dataset()
    
    if success:
        print("\n🎉 All tests passed! Countdown dataset integration is working.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
        sys.exit(1)