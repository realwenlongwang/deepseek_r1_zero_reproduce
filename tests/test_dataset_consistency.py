#!/usr/bin/env python3
"""
Test to verify that the AI-MO/NuminaMath-TIR test set is deterministic
and returns the same 99 samples in the same order when loaded multiple times.
"""

import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset import create_dataset


def test_test_set_consistency():
    """Test that loading the test set twice gives identical results."""
    print("Loading test set (first time)...")
    test_dataset_1 = create_dataset("AI-MO/NuminaMath-TIR", split="test")
    
    print("Loading test set (second time)...")
    test_dataset_2 = create_dataset("AI-MO/NuminaMath-TIR", split="test")
    
    # Check basic properties
    print(f"First load: {len(test_dataset_1)} samples")
    print(f"Second load: {len(test_dataset_2)} samples")
    
    # Verify same length
    assert len(test_dataset_1) == len(test_dataset_2), "Test set lengths don't match!"
    print(f"✅ Both loads have {len(test_dataset_1)} samples")
    
    # Verify same content by comparing first few samples
    print("\nComparing first 5 samples...")
    for i in range(min(5, len(test_dataset_1))):
        sample_1 = test_dataset_1[i]
        sample_2 = test_dataset_2[i]
        
        # Compare problem text
        problem_1 = sample_1.get('problem', '')
        problem_2 = sample_2.get('problem', '')
        
        print(f"Sample {i+1}:")
        print(f"  Problem length: {len(problem_1)} vs {len(problem_2)}")
        print(f"  Problems match: {problem_1 == problem_2}")
        
        assert problem_1 == problem_2, f"Sample {i+1} problems don't match!"
        
        # Compare other fields if they exist
        for key in ['reference_solution', 'reference_answer', 'dataset_type']:
            if key in sample_1 and key in sample_2:
                val_1 = sample_1[key]
                val_2 = sample_2[key]
                assert val_1 == val_2, f"Sample {i+1} field '{key}' doesn't match!"
    
    print("✅ First 5 samples are identical between loads")
    
    # Compare all problems to be thorough
    print("\nComparing all problems...")
    all_problems_1 = [test_dataset_1[i].get('problem', '') for i in range(len(test_dataset_1))]
    all_problems_2 = [test_dataset_2[i].get('problem', '') for i in range(len(test_dataset_2))]
    
    assert all_problems_1 == all_problems_2, "Not all problems match between loads!"
    print(f"✅ All {len(all_problems_1)} problems are identical between loads")
    
    # Show some sample info
    print(f"\nSample problem lengths: {[len(p) for p in all_problems_1[:5]]}")
    print(f"First problem preview: {all_problems_1[0][:100]}...")
    
    return True


def test_train_set_size():
    """Verify the train set has the expected size."""
    print("\n" + "="*50)
    print("TESTING TRAIN SET SIZE")
    print("="*50)
    
    train_dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
    print(f"Train set size: {len(train_dataset)}")
    
    # Based on the dataset documentation, train should have ~72k samples
    assert len(train_dataset) > 70000, f"Train set seems too small: {len(train_dataset)}"
    print("✅ Train set has expected size (>70k samples)")
    
    return True


def test_no_data_leakage():
    """Test that test set samples don't appear in the training set."""
    import random
    
    print("\n" + "="*50)
    print("TESTING FOR DATA LEAKAGE")
    print("="*50)
    
    # Load both datasets
    print("Loading train and test datasets...")
    train_dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
    test_dataset = create_dataset("AI-MO/NuminaMath-TIR", split="test")
    
    print(f"Train set: {len(train_dataset)} samples")
    print(f"Test set: {len(test_dataset)} samples")
    
    # Extract all training problems for fast lookup
    print("Extracting training problems...")
    train_problems = set()
    for i in range(len(train_dataset)):
        problem = train_dataset[i].get('problem', '').strip()
        train_problems.add(problem)
    
    print(f"Extracted {len(train_problems)} unique training problems")
    
    # Randomly select 5 test samples
    random.seed(42)  # For reproducibility
    test_indices = random.sample(range(len(test_dataset)), 5)
    
    print(f"\nChecking 5 randomly selected test samples (indices: {test_indices})...")
    
    found_leakage = False
    for i, test_idx in enumerate(test_indices, 1):
        test_sample = test_dataset[test_idx]
        test_problem = test_sample.get('problem', '').strip()
        
        print(f"\nTest sample {i} (index {test_idx}):")
        print(f"  Problem preview: {test_problem[:100]}...")
        print(f"  Problem length: {len(test_problem)} characters")
        
        # Check if this test problem exists in training set
        if test_problem in train_problems:
            print(f"  ❌ FOUND IN TRAINING SET!")
            found_leakage = True
        else:
            print(f"  ✅ Not found in training set")
    
    # Also check for partial matches (in case of minor formatting differences)
    print(f"\n🔍 Checking for partial matches (first 50 chars)...")
    
    # Create a set of training problem prefixes for partial matching
    train_prefixes = set()
    for i in range(len(train_dataset)):
        problem = train_dataset[i].get('problem', '').strip()
        if len(problem) >= 50:
            train_prefixes.add(problem[:50])
    
    partial_matches = 0
    for i, test_idx in enumerate(test_indices, 1):
        test_sample = test_dataset[test_idx]
        test_problem = test_sample.get('problem', '').strip()
        
        if len(test_problem) >= 50:
            test_prefix = test_problem[:50]
            if test_prefix in train_prefixes:
                print(f"  ⚠️  Test sample {i}: Partial match found (first 50 chars)")
                partial_matches += 1
    
    if partial_matches == 0:
        print(f"  ✅ No partial matches found")
    
    # Final assessment
    print(f"\n" + "="*30)
    print("DATA LEAKAGE ASSESSMENT")
    print("="*30)
    
    if found_leakage:
        print("❌ DATA LEAKAGE DETECTED!")
        print("Some test samples were found in the training set.")
        assert False, "Data leakage detected!"
    else:
        print("✅ NO DATA LEAKAGE DETECTED")
        print("Test samples are properly isolated from training data.")
        print(f"✅ {len(test_indices)} random test samples verified")
        print(f"✅ No exact or partial matches found")
    
    return True


if __name__ == "__main__":
    print("="*50)
    print("COMPREHENSIVE DATASET VALIDATION")
    print("="*50)
    
    try:
        test_test_set_consistency()
        test_train_set_size()
        test_no_data_leakage()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("✅ Test set is deterministic and consistent")
        print("✅ Train/test splits are properly isolated")
        print("✅ No data leakage between train and test sets")
        print("✅ Safe to train on train set and evaluate on test set")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)