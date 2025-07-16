#!/usr/bin/env python3
"""
Simple test suite for train/test split functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from src.data.dataset import ReasoningDataset, create_dataset, create_train_test_datasets


class TestTrainTestSplitSimple(unittest.TestCase):
    """Simple test train/test split functionality."""
    
    def test_split_detection_logic(self):
        """Test the split detection logic directly."""
        print(f"\n=== Testing split detection logic ===")
        
        # Test with NuminaMath-TIR (has existing test split)
        dataset_with_test = ReasoningDataset(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="train",
            create_splits=True,
            test_size=0.1,
            split_seed=42
        )
        
        # Should detect existing test split
        has_test = dataset_with_test._check_test_split_exists()
        print(f"NuminaMath-TIR has test split: {has_test}")
        self.assertTrue(has_test)
        
        # Test with Countdown-Tasks (no existing test split)
        dataset_without_test = ReasoningDataset(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            split="train",
            create_splits=True,
            test_size=0.1,
            split_seed=42
        )
        
        # Should not detect existing test split
        has_test = dataset_without_test._check_test_split_exists()
        print(f"Countdown-Tasks has test split: {has_test}")
        self.assertFalse(has_test)
    
    def test_basic_dataset_creation(self):
        """Test basic dataset creation with splits."""
        print(f"\n=== Testing basic dataset creation ===")
        
        # Create a small test with existing splits
        train_dataset = create_dataset(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="train",
            create_splits=True,
            test_size=0.1,
            split_seed=42
        )
        
        test_dataset = create_dataset(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="test",
            create_splits=True,
            test_size=0.1,
            split_seed=42
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Basic checks
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        # For NuminaMath-TIR, we expect specific sizes
        self.assertEqual(len(train_dataset), 72441)  # Known train size
        self.assertEqual(len(test_dataset), 99)     # Known test size
    
    def test_reproducibility(self):
        """Test that splits are reproducible with the same seed."""
        print(f"\n=== Testing reproducibility ===")
        
        # Create datasets with same seed twice
        train_ds1 = create_dataset(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="train",
            create_splits=True,
            test_size=0.1,
            split_seed=42
        )
        
        train_ds2 = create_dataset(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="train",
            create_splits=True,
            test_size=0.1,
            split_seed=42
        )
        
        # Should have same sizes
        self.assertEqual(len(train_ds1), len(train_ds2))
        print(f"Reproducibility test passed - both have {len(train_ds1)} examples")
    
    def test_disable_split_creation(self):
        """Test that split creation can be disabled."""
        print(f"\n=== Testing disabled split creation ===")
        
        # Create dataset with split creation disabled
        train_dataset = create_dataset(
            dataset_name="AI-MO/NuminaMath-TIR",
            split="train",
            create_splits=False  # Disable split creation
        )
        
        print(f"Dataset with disabled split creation - Train: {len(train_dataset)}")
        
        self.assertGreater(len(train_dataset), 0)
        self.assertEqual(len(train_dataset), 72441)  # Should be original train size


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)