#!/usr/bin/env python3
"""
Test suite for train/test split functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from src.data.dataset import ReasoningDataset, create_dataset, create_train_test_datasets


class TestTrainTestSplit(unittest.TestCase):
    """Test train/test split functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dataset_with_test = "AI-MO/NuminaMath-TIR"  # Has existing test split
        self.dataset_without_test = "Jiayi-Pan/Countdown-Tasks-3to4"  # No test split
        self.split_seed = 42
        self.test_size = 0.1
    
    def test_dataset_with_existing_test_split(self):
        """Test dataset that already has a test split."""
        print(f"\n=== Testing dataset with existing test split: {self.dataset_with_test} ===")
        
        # Create train dataset
        train_dataset = create_dataset(
            dataset_name=self.dataset_with_test,
            split="train",
            create_splits=True,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        # Create test dataset
        test_dataset = create_dataset(
            dataset_name=self.dataset_with_test,
            split="test",
            create_splits=True,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Verify that datasets are loaded
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        # For datasets with existing test splits, the test dataset should be small
        # (NuminaMath-TIR has 99 test examples)
        self.assertLess(len(test_dataset), 200)
    
    def test_dataset_without_test_split(self):
        """Test dataset that doesn't have a test split."""
        print(f"\n=== Testing dataset without test split: {self.dataset_without_test} ===")
        
        # Create train dataset
        train_dataset = create_dataset(
            dataset_name=self.dataset_without_test,
            split="train",
            create_splits=True,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        # Create test dataset
        test_dataset = create_dataset(
            dataset_name=self.dataset_without_test,
            split="test",
            create_splits=True,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Verify that datasets are loaded
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        # Verify train/test split ratio
        total_size = len(train_dataset) + len(test_dataset)
        actual_test_ratio = len(test_dataset) / total_size
        expected_test_ratio = self.test_size
        
        print(f"Expected test ratio: {expected_test_ratio}")
        print(f"Actual test ratio: {actual_test_ratio:.3f}")
        
        # Allow for some tolerance due to rounding
        self.assertAlmostEqual(actual_test_ratio, expected_test_ratio, places=2)
    
    def test_create_train_test_datasets_function(self):
        """Test the create_train_test_datasets helper function."""
        print(f"\n=== Testing create_train_test_datasets function ===")
        
        # Test with dataset that has existing splits
        train_ds, test_ds = create_train_test_datasets(
            dataset_name=self.dataset_with_test,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        print(f"Dataset with existing splits - Train: {len(train_ds)}, Test: {len(test_ds)}")
        
        self.assertGreater(len(train_ds), 0)
        self.assertGreater(len(test_ds), 0)
        
        # Test with dataset that doesn't have existing splits
        train_ds2, test_ds2 = create_train_test_datasets(
            dataset_name=self.dataset_without_test,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        print(f"Dataset without existing splits - Train: {len(train_ds2)}, Test: {len(test_ds2)}")
        
        self.assertGreater(len(train_ds2), 0)
        self.assertGreater(len(test_ds2), 0)
    
    def test_split_reproducibility(self):
        """Test that splits are reproducible with the same seed."""
        print(f"\n=== Testing split reproducibility ===")
        
        # Create datasets with same seed
        train_ds1, test_ds1 = create_train_test_datasets(
            dataset_name=self.dataset_without_test,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        train_ds2, test_ds2 = create_train_test_datasets(
            dataset_name=self.dataset_without_test,
            test_size=self.test_size,
            split_seed=self.split_seed
        )
        
        # Should have same sizes
        self.assertEqual(len(train_ds1), len(train_ds2))
        self.assertEqual(len(test_ds1), len(test_ds2))
        
        print(f"Reproducibility test passed - Train: {len(train_ds1)}, Test: {len(test_ds1)}")
        
        # Test with different seed
        train_ds3, test_ds3 = create_train_test_datasets(
            dataset_name=self.dataset_without_test,
            test_size=self.test_size,
            split_seed=123  # Different seed
        )
        
        # Should have same sizes but potentially different content
        self.assertEqual(len(train_ds1), len(train_ds3))
        self.assertEqual(len(test_ds1), len(test_ds3))
        
        print(f"Different seed test passed - Train: {len(train_ds3)}, Test: {len(test_ds3)}")
    
    def test_disable_split_creation(self):
        """Test that split creation can be disabled."""
        print(f"\n=== Testing disabled split creation ===")
        
        # Create dataset with split creation disabled
        train_dataset = create_dataset(
            dataset_name=self.dataset_with_test,
            split="train",
            create_splits=False  # Disable split creation
        )
        
        print(f"Dataset with disabled split creation - Train: {len(train_dataset)}")
        
        self.assertGreater(len(train_dataset), 0)
    
    def test_different_test_sizes(self):
        """Test different test split sizes."""
        print(f"\n=== Testing different test sizes ===")
        
        test_sizes = [0.05, 0.1, 0.2]
        
        for test_size in test_sizes:
            train_ds, test_ds = create_train_test_datasets(
                dataset_name=self.dataset_without_test,
                test_size=test_size,
                split_seed=self.split_seed
            )
            
            total_size = len(train_ds) + len(test_ds)
            actual_test_ratio = len(test_ds) / total_size
            
            print(f"Test size {test_size}: Train={len(train_ds)}, Test={len(test_ds)}, Ratio={actual_test_ratio:.3f}")
            
            # Allow for some tolerance due to rounding
            self.assertAlmostEqual(actual_test_ratio, test_size, places=2)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)