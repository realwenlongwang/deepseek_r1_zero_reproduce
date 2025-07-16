#!/usr/bin/env python3
"""
Test train/test split creation for datasets without existing splits.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from src.data.dataset import create_train_test_datasets


class TestCountdownSplit(unittest.TestCase):
    """Test train/test split creation for Countdown-Tasks dataset."""
    
    def test_countdown_split_creation(self):
        """Test creating splits for Countdown-Tasks dataset."""
        print(f"\n=== Testing Countdown-Tasks split creation ===")
        
        # Create train and test datasets
        train_dataset, test_dataset = create_train_test_datasets(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            test_size=0.1,
            split_seed=42
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Basic checks
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        # Check split ratio
        total_size = len(train_dataset) + len(test_dataset)
        actual_test_ratio = len(test_dataset) / total_size
        expected_test_ratio = 0.1
        
        print(f"Total size: {total_size}")
        print(f"Expected test ratio: {expected_test_ratio}")
        print(f"Actual test ratio: {actual_test_ratio:.3f}")
        
        # Allow for some tolerance due to rounding
        self.assertAlmostEqual(actual_test_ratio, expected_test_ratio, places=2)
        
        # Check that we're getting the expected total (should be around 490k)
        self.assertGreater(total_size, 400000)
        self.assertLess(total_size, 500000)
    
    def test_countdown_reproducibility(self):
        """Test that Countdown-Tasks splits are reproducible."""
        print(f"\n=== Testing Countdown-Tasks reproducibility ===")
        
        # Create datasets with same seed twice
        train_ds1, test_ds1 = create_train_test_datasets(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            test_size=0.1,
            split_seed=42
        )
        
        train_ds2, test_ds2 = create_train_test_datasets(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            test_size=0.1,
            split_seed=42
        )
        
        # Should have same sizes
        self.assertEqual(len(train_ds1), len(train_ds2))
        self.assertEqual(len(test_ds1), len(test_ds2))
        
        print(f"Reproducibility test passed - Train: {len(train_ds1)}, Test: {len(test_ds1)}")
        
        # Test with different seed
        train_ds3, test_ds3 = create_train_test_datasets(
            dataset_name="Jiayi-Pan/Countdown-Tasks-3to4",
            test_size=0.1,
            split_seed=123  # Different seed
        )
        
        # Should have same sizes (same total data)
        self.assertEqual(len(train_ds1), len(train_ds3))
        self.assertEqual(len(test_ds1), len(test_ds3))
        
        print(f"Different seed test passed - Train: {len(train_ds3)}, Test: {len(test_ds3)}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)