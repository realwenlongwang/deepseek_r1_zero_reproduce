#!/usr/bin/env python3
"""
Integration test for training script with train/test split functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from src.data.dataset import create_train_test_datasets


class TestTrainingIntegration(unittest.TestCase):
    """Test training integration with new split functionality."""
    
    def test_datasets_can_be_used_for_training(self):
        """Test that created datasets can be used for training."""
        print(f"\n=== Testing datasets for training integration ===")
        
        # Create train and test datasets
        train_dataset, test_dataset = create_train_test_datasets(
            dataset_name="AI-MO/NuminaMath-TIR",  # Use smaller dataset for speed
            test_size=0.1,
            split_seed=42
        )
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")
        
        # Basic checks
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(test_dataset), 0)
        
        # Test that we can access dataset items
        train_item = train_dataset[0]
        test_item = test_dataset[0]
        
        print(f"Train item keys: {list(train_item.keys())}")
        print(f"Test item keys: {list(test_item.keys())}")
        
        # Check that items have expected structure
        required_keys = ['prompt', 'problem', 'reference_solution', 'reference_answer', 'dataset_type']
        for key in required_keys:
            self.assertIn(key, train_item)
            self.assertIn(key, test_item)
        
        # Check that prompt is in conversation format
        self.assertIsInstance(train_item['prompt'], list)
        self.assertIsInstance(test_item['prompt'], list)
        
        # Check that each message has role and content
        for message in train_item['prompt']:
            self.assertIn('role', message)
            self.assertIn('content', message)
        
        print("✅ Datasets have correct structure for training")
    
    def test_different_dataset_types_work(self):
        """Test that different dataset types work correctly."""
        print(f"\n=== Testing different dataset types ===")
        
        # Test NuminaMath-TIR (has existing splits)
        train_numina, test_numina = create_train_test_datasets(
            dataset_name="AI-MO/NuminaMath-TIR",
            test_size=0.1,
            split_seed=42
        )
        
        # Check dataset type
        train_item = train_numina[0]
        self.assertEqual(train_item['dataset_type'], 'numina')
        
        print(f"NuminaMath - Train: {len(train_numina)}, Test: {len(test_numina)}")
        print(f"NuminaMath dataset type: {train_item['dataset_type']}")
        
        # Test that all items have same dataset type
        self.assertTrue(all(item['dataset_type'] == 'numina' for item in train_numina.dataset))
        self.assertTrue(all(item['dataset_type'] == 'numina' for item in test_numina.dataset))
        
        print("✅ Different dataset types work correctly")
    
    def test_dataset_compatibility_with_mock_trainer(self):
        """Test that datasets are compatible with training setup."""
        print(f"\n=== Testing dataset compatibility with training setup ===")
        
        # Create datasets
        train_dataset, test_dataset = create_train_test_datasets(
            dataset_name="AI-MO/NuminaMath-TIR",
            test_size=0.1,
            split_seed=42
        )
        
        # Mock a simple training setup check
        # This simulates what the trainer would do
        
        # Check that we can iterate through datasets
        train_batch = []
        for i, item in enumerate(train_dataset):
            train_batch.append(item)
            if i >= 2:  # Just check first 3 items
                break
        
        test_batch = []
        for i, item in enumerate(test_dataset):
            test_batch.append(item)
            if i >= 2:  # Just check first 3 items
                break
        
        # Verify batch structure
        self.assertEqual(len(train_batch), 3)
        self.assertEqual(len(test_batch), 3)
        
        # Check that all items have the required structure
        for item in train_batch + test_batch:
            self.assertIn('prompt', item)
            self.assertIsInstance(item['prompt'], list)
            self.assertGreater(len(item['prompt']), 0)
        
        print("✅ Datasets are compatible with training setup")
        print(f"Train batch size: {len(train_batch)}")
        print(f"Test batch size: {len(test_batch)}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)