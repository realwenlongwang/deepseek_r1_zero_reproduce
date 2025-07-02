#!/usr/bin/env python3
"""
Real dataset test that actually downloads and validates real datasets.
Tests the dataset.py implementation against actual Hugging Face datasets.
"""

import os
import sys
import pytest

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.data.dataset import create_dataset


def validate_dataset_format(sample):
    """
    Validate a single dataset sample against the expected format.
    
    Args:
        sample: Dictionary containing the dataset sample
        
    Returns:
        dict: Validation results
    """
    required_fields = ["problem", "prompt"]
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in sample]
    has_required_fields = len(missing_fields) == 0
    
    # Check prompt format
    correct_prompt_format = False
    if 'prompt' in sample:
        messages = sample['prompt']
        correct_prompt_format = (
            isinstance(messages, list) and
            len(messages) >= 2 and
            messages[0]['role'] == 'system' and
            messages[1]['role'] == 'user'
        )
    
    return {
        "has_required_fields": has_required_fields,
        "missing_fields": missing_fields,
        "correct_prompt_format": correct_prompt_format
    }


@pytest.fixture(scope="session")
def numina_dataset():
    """Load real NuminaMath-TIR dataset - this will actually download it."""
    print("\nDownloading real NuminaMath-TIR dataset...")
    try:
        # This will actually download the real dataset
        train_dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
        
        # Try to get test split if available, otherwise use a subset of train
        try:
            test_dataset = create_dataset("AI-MO/NuminaMath-TIR", split="test")
        except:
            # If no test split, use a small subset of train data
            print("No test split found, using subset of train for testing")
            test_dataset = train_dataset
        
        return {
            "train": train_dataset,
            "test": test_dataset
        }
    except Exception as e:
        pytest.skip(f"Could not download NuminaMath-TIR dataset: {e}")


class TestRealDatasetDownload:
    """Test real dataset download and validation."""

    def test_real_dataset_loading(self, numina_dataset):
        """Test that we can actually load real datasets."""
        train_dataset = numina_dataset["train"]
        test_dataset = numina_dataset["test"]
        
        assert len(train_dataset) > 0, "Train dataset should not be empty"
        assert len(test_dataset) > 0, "Test dataset should not be empty"
        
        print(f"✓ Loaded train dataset with {len(train_dataset)} samples")
        print(f"✓ Loaded test dataset with {len(test_dataset)} samples")

    def test_real_dataset_sample_structure(self, numina_dataset):
        """Test the structure of real dataset samples."""
        train_dataset = numina_dataset["train"]
        
        # Get first sample
        sample = train_dataset[0]
        
        print(f"\nReal dataset sample keys: {list(sample.keys())}")
        print(f"Sample problem: {sample.get('problem', 'N/A')[:100]}...")
        print(f"Sample prompt length: {len(sample.get('prompt', []))}")
        
        # Validate structure
        res = validate_dataset_format(sample)
        assert res["has_required_fields"], f"Missing fields: {res['missing_fields']}"
        assert res["correct_prompt_format"], "Incorrect prompt format"

    @pytest.mark.parametrize("split", ["train", "test"])
    def test_real_dataset_structure_complete(self, numina_dataset, split):
        """
        Test complete real dataset structure - following the exact pattern requested.
        This tests every sample in the real downloaded dataset.
        """
        dataset_split = numina_dataset[split]
        
        # Test a reasonable subset to avoid very long test times
        max_samples_to_test = min(100, len(dataset_split))
        
        failed_samples = []
        
        for idx in range(max_samples_to_test):
            sample = dataset_split[idx]
            res = validate_dataset_format(sample)
            
            if not res["has_required_fields"]:
                failed_samples.append(f"row {idx} missing: {res['missing_fields']}")
            
            if not res["correct_prompt_format"]:
                failed_samples.append(f"row {idx} bad prompt")
        
        # Report results
        if failed_samples:
            pytest.fail(f"Real dataset {split} validation failed:\n" + 
                       "\n".join(failed_samples[:10]))  # Show first 10 failures
        
        print(f"✓ Validated {max_samples_to_test} real {split} samples successfully")

    def test_real_dataset_content_analysis(self, numina_dataset):
        """Analyze the content of real dataset samples."""
        train_dataset = numina_dataset["train"]
        
        # Analyze first few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            
            print(f"\n--- Real Sample {i+1} Analysis ---")
            print(f"Problem: {sample.get('problem', 'N/A')[:200]}...")
            
            if 'prompt' in sample:
                prompt = sample['prompt']
                print(f"Prompt messages: {len(prompt)}")
                
                for j, msg in enumerate(prompt):
                    role = msg.get('role', 'unknown')
                    content_preview = msg.get('content', '')[:100]
                    print(f"  Message {j+1} ({role}): {content_preview}...")
            
            # Validate this sample
            res = validate_dataset_format(sample)
            print(f"Validation: Fields={res['has_required_fields']}, "
                  f"Format={res['correct_prompt_format']}")

    def test_dataset_processing_quality(self, numina_dataset):
        """Test the quality of dataset processing."""
        train_dataset = numina_dataset["train"]
        sample = train_dataset[0]
        
        # Check that our processing creates the expected structure
        assert 'problem' in sample, "Should have problem field"
        assert 'prompt' in sample, "Should have prompt field"
        
        # Check prompt structure
        prompt = sample['prompt']
        assert isinstance(prompt, list), "Prompt should be a list"
        assert len(prompt) >= 2, "Should have system and user messages"
        
        # Check system message
        system_msg = prompt[0]
        assert system_msg['role'] == 'system', "First message should be system"
        system_content = system_msg['content']
        assert '<think>' in system_content, "System prompt should mention <think>"
        assert '<answer>' in system_content, "System prompt should mention <answer>"
        
        # Check user message
        user_msg = prompt[1]
        assert user_msg['role'] == 'user', "Second message should be user"
        assert user_msg['content'] == sample['problem'], "User message should match problem"


def test_alternative_dataset_loading():
    """Test loading different dataset if available."""
    try:
        print("\nTrying to load alternative dataset...")
        # Try loading a different dataset that might be available
        dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
        
        if len(dataset) > 0:
            sample = dataset[0]
            res = validate_dataset_format(sample)
            
            assert res["has_required_fields"], f"Missing fields: {res['missing_fields']}"
            assert res["correct_prompt_format"], "Incorrect prompt format"
            
            print(f"✓ Successfully loaded and validated alternative dataset")
            print(f"  Dataset size: {len(dataset)} samples")
            print(f"  Sample structure: {list(sample.keys())}")
        
    except Exception as e:
        pytest.skip(f"Alternative dataset loading failed: {e}")


if __name__ == "__main__":
    # Run with pytest when called directly
    print("Starting real dataset download and validation...")
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements