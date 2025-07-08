#!/usr/bin/env python3
"""
Test checkpoint naming functionality to ensure unique, descriptive directory names.
"""

import os
import sys
import re
import time
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from train_grpo import generate_unique_output_dir


class TestCheckpointNaming:
    """Test the checkpoint naming functionality."""
    
    def test_generate_unique_output_dir_0_5b_model(self):
        """Test output directory generation for 0.5B model."""
        model_name = "./models/Qwen2.5-0.5B"
        reward_funcs = ["accuracy", "format", "reasoning_steps"]
        
        output_dir = generate_unique_output_dir(model_name, reward_funcs)
        
        # Check format
        assert output_dir.startswith("saved_models/"), f"Should start with saved_models/, got: {output_dir}"
        assert "qwen2.5-0.5b" in output_dir, f"Should contain model size, got: {output_dir}"
        assert "accuracy-format-reasoning" in output_dir, f"Should contain reward funcs, got: {output_dir}"
        
        # Check timestamp format (YYYYMMDD_HHMMSS)
        timestamp_pattern = r'\d{8}_\d{6}$'
        assert re.search(timestamp_pattern, output_dir), f"Should end with timestamp, got: {output_dir}"
        
        print(f"✅ 0.5B model output: {output_dir}")
    
    def test_generate_unique_output_dir_7b_model(self):
        """Test output directory generation for 7B model."""
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        reward_funcs = ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"]
        
        output_dir = generate_unique_output_dir(model_name, reward_funcs)
        
        # Check format
        assert output_dir.startswith("saved_models/"), f"Should start with saved_models/, got: {output_dir}"
        assert "qwen2.5-7b" in output_dir, f"Should contain model size, got: {output_dir}"
        assert "all-rewards" in output_dir, f"Should use 'all-rewards' for 4+ funcs, got: {output_dir}"
        
        print(f"✅ 7B model output: {output_dir}")
    
    def test_generate_unique_output_dir_unknown_model(self):
        """Test output directory generation for unknown model."""
        model_name = "custom/my-model-v2"
        reward_funcs = ["accuracy", "format"]
        
        output_dir = generate_unique_output_dir(model_name, reward_funcs)
        
        # Check format
        assert output_dir.startswith("saved_models/"), f"Should start with saved_models/, got: {output_dir}"
        assert "my-model-v2" in output_dir, f"Should contain cleaned model name, got: {output_dir}"
        assert "accuracy-format" in output_dir, f"Should contain reward funcs, got: {output_dir}"
        
        print(f"✅ Unknown model output: {output_dir}")
    
    def test_uniqueness_across_time(self):
        """Test that consecutive calls generate unique directories."""
        model_name = "./models/Qwen2.5-0.5B"
        reward_funcs = ["accuracy", "format"]
        
        # Generate multiple directories with small time gaps
        dirs = []
        for i in range(3):
            output_dir = generate_unique_output_dir(model_name, reward_funcs)
            dirs.append(output_dir)
            time.sleep(1)  # Ensure different timestamps
        
        # All should be unique
        assert len(set(dirs)) == 3, f"Expected 3 unique dirs, got: {dirs}"
        
        # All should follow the same pattern except timestamp
        for i, dir_name in enumerate(dirs):
            print(f"✅ Unique dir {i+1}: {dir_name}")
    
    def test_reward_function_combinations(self):
        """Test different reward function combinations."""
        model_name = "./models/Qwen2.5-0.5B"
        
        test_cases = [
            (["accuracy"], "accuracy"),
            (["accuracy", "format"], "accuracy-format"),
            (["accuracy", "format", "reasoning"], "accuracy-format-reasoning"),
            (["accuracy", "format", "reasoning", "cosine"], "all-rewards"),
            (["a", "b", "c", "d", "e"], "all-rewards"),
        ]
        
        for reward_funcs, expected_substr in test_cases:
            output_dir = generate_unique_output_dir(model_name, reward_funcs)
            assert expected_substr in output_dir, f"Expected '{expected_substr}' in '{output_dir}'"
            print(f"✅ Reward funcs {reward_funcs} → {expected_substr}")
    
    def test_model_size_extraction(self):
        """Test model size extraction from various model names."""
        reward_funcs = ["accuracy"]
        
        test_cases = [
            ("./models/Qwen2.5-0.5B", "qwen2.5-0.5b"),
            ("Qwen/Qwen2.5-1B", "qwen2.5-1b"),
            ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5-3b"),
            ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b"),
            ("Qwen/Qwen2.5-14B", "qwen2.5-14b"),
            ("custom/weird-model-name", "weird-model-name"),
        ]
        
        for model_name, expected_size in test_cases:
            output_dir = generate_unique_output_dir(model_name, reward_funcs)
            assert expected_size in output_dir, f"Expected '{expected_size}' in '{output_dir}' for model '{model_name}'"
            print(f"✅ Model '{model_name}' → size '{expected_size}'")
    
    def test_directory_format_compliance(self):
        """Test that generated directories follow filesystem naming conventions."""
        model_name = "./models/Qwen2.5-0.5B"
        reward_funcs = ["accuracy", "format", "reasoning_steps"]
        
        output_dir = generate_unique_output_dir(model_name, reward_funcs)
        
        # Should not contain problematic characters
        problematic_chars = ['<', '>', ':', '"', '|', '?', '*', ' ']
        for char in problematic_chars:
            assert char not in output_dir, f"Directory name should not contain '{char}': {output_dir}"
        
        # Should be a valid relative path
        assert os.path.normpath(output_dir) == output_dir, f"Should be normalized path: {output_dir}"
        
        print(f"✅ Directory format compliant: {output_dir}")


def test_integration_with_training_args():
    """Test integration with actual training script logic."""
    # Mock datetime to get predictable output
    with patch('train_grpo.datetime') as mock_datetime:
        mock_datetime.datetime.now.return_value.strftime.return_value = "20250106_143022"
        
        model_name = "./models/Qwen2.5-0.5B"
        reward_funcs = ["accuracy", "format", "reasoning_steps"]
        
        output_dir = generate_unique_output_dir(model_name, reward_funcs)
        expected = "saved_models/qwen2.5-0.5b_accuracy-format-reasoning_20250106_143022"
        
        assert output_dir == expected, f"Expected: {expected}, Got: {output_dir}"
        print(f"✅ Integration test passed: {output_dir}")


if __name__ == "__main__":
    print("🧪 Testing checkpoint naming functionality...")
    
    test_instance = TestCheckpointNaming()
    
    # Run all tests
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"\n--- {method_name} ---")
        method = getattr(test_instance, method_name)
        method()
    
    print("\n--- integration test ---")
    test_integration_with_training_args()
    
    print(f"\n🎉 All {len(test_methods) + 1} tests passed!")
    print("\n📁 Example output directories:")
    print("   saved_models/qwen2.5-0.5b_accuracy-format-reasoning_20250106_143022")
    print("   saved_models/qwen2.5-7b_all-rewards_20250106_143515") 
    print("   saved_models/custom-model_accuracy_20250106_144201")