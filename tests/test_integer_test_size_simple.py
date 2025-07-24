#!/usr/bin/env python3
"""
Simple test for integer test_size validation.
"""

import pytest
import os
import sys
import tempfile

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import ConfigManager
from src.config.validator import ValidationError


def test_negative_test_size_validation():
    """Test that validation rejects negative test_size."""
    config_content = """
model:
  name: "test/model"
  
dataset:
  name: "test/dataset"
  split:
    test_size: -10
    seed: 42
    
TrainingArguments:
  per_device_train_batch_size: 8
  
GRPOConfig:
  max_completion_length: 512
  
rewards:
  functions: ["format"]
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        config_manager = ConfigManager(config_file=temp_config_path)
        
        # This should raise a validation error
        with pytest.raises(ValidationError) as exc_info:
            config_manager.load_config()
        
        assert "test_size must be a positive integer" in str(exc_info.value)
        
    finally:
        os.unlink(temp_config_path)


def test_zero_test_size_validation():
    """Test that validation rejects zero test_size."""
    config_content = """
model:
  name: "test/model"
  
dataset:
  name: "test/dataset"
  split:
    test_size: 0
    seed: 42
    
TrainingArguments:
  per_device_train_batch_size: 8
  
GRPOConfig:
  max_completion_length: 512
  
rewards:
  functions: ["format"]
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(config_content)
        temp_config_path = f.name
    
    try:
        config_manager = ConfigManager(config_file=temp_config_path)
        
        # This should raise a validation error
        with pytest.raises(ValidationError) as exc_info:
            config_manager.load_config()
        
        assert "test_size must be a positive integer" in str(exc_info.value)
        
    finally:
        os.unlink(temp_config_path)


if __name__ == "__main__":
    test_negative_test_size_validation()
    test_zero_test_size_validation()
    print("All tests passed!")