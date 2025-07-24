#!/usr/bin/env python3
"""
Test integer test_size functionality for dataset splitting.
"""

import pytest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.config import ConfigManager


class TestIntegerTestSize:
    """Test integer test_size functionality."""
    
    def test_config_manager_integer_test_size(self):
        """Test that ConfigManager correctly handles integer test_size."""
        # Create a temporary config with integer test_size
        config_content = """
model:
  name: "test/model"
  
dataset:
  name: "test/dataset"
  split:
    test_size: 256
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
            # Load config
            config_manager = ConfigManager(config_file=temp_config_path)
            config = config_manager.load_config()
            
            # Verify test_size is integer
            assert isinstance(config.dataset.split.test_size, int)
            assert config.dataset.split.test_size == 256
            
        finally:
            os.unlink(temp_config_path)
    
    def test_profile_config_integer_test_size(self):
        """Test that profile configurations work with integer test_size."""
        config_content = """
model:
  name: "test/model"
  
dataset:
  name: "test/dataset"
  split:
    test_size: 128
    seed: 42

TrainingArguments:
  per_device_train_batch_size: 8
  
GRPOConfig:
  max_completion_length: 512
  
rewards:
  functions: ["format"]

_profiles:
  test:
    dataset.split.test_size: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_config_path = f.name
        
        try:
            # Load config with test profile
            config_manager = ConfigManager(config_file=temp_config_path, profile="test")
            config = config_manager.load_config()
            
            # Verify profile override worked
            assert config.dataset.split.test_size == 10
            
        finally:
            os.unlink(temp_config_path)
    
    def test_ratio_calculation_logic(self):
        """Test the ratio calculation logic directly."""
        # Test normal case
        test_size = 128
        total_samples = 1000
        expected_ratio = min(test_size / total_samples, 0.9)
        assert expected_ratio == 0.128
        
        # Test capped case
        test_size = 500
        total_samples = 100
        expected_ratio = min(test_size / total_samples, 0.9)
        assert expected_ratio == 0.9
        
        # Test small dataset
        test_size = 3
        total_samples = 5
        expected_ratio = min(test_size / total_samples, 0.9)
        assert expected_ratio == 0.6
    
    def test_validation_accepts_integer_test_size(self):
        """Test that validation accepts integer test_size."""
        config_content = """
model:
  name: "test/model"
  
dataset:
  name: "test/dataset"
  split:
    test_size: 64
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
            # This should not raise any validation errors
            config_manager = ConfigManager(config_file=temp_config_path)
            config = config_manager.load_config()
            
            # Verify the value was loaded correctly
            assert config.dataset.split.test_size == 64
            
        finally:
            os.unlink(temp_config_path)
    
    def test_validation_rejects_negative_test_size(self):
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
            with pytest.raises(Exception) as exc_info:
                config_manager.load_config()
            
            assert "test_size must be a positive integer" in str(exc_info.value)
            
        finally:
            os.unlink(temp_config_path)
    
    def test_validation_rejects_zero_test_size(self):
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
            with pytest.raises(Exception) as exc_info:
                config_manager.load_config()
            
            assert "test_size must be a positive integer" in str(exc_info.value)
            
        finally:
            os.unlink(temp_config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])