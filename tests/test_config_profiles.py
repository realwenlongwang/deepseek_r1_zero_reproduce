#!/usr/bin/env python3
"""
Test suite for configuration profiles functionality.
Run with: pytest tests/test_config_profiles.py -v
"""

import pytest
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import ConfigManager, Config, ValidationError


class TestConfigProfiles:
    """Test class for configuration profiles functionality."""
    
    @pytest.fixture
    def config_file(self):
        """Fixture for config file path."""
        return "config.yaml"
    
    @pytest.fixture
    def default_config_manager(self, config_file):
        """Fixture for default config manager."""
        return ConfigManager(
            config_file=config_file,
            enable_legacy_compatibility=True
        )
    
    def test_default_profile_loads(self, default_config_manager):
        """Test that default profile loads correctly."""
        config = default_config_manager.load_config()
        
        assert config.TrainingArguments.num_train_epochs == 1.0
        assert config.TrainingArguments.per_device_train_batch_size == 8
        assert config.TrainingArguments.logging_steps == 10
        assert config.model.name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.GRPOConfig.use_vllm == False
        
    def test_dev_profile_overrides(self, config_file):
        """Test that dev profile overrides work correctly."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        assert config.TrainingArguments.logging_steps == 5  # Overridden
        assert config.TrainingArguments.save_steps == 25    # Overridden
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"  # Overridden
        assert config.GRPOConfig.use_vllm == True  # Overridden
        
    def test_prod_profile_overrides(self, config_file):
        """Test that prod profile overrides work correctly."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="prod",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        assert config.TrainingArguments.num_train_epochs == 3.0  # Overridden
        assert config.TrainingArguments.learning_rate == 3.0e-5  # Overridden
        assert config.model.name == "Qwen/Qwen2.5-7B-Instruct"  # Overridden
        assert config.monitoring.wandb.enabled == True  # Overridden
        
    def test_test_profile_overrides(self, config_file):
        """Test that test profile overrides work correctly."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="test",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        assert config.TrainingArguments.num_train_epochs == 0.01  # Overridden
        assert config.dataset.split.test_size == 0.05  # Overridden
        assert config.rewards.functions == ["format"]  # Overridden
        
    def test_profile_with_cli_overrides(self, config_file):
        """Test that CLI overrides work with profiles."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        
        config = config_manager.load_config(cli_args=[
            "--TrainingArguments.per_device_train_batch_size=16",
            "--GRPOConfig.max_completion_length=2048"
        ])
        
        # Profile values should be applied
        assert config.TrainingArguments.logging_steps == 5  # From dev profile
        # CLI overrides should win
        assert config.TrainingArguments.per_device_train_batch_size == 16  # CLI override
        assert config.GRPOConfig.max_completion_length == 2048  # CLI override
        
    def test_invalid_profile_uses_default(self, config_file):
        """Test that invalid profile falls back to default."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="nonexistent",
            enable_legacy_compatibility=True
        )
        
        config = config_manager.load_config()
        
        # Should have default values
        assert config.TrainingArguments.num_train_epochs == 1.0
        assert config.TrainingArguments.logging_steps == 10
        assert config.model.name == "Qwen/Qwen2.5-3B-Instruct"
        
    @pytest.mark.parametrize("profile", ["dev", "prod", "test", "profile"])
    def test_all_profiles_load_successfully(self, config_file, profile):
        """Test that all defined profiles can load without errors."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile=profile,
            enable_legacy_compatibility=True
        )
        
        config = config_manager.load_config()
        assert isinstance(config, Config)
        
    def test_profile_precedence_order(self, config_file):
        """Test configuration precedence: CLI > Profile > Default."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        
        config = config_manager.load_config(cli_args=[
            "--TrainingArguments.logging_steps=100"  # Override dev profile value of 5
        ])
        
        # CLI should win over profile
        assert config.TrainingArguments.logging_steps == 100  # CLI override
        # Profile should win over default
        assert config.TrainingArguments.save_steps == 25  # From dev profile
        
    def test_nested_config_overrides(self, config_file):
        """Test that nested configuration overrides work."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="prod",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Test nested overrides
        assert config.callbacks.checkpoint_preservation.enabled == True
        assert config.callbacks.checkpoint_preservation.every_n_steps == 1000
        # Non-overridden nested values should keep defaults
        assert config.callbacks.checkpoint_preservation.directory == "permanent_checkpoints"
        
    def test_new_config_structure_with_profiles(self, config_file):
        """Test that new TrainingArguments and GRPOConfig structure works with profiles."""
        config_manager = ConfigManager(
            config_file=config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Test TrainingArguments overrides
        assert hasattr(config, 'TrainingArguments')
        assert config.TrainingArguments.logging_steps == 5
        
        # Test GRPOConfig overrides
        assert hasattr(config, 'GRPOConfig')
        assert config.GRPOConfig.use_vllm == True
        
        # Test that new fields are present
        assert hasattr(config.TrainingArguments, 'max_steps')
        assert hasattr(config.GRPOConfig, 'log_completions')
        assert hasattr(config.GRPOConfig, 'wandb_log_unique_prompts')
        assert config.GRPOConfig.log_completions == True
        assert config.GRPOConfig.wandb_log_unique_prompts == True