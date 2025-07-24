#!/usr/bin/env python3
"""
Test suite for configuration profiles functionality.
Tests that _profiles in config.yaml work correctly with the new configuration structure.
"""

import pytest
import sys
import os
from typing import Dict, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import ConfigManager, Config, ValidationError


class TestConfigProfiles:
    """Test class for configuration profiles functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config_file = "config.yaml"
        self.config_manager = ConfigManager(
            config_file=self.config_file,
            enable_legacy_compatibility=True
        )
    
    def test_default_profile_loads(self):
        """Test that default profile (no profile specified) loads correctly."""
        config = self.config_manager.load_config()
        
        # Check that default values are loaded
        assert config.TrainingArguments.num_train_epochs == 1.0
        assert config.TrainingArguments.per_device_train_batch_size == 8
        assert config.TrainingArguments.logging_steps == 10
        assert config.TrainingArguments.save_steps == 50
        assert config.model.name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.GRPOConfig.use_vllm == False
        assert config.monitoring.wandb.enabled == False
        
    def test_dev_profile_loads(self):
        """Test that 'dev' profile loads correctly."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Check dev profile specific values
        assert config.TrainingArguments.num_train_epochs == 1.0
        assert config.TrainingArguments.per_device_train_batch_size == 8
        assert config.TrainingArguments.logging_steps == 5  # Changed from default 10
        assert config.TrainingArguments.save_steps == 25    # Changed from default 50
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"  # Changed from default
        assert config.GRPOConfig.use_vllm == True  # Changed from default False
        assert config.monitoring.wandb.enabled == False
        assert config.monitoring.logging.profiling_mode == False
        assert config.callbacks.checkpoint_preservation.enabled == False
        
    def test_prod_profile_loads(self):
        """Test that 'prod' profile loads correctly."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="prod",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Check prod profile specific values
        assert config.TrainingArguments.num_train_epochs == 3.0  # Changed from default 1.0
        assert config.TrainingArguments.per_device_train_batch_size == 8
        assert config.TrainingArguments.learning_rate == 3.0e-5  # Changed from default 5e-5
        assert config.model.name == "Qwen/Qwen2.5-7B-Instruct"  # Changed from default
        assert config.monitoring.wandb.enabled == True  # Changed from default False
        assert config.monitoring.logging.profiling_mode == False
        assert config.callbacks.comprehensive_logging.enabled == False
        assert config.callbacks.checkpoint_preservation.enabled == True
        assert config.callbacks.checkpoint_preservation.every_n_steps == 1000
        
    def test_test_profile_loads(self):
        """Test that 'test' profile loads correctly."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="test",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Check test profile specific values
        assert config.TrainingArguments.num_train_epochs == 0.01  # Very small for testing
        assert config.TrainingArguments.per_device_train_batch_size == 8
        assert config.TrainingArguments.logging_steps == 1  # Changed from default 10
        assert config.TrainingArguments.save_steps == 10    # Changed from default 50
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"  # Changed from default
        assert config.monitoring.wandb.enabled == False
        assert config.callbacks.checkpoint_preservation.enabled == False
        assert config.dataset.split.test_size == 0.05  # Changed from default 0.1
        assert config.rewards.functions == ["format"]  # Changed from default
        
    def test_profile_profile_loads(self):
        """Test that 'profile' profile loads correctly."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="profile",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Check profile profile specific values
        assert config.TrainingArguments.num_train_epochs == 0.1  # Small for profiling
        assert config.TrainingArguments.per_device_train_batch_size == 8
        assert config.TrainingArguments.logging_steps == 1  # Changed from default 10
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"  # Changed from default
        assert config.monitoring.wandb.enabled == False
        assert config.monitoring.logging.profiling_mode == True  # Changed from default
        assert config.callbacks.comprehensive_logging.enabled == True  # Changed from default
        assert config.callbacks.comprehensive_logging.log_examples == True
        assert config.callbacks.checkpoint_preservation.enabled == False
        
    def test_invalid_profile_uses_default(self):
        """Test that invalid profile name logs warning and uses default."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="nonexistent",
            enable_legacy_compatibility=True
        )
        
        # Should not raise error, but use default configuration
        config = config_manager.load_config()
        
        # Should have default values (same as no profile specified)
        assert config.TrainingArguments.num_train_epochs == 1.0
        assert config.TrainingArguments.logging_steps == 10
        assert config.model.name == "Qwen/Qwen2.5-3B-Instruct"
        assert config.GRPOConfig.use_vllm == False
            
    def test_profile_with_cli_overrides(self):
        """Test that CLI overrides work correctly with profiles."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        
        # Override some values from dev profile
        config = config_manager.load_config(cli_args=[
            "--TrainingArguments.per_device_train_batch_size=16",
            "--GRPOConfig.max_completion_length=2048"
        ])
        
        # Check that profile values are applied first, then CLI overrides
        assert config.TrainingArguments.logging_steps == 5  # From dev profile
        assert config.TrainingArguments.save_steps == 25    # From dev profile
        assert config.TrainingArguments.per_device_train_batch_size == 16  # CLI override
        assert config.GRPOConfig.max_completion_length == 2048  # CLI override
        assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"  # From dev profile
        
    def test_profile_precedence_order(self):
        """Test that configuration precedence works correctly: CLI > Profile > Default."""
        # Test with dev profile and CLI override
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        
        config = config_manager.load_config(cli_args=[
            "--TrainingArguments.logging_steps=100"  # Override dev profile value of 5
        ])
        
        # Values should be: CLI (100) > Profile (5) > Default (10)
        assert config.TrainingArguments.logging_steps == 100  # CLI override wins
        assert config.TrainingArguments.save_steps == 25      # Profile value wins over default
        assert config.TrainingArguments.per_device_train_batch_size == 8  # Default value (same in profile)
        
    def test_nested_profile_overrides(self):
        """Test that nested configuration overrides work correctly in profiles."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="prod",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Test that nested overrides work
        assert config.callbacks.checkpoint_preservation.enabled == True
        assert config.callbacks.checkpoint_preservation.every_n_steps == 1000
        
        # Test that non-overridden nested values keep defaults
        assert config.callbacks.checkpoint_preservation.directory == "permanent_checkpoints"
        
    def test_profile_with_training_arguments_overrides(self):
        """Test that TrainingArguments overrides work correctly in profiles."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="prod",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Check that TrainingArguments fields are overridden correctly
        assert config.TrainingArguments.num_train_epochs == 3.0
        assert config.TrainingArguments.learning_rate == 3.0e-5
        assert config.TrainingArguments.per_device_train_batch_size == 8
        
        # Check that non-overridden TrainingArguments keep defaults
        assert config.TrainingArguments.warmup_ratio == 0.1
        assert config.TrainingArguments.weight_decay == 0.01
        assert config.TrainingArguments.bf16 == True
        
    def test_profile_with_grpo_config_overrides(self):
        """Test that GRPOConfig overrides work correctly in profiles."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="dev",
            enable_legacy_compatibility=True
        )
        config = config_manager.load_config()
        
        # Check that GRPOConfig fields are overridden correctly
        assert config.GRPOConfig.use_vllm == True  # Overridden in dev profile
        
        # Check that non-overridden GRPOConfig keep defaults
        assert config.GRPOConfig.max_completion_length == 1024
        assert config.GRPOConfig.num_generations == 8
        assert config.GRPOConfig.vllm_mode == "colocate"
        assert config.GRPOConfig.log_completions == True
        assert config.GRPOConfig.wandb_log_unique_prompts == True
        
    def test_profile_validation(self):
        """Test that profile configurations are validated correctly."""
        config_manager = ConfigManager(
            config_file=self.config_file,
            profile="test",
            enable_legacy_compatibility=True
        )
        
        # Test profile should load successfully (batch size 8 is valid)
        config = config_manager.load_config()
        assert config.TrainingArguments.per_device_train_batch_size == 8
        
        # Test that invalid overrides still fail validation
        with pytest.raises(ValidationError, match="Effective batch size"):
            config_manager.load_config(cli_args=[
                "--TrainingArguments.per_device_train_batch_size=3"  # Invalid for GRPO
            ])
            
    def test_all_profiles_load_successfully(self):
        """Test that all defined profiles can load without errors."""
        profiles = ["dev", "prod", "test", "profile"]
        
        for profile_name in profiles:
            config_manager = ConfigManager(
                config_file=self.config_file,
                profile=profile_name,
                enable_legacy_compatibility=True
            )
            
            # Should not raise any exceptions
            config = config_manager.load_config()
            assert isinstance(config, Config)
            print(f"✅ Profile '{profile_name}' loaded successfully")
            
    def test_profile_specific_values_are_different(self):
        """Test that profiles actually change configuration values."""
        # Load default config
        default_config = ConfigManager(
            config_file=self.config_file,
            enable_legacy_compatibility=True
        ).load_config()
        
        # Load dev profile
        dev_config = ConfigManager(
            config_file=self.config_file,
            profile="dev",
            enable_legacy_compatibility=True
        ).load_config()
        
        # Load prod profile
        prod_config = ConfigManager(
            config_file=self.config_file,
            profile="prod",
            enable_legacy_compatibility=True
        ).load_config()
        
        # Verify that profiles actually change values
        assert dev_config.TrainingArguments.logging_steps != default_config.TrainingArguments.logging_steps
        assert dev_config.model.name != default_config.model.name
        assert dev_config.GRPOConfig.use_vllm != default_config.GRPOConfig.use_vllm
        
        assert prod_config.TrainingArguments.num_train_epochs != default_config.TrainingArguments.num_train_epochs
        assert prod_config.TrainingArguments.learning_rate != default_config.TrainingArguments.learning_rate
        assert prod_config.monitoring.wandb.enabled != default_config.monitoring.wandb.enabled
        
        print("✅ All profiles have different values from default")


def run_profile_tests():
    """Run all profile tests and report results."""
    print("🧪 Running Configuration Profile Tests...")
    print("=" * 60)
    
    test_class = TestConfigProfiles()
    test_class.setup_method()
    
    tests = [
        ("Default Profile", test_class.test_default_profile_loads),
        ("Dev Profile", test_class.test_dev_profile_loads),
        ("Prod Profile", test_class.test_prod_profile_loads),
        ("Test Profile", test_class.test_test_profile_loads),
        ("Profile Profile", test_class.test_profile_profile_loads),
        ("Invalid Profile Uses Default", test_class.test_invalid_profile_uses_default),
        ("Profile + CLI Overrides", test_class.test_profile_with_cli_overrides),
        ("Profile Precedence", test_class.test_profile_precedence_order),
        ("Nested Overrides", test_class.test_nested_profile_overrides),
        ("TrainingArguments Overrides", test_class.test_profile_with_training_arguments_overrides),
        ("GRPOConfig Overrides", test_class.test_profile_with_grpo_config_overrides),
        ("Profile Validation", test_class.test_profile_validation),
        ("All Profiles Load", test_class.test_all_profiles_load_successfully),
        ("Profile Value Differences", test_class.test_profile_specific_values_are_different),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            print(f"✅ {test_name}: PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            failed += 1
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All profile tests passed!")
        return True
    else:
        print("💥 Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_profile_tests()
    sys.exit(0 if success else 1)