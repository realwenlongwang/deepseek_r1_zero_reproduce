#!/usr/bin/env python3
"""
Test script for the new configuration system.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ConfigManager, ValidationError


def test_basic_config_loading():
    """Test basic configuration loading."""
    print("Testing basic configuration loading...")
    
    config_manager = ConfigManager(config_file="config.yaml")
    config = config_manager.load_config()
    
    print(f"✅ Model name: {config.model.name}")
    print(f"✅ Learning rate: {config.training.optimization.learning_rate}")
    print(f"✅ Reward functions: {config.rewards.functions}")
    print(f"✅ Wandb enabled: {config.monitoring.wandb.enabled}")
    
    return config


def test_cli_overrides():
    """Test CLI argument overrides."""
    print("\nTesting CLI overrides...")
    
    # Test CLI arguments
    cli_args = [
        "--model.name", "Qwen/Qwen2.5-7B",
        "--training.optimization.learning_rate", "1e-4",
        "--rewards.functions", "[accuracy,format,reasoning_steps]",
        "--monitoring.wandb.enabled", "false"
    ]
    
    config_manager = ConfigManager(config_file="config.yaml")
    config = config_manager.load_config(cli_args=cli_args)
    
    print(f"✅ Overridden model name: {config.model.name}")
    print(f"✅ Overridden learning rate: {config.training.optimization.learning_rate}")
    print(f"✅ Overridden reward functions: {config.rewards.functions}")
    print(f"✅ Overridden wandb enabled: {config.monitoring.wandb.enabled}")
    
    return config


def test_array_syntax():
    """Test array syntax parsing."""
    print("\nTesting array syntax...")
    
    # Test different array formats
    test_cases = [
        ["--rewards.functions", "[accuracy,format]"],
        ["--rewards.functions=[accuracy,format,reasoning_steps]"],
        ["--rewards.functions", "[accuracy, format, reasoning_steps]"],
    ]
    
    for i, cli_args in enumerate(test_cases):
        print(f"Test case {i+1}: {cli_args}")
        config_manager = ConfigManager(config_file="config.yaml")
        config = config_manager.load_config(cli_args=cli_args)
        print(f"  Result: {config.rewards.functions}")
    
    return config


def test_profiles():
    """Test configuration profiles."""
    print("\nTesting configuration profiles...")
    
    profiles = ["default", "dev", "prod", "test"]
    
    for profile in profiles:
        print(f"Testing profile: {profile}")
        config_manager = ConfigManager(config_file="config.yaml", profile=profile)
        config = config_manager.load_config()
        
        print(f"  Model: {config.model.name}")
        print(f"  Epochs: {config.training.epochs}")
        print(f"  Wandb: {config.monitoring.wandb.enabled}")
        print(f"  Batch size: {config.training.batch_size.per_device_train}")
        print()


def test_legacy_compatibility():
    """Test legacy argument compatibility."""
    print("\nTesting legacy compatibility...")
    
    # Test legacy arguments
    legacy_args = [
        "--model_name", "Qwen/Qwen2.5-7B",
        "--learning_rate", "1e-4",
        "--per_device_train_batch_size", "8",
        "--reward_funcs", "accuracy,format,reasoning_steps",
        "--no_wandb"
    ]
    
    config_manager = ConfigManager(config_file="config.yaml", enable_legacy_compatibility=True)
    config = config_manager.load_config(cli_args=legacy_args)
    
    print(f"✅ Legacy model name: {config.model.name}")
    print(f"✅ Legacy learning rate: {config.training.optimization.learning_rate}")
    print(f"✅ Legacy batch size: {config.training.batch_size.per_device_train}")
    print(f"✅ Legacy reward functions: {config.rewards.functions}")
    print(f"✅ Legacy no_wandb: {config.monitoring.wandb.enabled}")
    
    return config


def test_validation():
    """Test configuration validation."""
    print("\nTesting configuration validation...")
    
    try:
        # Test invalid configuration
        cli_args = [
            "--training.optimization.learning_rate", "-1",  # Invalid: negative learning rate
            "--training.batch_size.per_device_train", "7",  # Invalid: not divisible by 8
        ]
        
        config_manager = ConfigManager(config_file="config.yaml")
        config = config_manager.load_config(cli_args=cli_args)
        print("❌ Validation should have failed!")
        
    except ValidationError as e:
        print(f"✅ Validation correctly failed: {str(e)[:100]}...")
    
    try:
        # Test valid configuration
        cli_args = [
            "--training.optimization.learning_rate", "1e-4",
            "--training.batch_size.per_device_train", "8",
        ]
        
        config_manager = ConfigManager(config_file="config.yaml")
        config = config_manager.load_config(cli_args=cli_args)
        print("✅ Valid configuration passed validation")
        
    except ValidationError as e:
        print(f"❌ Valid configuration failed: {str(e)[:100]}...")


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING NEW CONFIGURATION SYSTEM")
    print("="*60)
    
    try:
        test_basic_config_loading()
        test_cli_overrides()
        test_array_syntax()
        test_profiles()
        test_legacy_compatibility()
        test_validation()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()