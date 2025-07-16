#!/usr/bin/env python3
"""
Test script for the new training script with configuration system.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_configuration_loading():
    """Test that the configuration loads properly."""
    print("Testing configuration loading...")
    
    try:
        from src.config import ConfigManager
        
        # Test basic configuration loading
        config_manager = ConfigManager(config_file="../config.yaml")
        config = config_manager.load_config()
        
        print(f"✅ Configuration loaded successfully")
        print(f"   Model: {config.model.name}")
        print(f"   Reward functions: {config.rewards.functions}")
        print(f"   Learning rate: {config.training.optimization.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        return False


def test_reward_functions():
    """Test that reward functions can be loaded."""
    print("\nTesting reward functions...")
    
    try:
        from src.config import ConfigManager
        from src.rewards.openr1_rewards import get_reward_funcs
        from src.config.grpo_config import GRPOScriptArguments
        
        # Load configuration
        config_manager = ConfigManager(config_file="../config.yaml")
        config = config_manager.load_config()
        
        # Create script args for reward functions
        script_args = GRPOScriptArguments(
            reward_funcs=config.rewards.functions,
            cosine_min_value_wrong=config.rewards.cosine.min_value_wrong,
            cosine_max_value_wrong=config.rewards.cosine.max_value_wrong,
            cosine_min_value_correct=config.rewards.cosine.min_value_correct,
            cosine_max_value_correct=config.rewards.cosine.max_value_correct,
            cosine_max_len=config.rewards.cosine.max_len,
            repetition_n_grams=config.rewards.repetition.n_grams,
            repetition_max_penalty=config.rewards.repetition.max_penalty,
            code_language=config.rewards.code.language,
            max_completion_len=config.rewards.soft_punish.max_completion_len,
            soft_punish_cache=config.rewards.soft_punish.cache,
        )
        
        # Get reward functions
        reward_functions = get_reward_funcs(script_args)
        
        print(f"✅ Reward functions loaded successfully")
        print(f"   Number of functions: {len(reward_functions)}")
        print(f"   Functions: {config.rewards.functions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward functions loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_overrides():
    """Test CLI overrides work with the new system."""
    print("\nTesting CLI overrides...")
    
    try:
        from src.config import ConfigManager
        
        # Test with CLI overrides
        cli_args = [
            "--model.name", "Qwen/Qwen2.5-7B",
            "--rewards.functions", "[accuracy,format,reasoning_steps]",
            "--training.optimization.learning_rate", "1e-4"
        ]
        
        config_manager = ConfigManager(config_file="../config.yaml")
        config = config_manager.load_config(cli_args=cli_args)
        
        print(f"✅ CLI overrides work correctly")
        print(f"   Model: {config.model.name}")
        print(f"   Reward functions: {config.rewards.functions}")
        print(f"   Learning rate: {config.training.optimization.learning_rate}")
        
        return True
        
    except Exception as e:
        print(f"❌ CLI overrides failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING TRAIN_GRPO_NEW.PY COMPONENTS")
    print("="*60)
    
    tests = [
        test_configuration_loading,
        test_reward_functions,
        test_cli_overrides,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"✅ ALL TESTS PASSED! ({passed}/{len(tests)})")
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{len(tests)} passed)")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)