"""
Configuration system for DeepSeek R1 Zero GRPO training.

This module provides a centralized configuration system with:
- Hierarchical YAML configuration
- CLI argument overrides with dot notation
- Array syntax with brackets: [item1,item2,item3]
- Type validation and business logic checks
- Backwards compatibility with legacy arguments
- Environment variable support
- Configuration profiles (dev, prod, test, etc.)

Usage:
    from src.config import ConfigManager
    
    manager = ConfigManager("config.yaml", profile="dev")
    config = manager.load_config(cli_args=sys.argv[1:])
    
    # Access configuration
    print(config.model.name)
    print(config.training.optimization.learning_rate)
    print(config.rewards.functions)
"""

from .manager import ConfigManager, create_config_manager
from .schemas import Config, get_config_field_types, get_array_fields
from .validator import ConfigValidator, ValidationError
from .overrides import OverrideHandler, LegacyArgumentHandler

__all__ = [
    "ConfigManager",
    "create_config_manager",
    "Config",
    "ConfigValidator",
    "ValidationError",
    "OverrideHandler",
    "LegacyArgumentHandler",
    "get_config_field_types",
    "get_array_fields",
]