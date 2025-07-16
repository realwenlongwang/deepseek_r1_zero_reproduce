"""
Configuration manager for centralized YAML configuration with CLI overrides.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import asdict, fields, is_dataclass
from .schemas import Config
from .validator import ConfigValidator, ValidationError
from .overrides import OverrideHandler, LegacyArgumentHandler


logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Central configuration manager that handles YAML loading, validation, and CLI overrides.
    
    Features:
    - Hierarchical YAML configuration
    - CLI argument overrides with dot notation
    - Array syntax with brackets: [item1,item2,item3]
    - Type validation and business logic checks
    - Backwards compatibility with legacy arguments
    - Environment variable support
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 profile: str = "default",
                 enable_legacy_compatibility: bool = True):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to YAML configuration file
            profile: Configuration profile to use (default, dev, prod)
            enable_legacy_compatibility: Whether to support legacy CLI arguments
        """
        self.config_file = config_file or "config.yaml"
        self.profile = profile
        self.enable_legacy_compatibility = enable_legacy_compatibility
        
        self.validator = ConfigValidator()
        self.override_handler = OverrideHandler()
        self.legacy_handler = LegacyArgumentHandler() if enable_legacy_compatibility else None
        
        # Store loaded configuration
        self._config_dict: Optional[Dict[str, Any]] = None
        self._config_object: Optional[Config] = None
    
    def load_config(self, 
                   cli_args: Optional[List[str]] = None,
                   env_overrides: Optional[Dict[str, str]] = None) -> Config:
        """
        Load and validate configuration from YAML file, CLI arguments, and environment variables.
        
        Args:
            cli_args: CLI arguments to override configuration
            env_overrides: Environment variable overrides
            
        Returns:
            Validated configuration object
        """
        # Load base configuration from YAML
        config_dict = self._load_yaml_config()
        
        # Apply profile-specific overrides
        config_dict = self._apply_profile_overrides(config_dict)
        
        # Apply environment variable overrides
        if env_overrides:
            config_dict = self._apply_env_overrides(config_dict, env_overrides)
        
        # Apply CLI overrides
        if cli_args:
            config_dict = self._apply_cli_overrides(config_dict, cli_args)
        
        # Convert to Config object
        config = self._dict_to_config(config_dict)
        
        # Validate configuration
        warnings = self.validator.validate_config(config)
        if warnings:
            logger.warning("Configuration validation warnings:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        # Store for later access
        self._config_dict = config_dict
        self._config_object = config
        
        return config
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load base configuration from YAML file."""
        if not os.path.exists(self.config_file):
            logger.info(f"Configuration file '{self.config_file}' not found. Using defaults.")
            return {}
        
        try:
            with open(self.config_file, 'r') as file:
                config_dict = yaml.safe_load(file) or {}
            
            logger.info(f"Loaded configuration from '{self.config_file}'")
            return config_dict
        
        except yaml.YAMLError as e:
            raise ValidationError(f"Invalid YAML in configuration file '{self.config_file}': {e}")
        except Exception as e:
            raise ValidationError(f"Failed to load configuration file '{self.config_file}': {e}")
    
    def _apply_profile_overrides(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply profile-specific configuration overrides."""
        if self.profile == "default":
            return config_dict
        
        profiles = config_dict.get("_profiles", {})
        if self.profile not in profiles:
            logger.warning(f"Profile '{self.profile}' not found in configuration. Using default.")
            return config_dict
        
        profile_overrides = profiles[self.profile]
        logger.info(f"Applying profile overrides for '{self.profile}'")
        
        # Apply profile overrides using the same mechanism as CLI overrides
        return self.override_handler.apply_overrides(config_dict, profile_overrides)
    
    def _apply_env_overrides(self, config_dict: Dict[str, Any], env_overrides: Dict[str, str]) -> Dict[str, Any]:
        """Apply environment variable overrides."""
        if not env_overrides:
            return config_dict
        
        # Convert environment variables to configuration overrides
        converted_overrides = {}
        for env_var, value in env_overrides.items():
            # Convert GRPO_TRAINING_LEARNING_RATE to training.learning_rate
            if env_var.startswith("GRPO_"):
                config_key = env_var[5:].lower().replace("_", ".")
                converted_overrides[config_key] = value
        
        if converted_overrides:
            logger.info(f"Applying {len(converted_overrides)} environment variable overrides")
            
            # Parse values using the same logic as CLI overrides
            parsed_overrides = {}
            for key, value in converted_overrides.items():
                parsed_overrides[key] = self.override_handler._parse_value(key, value)
            
            return self.override_handler.apply_overrides(config_dict, parsed_overrides)
        
        return config_dict
    
    def _apply_cli_overrides(self, config_dict: Dict[str, Any], cli_args: List[str]) -> Dict[str, Any]:
        """Apply CLI argument overrides."""
        if not cli_args:
            return config_dict
        
        # Handle legacy arguments if enabled
        if self.enable_legacy_compatibility:
            cli_args = self.legacy_handler.convert_legacy_args(cli_args)
        
        # Parse CLI overrides
        try:
            overrides = self.override_handler.parse_cli_overrides(cli_args)
            
            if overrides:
                logger.info(f"Applying {len(overrides)} CLI overrides")
                for key, value in overrides.items():
                    logger.debug(f"  {key} = {value}")
                
                return self.override_handler.apply_overrides(config_dict, overrides)
        
        except Exception as e:
            raise ValidationError(f"Failed to parse CLI arguments: {e}")
        
        return config_dict
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert configuration dictionary to Config dataclass object."""
        try:
            return self._convert_dict_to_dataclass(Config, config_dict)
        except Exception as e:
            raise ValidationError(f"Failed to convert configuration dictionary to Config object: {e}")
    
    def _convert_dict_to_dataclass(self, cls, data: Dict[str, Any]):
        """Recursively convert dictionary to dataclass."""
        if not isinstance(data, dict):
            return data
        
        # Get default instance to fill missing fields
        default_instance = cls()
        
        kwargs = {}
        for field in fields(cls):
            field_name = field.name
            field_type = field.type
            
            if field_name in data:
                field_value = data[field_name]
                
                # Handle nested dataclasses
                if is_dataclass(field_type):
                    # Get default value for nested dataclass
                    default_nested = getattr(default_instance, field_name)
                    nested_dict = asdict(default_nested)
                    
                    # Update with provided values
                    if isinstance(field_value, dict):
                        nested_dict.update(field_value)
                    
                    kwargs[field_name] = self._convert_dict_to_dataclass(field_type, nested_dict)
                else:
                    kwargs[field_name] = field_value
            else:
                # Use default value
                kwargs[field_name] = getattr(default_instance, field_name)
        
        return cls(**kwargs)
    
    def save_config(self, config: Config, output_file: str):
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration object to save
            output_file: Output file path
        """
        try:
            config_dict = asdict(config)
            
            # Remove private fields that shouldn't be saved
            config_dict = self._clean_config_dict(config_dict)
            
            with open(output_file, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Configuration saved to '{output_file}'")
        
        except Exception as e:
            raise ValidationError(f"Failed to save configuration to '{output_file}': {e}")
    
    def _clean_config_dict(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Clean configuration dictionary for saving."""
        # Remove private fields and None values
        cleaned = {}
        for key, value in config_dict.items():
            if key.startswith('_'):
                continue
            
            if isinstance(value, dict):
                cleaned_value = self._clean_config_dict(value)
                if cleaned_value:  # Only include non-empty dictionaries
                    cleaned[key] = cleaned_value
            elif value is not None:
                cleaned[key] = value
        
        return cleaned
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the loaded configuration as a dictionary."""
        return self._config_dict.copy() if self._config_dict else {}
    
    def get_config_object(self) -> Optional[Config]:
        """Get the loaded configuration as a Config object."""
        return self._config_object
    
    def print_config(self, config: Optional[Config] = None):
        """Print configuration in a readable format."""
        if config is None:
            config = self._config_object
        
        if config is None:
            logger.error("No configuration loaded")
            return
        
        print("="*80)
        print("CONFIGURATION SUMMARY")
        print("="*80)
        
        config_dict = asdict(config)
        self._print_config_section(config_dict, "")
        
        print("="*80)
    
    def _print_config_section(self, config_dict: Dict[str, Any], prefix: str, indent: int = 0):
        """Recursively print configuration sections."""
        for key, value in config_dict.items():
            if key.startswith('_'):
                continue
            
            full_key = f"{prefix}.{key}" if prefix else key
            indent_str = "  " * indent
            
            if isinstance(value, dict):
                print(f"{indent_str}{key}:")
                self._print_config_section(value, full_key, indent + 1)
            else:
                print(f"{indent_str}{key}: {value}")


def create_config_manager(config_file: Optional[str] = None, 
                         profile: str = "default",
                         cli_args: Optional[List[str]] = None) -> ConfigManager:
    """
    Convenience function to create and configure a ConfigManager.
    
    Args:
        config_file: Path to configuration file
        profile: Configuration profile
        cli_args: CLI arguments
        
    Returns:
        Configured ConfigManager instance
    """
    manager = ConfigManager(config_file=config_file, profile=profile)
    
    # Load environment variables with GRPO_ prefix
    env_overrides = {k: v for k, v in os.environ.items() if k.startswith("GRPO_")}
    
    # Load configuration
    config = manager.load_config(cli_args=cli_args, env_overrides=env_overrides)
    
    return manager