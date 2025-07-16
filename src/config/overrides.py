"""
Configuration override handler for CLI arguments.
Supports hierarchical configuration overrides with bracket array syntax.
"""

import re
import shlex
from typing import List, Dict, Any, Tuple, Union
from .schemas import get_config_field_types, get_array_fields, get_boolean_fields, get_numeric_fields


class OverrideHandler:
    """
    Handles CLI argument parsing and configuration overrides.
    
    Supports:
    - Dot notation: --training.learning_rate 1e-4
    - Array notation: --rewards.functions [accuracy,format]
    - Array notation: --rewards.functions=[accuracy,format]
    - Boolean flags: --monitoring.wandb.enabled true
    """
    
    def __init__(self):
        self.field_types = get_config_field_types()
        self.array_fields = set(get_array_fields())
        self.boolean_fields = set(get_boolean_fields())
        self.numeric_fields = set(get_numeric_fields())
    
    def parse_cli_overrides(self, args: List[str]) -> Dict[str, Any]:
        """
        Parse CLI arguments into configuration overrides.
        
        Args:
            args: List of CLI arguments (e.g., sys.argv[1:])
            
        Returns:
            Dictionary of configuration overrides in dot notation
        """
        overrides = {}
        i = 0
        
        while i < len(args):
            arg = args[i]
            
            if not arg.startswith('--'):
                i += 1
                continue
            
            # Remove -- prefix
            key = arg[2:]
            
            # Handle standard CLI help
            if key in ['help', 'h']:
                self._print_help()
                import sys
                sys.exit(0)
            
            # Handle = syntax: --key=value
            if '=' in key:
                key, value = key.split('=', 1)
                overrides[key] = self._parse_value(key, value)
                i += 1
            
            # Handle space syntax: --key value
            elif i + 1 < len(args) and not args[i + 1].startswith('--'):
                value = args[i + 1]
                overrides[key] = self._parse_value(key, value)
                i += 2
            
            # Handle boolean flags: --key (implies true)
            else:
                if key in self.boolean_fields:
                    overrides[key] = True
                else:
                    raise ValueError(f"Missing value for argument: {arg}")
                i += 1
        
        return overrides
    
    def _parse_value(self, key: str, value: str) -> Any:
        """Parse a single value with type inference."""
        if key in self.array_fields:
            return self._parse_array_value(value)
        elif key in self.boolean_fields:
            return self._parse_boolean_value(value)
        elif key in self.numeric_fields:
            return self._parse_numeric_value(key, value)
        else:
            return value  # String value
    
    def _parse_array_value(self, value: str) -> List[str]:
        """
        Parse array values with bracket notation support.
        
        Supported formats:
        - [item1,item2,item3]
        - [item1, item2, item3]  (with spaces)
        - item1,item2,item3      (fallback to comma-separated)
        """
        value = value.strip()
        
        # Handle empty input
        if not value:
            return []
        
        # Check for bracket notation
        if value.startswith('[') and value.endswith(']'):
            # Validate bracket matching
            if not self._validate_brackets(value):
                raise ValueError(f"Malformed array: bracket mismatch in '{value}'")
            
            # Remove brackets and get inner content
            inner = value[1:-1].strip()
            
            # Handle empty array []
            if not inner:
                return []
            
            # Parse the inner content
            items = self._parse_quoted_csv(inner)
            return [self._clean_item(item) for item in items if item.strip()]
        
        # Fallback: comma-separated without brackets
        elif ',' in value:
            items = self._parse_quoted_csv(value)
            return [self._clean_item(item) for item in items if item.strip()]
        
        # Single value (not an array)
        else:
            return [self._clean_item(value)]
    
    def _parse_boolean_value(self, value: str) -> bool:
        """Parse boolean values from strings."""
        value = value.lower().strip()
        
        if value in ('true', '1', 'yes', 'on', 'enabled'):
            return True
        elif value in ('false', '0', 'no', 'off', 'disabled'):
            return False
        else:
            raise ValueError(f"Invalid boolean value: '{value}'. Use true/false, 1/0, yes/no, on/off, enabled/disabled")
    
    def _parse_numeric_value(self, key: str, value: str) -> Union[int, float]:
        """Parse numeric values with type inference."""
        value = value.strip()
        
        try:
            # Determine if field expects int or float
            expected_type = self.field_types.get(key, "float")
            
            if expected_type == "int":
                return int(value)
            elif expected_type == "float":
                return float(value)
            else:
                # Try to infer from the value
                if '.' in value or 'e' in value.lower():
                    return float(value)
                else:
                    return int(value)
        except ValueError:
            raise ValueError(f"Invalid numeric value for '{key}': '{value}'")
    
    def _validate_brackets(self, value: str) -> bool:
        """Validate that brackets are properly matched."""
        open_count = value.count('[')
        close_count = value.count(']')
        
        # Must have exactly one opening and one closing bracket
        if open_count != 1 or close_count != 1:
            return False
        
        # Opening bracket must come before closing bracket
        open_pos = value.find('[')
        close_pos = value.rfind(']')
        
        return open_pos < close_pos
    
    def _parse_quoted_csv(self, text: str) -> List[str]:
        """
        Parse comma-separated values with support for quoted strings.
        
        Handles:
        - Simple: item1,item2,item3
        - Quoted: "item 1","item 2",item3
        - Mixed: 'item 1',"item 2",item3
        """
        # Use shlex to handle quoted strings properly
        try:
            # Replace commas outside quotes with a special delimiter
            items = []
            current_item = ""
            in_quotes = False
            quote_char = None
            
            i = 0
            while i < len(text):
                char = text[i]
                
                if char in ('"', "'") and (i == 0 or text[i-1] != '\\'):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                    current_item += char
                elif char == ',' and not in_quotes:
                    items.append(current_item.strip())
                    current_item = ""
                else:
                    current_item += char
                
                i += 1
            
            # Add the last item
            if current_item.strip():
                items.append(current_item.strip())
            
            return items
            
        except Exception:
            # Fallback to simple split if shlex parsing fails
            return [item.strip() for item in text.split(',')]
    
    def _clean_item(self, item: str) -> str:
        """Clean and normalize a single array item."""
        item = item.strip()
        
        # Remove surrounding quotes if present
        if len(item) >= 2:
            if (item.startswith('"') and item.endswith('"')) or \
               (item.startswith("'") and item.endswith("'")):
                item = item[1:-1]
        
        return item
    
    def apply_overrides(self, config_dict: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configuration overrides to a configuration dictionary.
        
        Args:
            config_dict: Base configuration dictionary
            overrides: Override values in dot notation
            
        Returns:
            Updated configuration dictionary
        """
        # Create a deep copy to avoid modifying the original
        result = self._deep_copy_dict(config_dict)
        
        for key, value in overrides.items():
            self._set_nested_value(result, key, value)
        
        return result
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of a dictionary."""
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = self._deep_copy_dict(value)
            elif isinstance(value, list):
                result[key] = value.copy()
            else:
                result[key] = value
        return result
    
    def _print_help(self):
        """Print help message for CLI usage."""
        print("DeepSeek R1 Zero GRPO Training with Centralized Configuration")
        print("=" * 60)
        print()
        print("Usage:")
        print("  python train_grpo.py [OPTIONS]")
        print()
        print("Configuration Options:")
        print("  --profile PROFILE         Configuration profile (default, dev, prod, test)")
        print("  --config CONFIG_FILE      Path to configuration file (default: config.yaml)")
        print()
        print("Override Examples:")
        print("  --model.name \"Qwen/Qwen2.5-7B\"")
        print("  --training.optimization.learning_rate 1e-4")
        print("  --rewards.functions [accuracy,format,reasoning_steps]")
        print("  --training.batch_size.per_device_train 8")
        print("  --monitoring.wandb.enabled false")
        print()
        print("Array Syntax:")
        print("  --rewards.functions [item1,item2,item3]")
        print("  --rewards.functions=[item1,item2,item3]")
        print("  --rewards.functions [item1, item2, item3]")
        print()
        print("Legacy Compatibility:")
        print("  --model_name \"Qwen/Qwen2.5-7B\"      # Maps to --model.name")
        print("  --learning_rate 1e-4                  # Maps to --training.optimization.learning_rate")
        print("  --no_wandb                           # Maps to --monitoring.wandb.enabled false")
        print()
        print("Profiles:")
        print("  default   - Standard configuration")
        print("  dev       - Development (faster, less logging)")
        print("  prod      - Production (full training)")
        print("  test      - Testing (minimal resources)")
        print("  profile   - Profiling (detailed metrics)")
        print()
        print("Environment Variables:")
        print("  GRPO_MODEL_NAME=Qwen/Qwen2.5-7B")
        print("  GRPO_TRAINING_LEARNING_RATE=1e-4")
        print()
        print("For more information, see CONFIG_SYSTEM.md")
    
    def _set_nested_value(self, config_dict: Dict[str, Any], key_path: str, value: Any):
        """Set a value in a nested dictionary using dot notation."""
        keys = key_path.split('.')
        current = config_dict
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                raise ValueError(f"Cannot override '{key_path}': '{key}' is not a dictionary")
            current = current[key]
        
        # Set the final value
        final_key = keys[-1]
        current[final_key] = value


def create_legacy_arg_mapping() -> Dict[str, str]:
    """
    Create mapping from legacy CLI arguments to new configuration paths.
    Used for backwards compatibility.
    """
    return {
        # Model arguments
        "--model_name": "model.name",
        "--torch_dtype": "model.torch_dtype",
        "--trust_remote_code": "model.trust_remote_code",
        "--attn_implementation": "model.attn_implementation",
        
        # Training arguments
        "--num_train_epochs": "training.epochs",
        "--per_device_train_batch_size": "training.batch_size.per_device_train",
        "--gradient_accumulation_steps": "training.batch_size.gradient_accumulation_steps",
        "--learning_rate": "training.optimization.learning_rate",
        "--logging_steps": "training.scheduling.logging_steps",
        "--eval_steps": "training.scheduling.eval_steps",
        "--save_steps": "training.scheduling.save_steps",
        "--max_completion_length": "grpo.max_completion_length",
        "--generation_batch_size": "training.batch_size.generation_batch_size",
        "--dataloader_num_workers": "training.dataloader.num_workers",
        
        # Dataset arguments
        "--dataset_name": "dataset.name",
        "--dataset_subset": "dataset.subset",
        "--test_size": "dataset.split.test_size",
        "--split_seed": "dataset.split.seed",
        
        # Reward function arguments
        "--reward_funcs": "rewards.functions",
        "--cosine_min_value_wrong": "rewards.cosine.min_value_wrong",
        "--cosine_max_value_wrong": "rewards.cosine.max_value_wrong",
        "--cosine_min_value_correct": "rewards.cosine.min_value_correct",
        "--cosine_max_value_correct": "rewards.cosine.max_value_correct",
        "--cosine_max_len": "rewards.cosine.max_len",
        "--repetition_n_grams": "rewards.repetition.n_grams",
        "--repetition_max_penalty": "rewards.repetition.max_penalty",
        
        # System arguments
        "--seed": "system.seed",
        "--output_dir": "system.output_dir",
        
        # Monitoring arguments
        "--wandb_project": "monitoring.wandb.project",
        "--wandb_run_name": "monitoring.wandb.run_name",
        "--no_wandb": "monitoring.wandb.enabled",  # Special case: inverse mapping
        
        # Checkpoint preservation arguments
        "--preserve_checkpoints_every": "callbacks.checkpoint_preservation.every_n_steps",
        "--preserve_checkpoints_dir": "callbacks.checkpoint_preservation.directory",
    }


class LegacyArgumentHandler:
    """Handles backwards compatibility with legacy CLI arguments."""
    
    def __init__(self):
        self.arg_mapping = create_legacy_arg_mapping()
        self.override_handler = OverrideHandler()
    
    def convert_legacy_args(self, args: List[str]) -> List[str]:
        """
        Convert legacy CLI arguments to new format.
        
        Args:
            args: List of CLI arguments
            
        Returns:
            List of converted arguments
        """
        converted_args = []
        i = 0
        
        while i < len(args):
            arg = args[i]
            
            if arg in self.arg_mapping:
                # Get the new configuration path
                new_key = self.arg_mapping[arg]
                
                # Special handling for inverse mappings
                if arg == "--no_wandb":
                    converted_args.extend([f"--{new_key}", "false"])
                    i += 1
                elif i + 1 < len(args) and not args[i + 1].startswith('--'):
                    # Has a value
                    value = args[i + 1]
                    converted_args.extend([f"--{new_key}", value])
                    i += 2
                else:
                    # Boolean flag
                    converted_args.extend([f"--{new_key}", "true"])
                    i += 1
            else:
                # Keep as-is
                converted_args.append(arg)
                i += 1
        
        return converted_args