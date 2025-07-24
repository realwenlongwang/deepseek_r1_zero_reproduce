"""
Configuration validator for type checking and business logic validation.
"""

import os
import torch
from typing import Dict, Any, List, Optional, Union
from dataclasses import fields, is_dataclass
from .schemas import Config, get_config_field_types


class ValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class ConfigValidator:
    """
    Validates configuration for type safety and business logic constraints.
    """
    
    def __init__(self):
        self.field_types = get_config_field_types()
    
    def validate_config(self, config: Config) -> List[str]:
        """
        Validate a complete configuration object.
        
        Args:
            config: Configuration object to validate
            
        Returns:
            List of validation warnings (errors raise exceptions)
        """
        warnings = []
        
        # Type validation
        self._validate_types(config)
        
        # Business logic validation
        warnings.extend(self._validate_model_config(config.model))
        warnings.extend(self._validate_training_arguments_config(config.TrainingArguments))
        warnings.extend(self._validate_dataset_config(config.dataset))
        warnings.extend(self._validate_grpo_config(config.GRPOConfig))
        warnings.extend(self._validate_rewards_config(config.rewards))
        warnings.extend(self._validate_system_config(config.system))
        warnings.extend(self._validate_monitoring_config(config.monitoring))
        warnings.extend(self._validate_callbacks_config(config.callbacks))
        
        # Cross-section validation
        warnings.extend(self._validate_cross_dependencies(config))
        
        return warnings
    
    def _validate_types(self, config: Config):
        """Validate that all configuration fields have correct types."""
        self._validate_dataclass_types(config, "")
    
    def _validate_dataclass_types(self, obj: Any, prefix: str):
        """Recursively validate dataclass field types."""
        if not is_dataclass(obj):
            return
        
        for field in fields(obj):
            field_name = field.name
            field_value = getattr(obj, field_name)
            full_path = f"{prefix}.{field_name}" if prefix else field_name
            
            if is_dataclass(field_value):
                self._validate_dataclass_types(field_value, full_path)
            elif isinstance(field_value, list):
                self._validate_list_field(field_value, full_path)
            else:
                self._validate_scalar_field(field_value, field.type, full_path)
    
    def _validate_list_field(self, value: List[Any], field_path: str):
        """Validate list field values."""
        if not isinstance(value, list):
            raise ValidationError(f"Field '{field_path}' must be a list, got {type(value).__name__}")
        
        # Additional validation for specific list fields
        if field_path == "rewards.functions":
            valid_functions = {"accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty", "equation"}
            for func in value:
                if func not in valid_functions:
                    raise ValidationError(f"Invalid reward function '{func}' in {field_path}. "
                                        f"Valid options: {valid_functions}")
    
    def _validate_scalar_field(self, value: Any, expected_type: type, field_path: str):
        """Validate scalar field values."""
        # Skip None values for optional fields
        if value is None:
            return
        
        # Type checking
        if expected_type == bool and not isinstance(value, bool):
            raise ValidationError(f"Field '{field_path}' must be boolean, got {type(value).__name__}")
        elif expected_type == int and not isinstance(value, int):
            raise ValidationError(f"Field '{field_path}' must be integer, got {type(value).__name__}")
        elif expected_type == float and not isinstance(value, (int, float)):
            raise ValidationError(f"Field '{field_path}' must be float, got {type(value).__name__}")
        elif expected_type == str and not isinstance(value, str):
            raise ValidationError(f"Field '{field_path}' must be string, got {type(value).__name__}")
    
    def _validate_model_config(self, model_config) -> List[str]:
        """Validate model configuration."""
        warnings = []
        
        # Check torch_dtype
        valid_dtypes = {"float32", "float16", "bfloat16", "int8"}
        if model_config.torch_dtype not in valid_dtypes:
            warnings.append(f"Unusual torch_dtype '{model_config.torch_dtype}'. "
                          f"Common values: {valid_dtypes}")
        
        # Check attention implementation
        if model_config.attn_implementation == "flash_attention_2":
            try:
                import flash_attn
            except ImportError:
                warnings.append("flash_attention_2 specified but flash-attn not installed. "
                              "Install with: pip install flash-attn")
        
        # Check device placement
        valid_placements = {"auto", "single", "multi"}
        if model_config.device_placement not in valid_placements:
            raise ValidationError(f"Invalid device_placement '{model_config.device_placement}'. "
                                f"Valid options: {valid_placements}")
        
        return warnings
    
    def _validate_training_arguments_config(self, training_args_config) -> List[str]:
        """Validate training arguments configuration."""
        warnings = []
        
        # GRPO requires effective batch size divisible by num_generations (8)
        effective_batch_size = (training_args_config.per_device_train_batch_size * 
                              training_args_config.gradient_accumulation_steps)
        
        if effective_batch_size % 8 != 0:
            raise ValidationError(f"Effective batch size ({effective_batch_size}) must be divisible by 8 for GRPO. "
                                f"Current: per_device_train_batch_size={training_args_config.per_device_train_batch_size} * "
                                f"gradient_accumulation_steps={training_args_config.gradient_accumulation_steps}")
        
        # Check learning rate range
        lr = training_args_config.learning_rate
        if lr <= 0:
            raise ValidationError(f"Learning rate must be positive, got {lr}")
        if lr > 1e-2:
            warnings.append(f"Learning rate {lr} seems high. Typical range: 1e-5 to 1e-3")
        
        # Check evaluation strategy
        eval_strategy = training_args_config.eval_strategy
        valid_strategies = {"no", "steps", "epoch"}
        if eval_strategy not in valid_strategies:
            raise ValidationError(f"Invalid eval_strategy '{eval_strategy}'. "
                                f"Valid options: {valid_strategies}")
        
        # Check save strategy
        save_strategy = training_args_config.save_strategy
        valid_save_strategies = {"no", "steps", "epoch"}
        if save_strategy not in valid_save_strategies:
            raise ValidationError(f"Invalid save_strategy '{save_strategy}'. "
                                f"Valid options: {valid_save_strategies}")
        
        # Check dataloader workers
        num_workers = training_args_config.dataloader_num_workers
        if num_workers < 0:
            raise ValidationError(f"dataloader_num_workers must be non-negative, got {num_workers}")
        
        # Check max_steps and num_train_epochs
        if training_args_config.max_steps > 0 and training_args_config.num_train_epochs > 0:
            warnings.append("Both max_steps and num_train_epochs are set. max_steps will take precedence.")
        
        return warnings
    
    def _validate_dataset_config(self, dataset_config) -> List[str]:
        """Validate dataset configuration."""
        warnings = []
        
        # Check test split size (now integer count)
        test_size = dataset_config.split.test_size
        if not isinstance(test_size, int) or test_size <= 0:
            raise ValidationError(f"test_size must be a positive integer, got {test_size}")
        
        # Check max_length
        max_length = dataset_config.processing.max_length
        if max_length <= 0:
            raise ValidationError(f"max_length must be positive, got {max_length}")
        if max_length > 8192:
            warnings.append(f"max_length {max_length} is quite large. Consider memory usage.")
        
        return warnings
    
    def _validate_grpo_config(self, grpo_config) -> List[str]:
        """Validate GRPO configuration."""
        warnings = []
        
        # Check num_generations
        if grpo_config.num_generations <= 0:
            raise ValidationError(f"num_generations must be positive, got {grpo_config.num_generations}")
        
        # Check max_completion_length
        if grpo_config.max_completion_length <= 0:
            raise ValidationError(f"max_completion_length must be positive, got {grpo_config.max_completion_length}")
        
        # Check vLLM configuration
        if grpo_config.use_vllm:
            gpu_util = grpo_config.vllm_gpu_memory_utilization
            if not 0 < gpu_util <= 1:
                raise ValidationError(f"vllm_gpu_memory_utilization must be between 0 and 1, got {gpu_util}")
            
            valid_modes = {"colocate", "separate"}
            if grpo_config.vllm_mode not in valid_modes:
                raise ValidationError(f"Invalid vllm_mode '{grpo_config.vllm_mode}'. "
                                    f"Valid options: {valid_modes}")
        
        return warnings
    
    def _validate_rewards_config(self, rewards_config) -> List[str]:
        """Validate rewards configuration."""
        warnings = []
        
        # Check that at least one reward function is specified
        if not rewards_config.functions:
            raise ValidationError("At least one reward function must be specified")
        
        # Check cosine reward parameters
        cosine = rewards_config.cosine
        if cosine.min_value_wrong >= cosine.max_value_wrong:
            raise ValidationError("cosine.min_value_wrong must be less than cosine.max_value_wrong")
        if cosine.min_value_correct >= cosine.max_value_correct:
            raise ValidationError("cosine.min_value_correct must be less than cosine.max_value_correct")
        if cosine.max_len <= 0:
            raise ValidationError("cosine.max_len must be positive")
        
        # Check repetition parameters
        repetition = rewards_config.repetition
        if repetition.n_grams <= 0:
            raise ValidationError("repetition.n_grams must be positive")
        if repetition.max_penalty >= 0:
            warnings.append("repetition.max_penalty is positive. Usually should be negative for penalty.")
        
        return warnings
    
    def _validate_system_config(self, system_config) -> List[str]:
        """Validate system configuration."""
        warnings = []
        
        # Check output directory
        if system_config.output_dir is not None:
            output_dir = system_config.output_dir
            parent_dir = os.path.dirname(output_dir)
            if parent_dir and not os.path.exists(parent_dir):
                warnings.append(f"Output directory parent '{parent_dir}' does not exist")
        
        return warnings
    
    def _validate_monitoring_config(self, monitoring_config) -> List[str]:
        """Validate monitoring configuration."""
        warnings = []
        
        # Check logging level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if monitoring_config.logging.level not in valid_levels:
            raise ValidationError(f"Invalid logging level '{monitoring_config.logging.level}'. "
                                f"Valid options: {valid_levels}")
        
        # Check wandb configuration
        if monitoring_config.wandb.enabled:
            try:
                import wandb
            except ImportError:
                warnings.append("wandb.enabled=true but wandb not installed. "
                              "Install with: pip install wandb")
        
        return warnings
    
    def _validate_callbacks_config(self, callbacks_config) -> List[str]:
        """Validate callbacks configuration."""
        warnings = []
        
        # Check checkpoint preservation
        checkpoint_config = callbacks_config.checkpoint_preservation
        if checkpoint_config.enabled and checkpoint_config.every_n_steps <= 0:
            raise ValidationError("checkpoint_preservation.every_n_steps must be positive when enabled")
        
        # Check reward trend window size
        if callbacks_config.reward_trend.window_size <= 0:
            raise ValidationError("reward_trend.window_size must be positive")
        
        return warnings
    
    def _validate_cross_dependencies(self, config: Config) -> List[str]:
        """Validate cross-section dependencies."""
        warnings = []
        
        # Check GPU availability for GPU-specific settings
        if not torch.cuda.is_available():
            if config.TrainingArguments.bf16:
                warnings.append("bf16=true but CUDA not available. bf16 requires GPU.")
            if config.TrainingArguments.tf32:
                warnings.append("tf32=true but CUDA not available. tf32 requires GPU.")
            if config.GRPOConfig.use_vllm:
                warnings.append("use_vllm=true but CUDA not available. vLLM requires GPU.")
        
        # Check memory constraints
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
            
            # Rough memory estimation for different model sizes
            model_name = config.model.name.lower()
            if "7b" in model_name and available_memory < 20:
                warnings.append(f"Model '{config.model.name}' may require more GPU memory than available "
                              f"({available_memory:.1f}GB). Consider using a smaller model or gradient checkpointing.")
            elif "3b" in model_name and available_memory < 10:
                warnings.append(f"Model '{config.model.name}' may require more GPU memory than available "
                              f"({available_memory:.1f}GB).")
        
        # Check reward functions compatibility
        if "cosine" in config.rewards.functions:
            if config.GRPOConfig.max_completion_length > config.rewards.cosine.max_len:
                warnings.append("GRPOConfig.max_completion_length exceeds cosine.max_len. "
                              "Cosine reward may not work properly for long completions.")
        
        return warnings
    
    def validate_dict(self, config_dict: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration dictionary before converting to Config object.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            List of validation warnings
        """
        warnings = []
        
        # Convert to Config object for validation
        try:
            config = self._dict_to_config(config_dict)
            warnings.extend(self.validate_config(config))
        except Exception as e:
            raise ValidationError(f"Failed to validate configuration: {e}")
        
        return warnings
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> Config:
        """Convert configuration dictionary to Config object."""
        try:
            # This is a simplified conversion - in practice, you'd use a proper
            # dictionary-to-dataclass conversion library like dacite or similar
            from dataclasses import fields
            
            def convert_dict_to_dataclass(cls, data):
                if not isinstance(data, dict):
                    return data
                
                kwargs = {}
                for field in fields(cls):
                    if field.name in data:
                        field_value = data[field.name]
                        if is_dataclass(field.type):
                            kwargs[field.name] = convert_dict_to_dataclass(field.type, field_value)
                        else:
                            kwargs[field.name] = field_value
                
                return cls(**kwargs)
            
            return convert_dict_to_dataclass(Config, config_dict)
        
        except Exception as e:
            raise ValidationError(f"Failed to convert dictionary to Config: {e}")