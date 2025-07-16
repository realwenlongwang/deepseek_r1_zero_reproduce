#!/usr/bin/env python3
"""
DeepSeek R1 Zero Training Script with Centralized YAML Configuration
Enhanced with comprehensive configuration management and CLI overrides.
"""

import os
import sys
import logging
import warnings
warnings.filterwarnings("ignore")
# Suppress specific Qwen2 gradient checkpointing warning
warnings.filterwarnings("ignore", message=".*Caching is incompatible with gradient checkpointing.*")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    set_seed,
)

# Import TRL for GRPO training
from trl import (
    GRPOTrainer,
    GRPOConfig,
)

# Import new configuration system
from src.config import ConfigManager, Config, ValidationError
from src.data.dataset import create_dataset, create_train_test_datasets
from src.rewards.openr1_rewards import get_reward_funcs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def setup_model_and_tokenizer(config: Config):
    """Setup model and tokenizer from configuration."""
    model_config = config.model
    logger.info(f"Loading model: {model_config.name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template for GRPO training
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}"
    
    # Determine device mapping based on configuration
    device_map = None
    if torch.cuda.is_available():
        if config.model.device_placement == "single":
            device_map = "cuda:0"
            logger.info("Using single GPU (cuda:0) for model")
        elif config.model.device_placement == "multi":
            device_map = "auto"
            logger.info("Using multi-GPU (auto) for model")
        else:  # auto
            # Auto-detect based on model size
            if any(size in model_config.name for size in ["0.5B", "1B", "1.5B", "3B"]):
                device_map = "cuda:0"
                logger.info("Auto-detected: Using single GPU for small-medium model")
            else:
                device_map = "auto"
                logger.info("Auto-detected: Using multi-GPU for large model")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        revision=model_config.revision,
        torch_dtype=getattr(torch, model_config.torch_dtype) if model_config.torch_dtype else None,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        use_cache=False,
        device_map=device_map
    )
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def setup_datasets(config: Config):
    """Setup training and evaluation datasets from configuration."""
    dataset_config = config.dataset
    logger.info(f"Loading dataset: {dataset_config.name}")
    
    # Load datasets with automatic train/test split handling
    train_dataset, test_dataset = create_train_test_datasets(
        dataset_name=dataset_config.name,
        test_size=dataset_config.split.test_size,
        split_seed=dataset_config.split.seed
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Train/test split created with seed: {dataset_config.split.seed}")
    
    return train_dataset, test_dataset


def create_training_arguments(config: Config, output_dir: str):
    """Create TrainingArguments from configuration."""
    from transformers import TrainingArguments
    
    training_config = config.training
    
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size.per_device_train,
        per_device_eval_batch_size=training_config.batch_size.per_device_eval,
        gradient_accumulation_steps=training_config.batch_size.gradient_accumulation_steps,
        learning_rate=training_config.optimization.learning_rate,
        warmup_ratio=training_config.optimization.warmup_ratio,
        weight_decay=training_config.optimization.weight_decay,
        logging_steps=training_config.scheduling.logging_steps,
        eval_strategy=training_config.scheduling.eval_strategy,
        eval_steps=training_config.scheduling.eval_steps,
        save_strategy=training_config.scheduling.save_strategy,
        save_steps=training_config.scheduling.save_steps,
        save_total_limit=training_config.scheduling.save_total_limit,
        dataloader_num_workers=training_config.dataloader.num_workers,
        dataloader_pin_memory=training_config.dataloader.pin_memory,
        dataloader_persistent_workers=training_config.dataloader.persistent_workers,
        dataloader_prefetch_factor=training_config.dataloader.prefetch_factor,
        dataloader_drop_last=training_config.dataloader.drop_last,
        seed=config.system.seed,
        bf16=training_config.precision.bf16,
        tf32=training_config.precision.tf32,
        gradient_checkpointing=training_config.precision.gradient_checkpointing,
        push_to_hub=False,
        report_to="none",
        remove_unused_columns=False,
        group_by_length=training_config.dataloader.group_by_length,
    )


def create_grpo_config_from_config(config: Config, training_args):
    """Create GRPOConfig from configuration."""
    grpo_config_dict = training_args.to_dict()
    grpo_settings = config.grpo
    
    # Add GRPO-specific settings
    grpo_config_dict.update({
        "max_completion_length": grpo_settings.max_completion_length,
        "generation_batch_size": config.training.batch_size.generation_batch_size,
        "num_generations": grpo_settings.num_generations,
        "use_vllm": grpo_settings.vllm.enabled,
        "vllm_mode": grpo_settings.vllm.mode,
        "vllm_gpu_memory_utilization": grpo_settings.vllm.gpu_memory_utilization,
        "use_liger_loss": grpo_settings.liger_loss,
    })
    
    # Create GRPOConfig
    try:
        grpo_config = GRPOConfig(**grpo_config_dict, generation_kwargs={})
    except TypeError:
        # Fallback for older TRL versions
        grpo_config = GRPOConfig(**grpo_config_dict)
    
    return grpo_config


def get_reward_functions(config: Config):
    """Get reward functions from configuration."""
    rewards_config = config.rewards
    
    # Create script args object compatible with openr1_rewards
    from src.config.grpo_config import GRPOScriptArguments
    script_args = GRPOScriptArguments(
        reward_funcs=rewards_config.functions,
        cosine_min_value_wrong=rewards_config.cosine.min_value_wrong,
        cosine_max_value_wrong=rewards_config.cosine.max_value_wrong,
        cosine_min_value_correct=rewards_config.cosine.min_value_correct,
        cosine_max_value_correct=rewards_config.cosine.max_value_correct,
        cosine_max_len=rewards_config.cosine.max_len,
        repetition_n_grams=rewards_config.repetition.n_grams,
        repetition_max_penalty=rewards_config.repetition.max_penalty,
        code_language=rewards_config.code.language,
        max_completion_len=rewards_config.soft_punish.max_completion_len,
        soft_punish_cache=rewards_config.soft_punish.cache,
    )
    
    return get_reward_funcs(script_args)


def get_callbacks(config: Config, training_args, delayed_dir_callback=None):
    """Get callbacks from configuration."""
    from src.config.grpo_config import (
        ProductionLoggingCallback,
        ComprehensiveLoggingCallback,
        RewardTrendCallback,
        CheckpointPreservationCallback,
        GRPOScriptArguments
    )
    
    callbacks = []
    
    # Create dummy script args for callbacks (backward compatibility)
    script_args = GRPOScriptArguments(
        reward_funcs=config.rewards.functions
    )
    
    # Choose logging callback based on configuration
    if config.monitoring.logging.profiling_mode or config.callbacks.comprehensive_logging.enabled:
        callbacks.append(ComprehensiveLoggingCallback(
            script_args, 
            log_examples=config.callbacks.comprehensive_logging.log_examples
        ))
    else:
        callbacks.append(ProductionLoggingCallback(script_args))
    
    # Add reward trend callback
    callbacks.append(RewardTrendCallback(
        window_size=config.callbacks.reward_trend.window_size
    ))
    
    # Add delayed directory creation callback if provided
    if delayed_dir_callback is not None:
        callbacks.append(delayed_dir_callback)
    
    # Add checkpoint preservation callback if enabled
    if config.callbacks.checkpoint_preservation.enabled:
        callbacks.append(CheckpointPreservationCallback(
            preserve_every_n_steps=config.callbacks.checkpoint_preservation.every_n_steps,
            preserve_dir=config.callbacks.checkpoint_preservation.directory
        ))
    
    return callbacks


def create_grpo_trainer(config: Config, model, tokenizer, reward_functions, training_args, train_dataset, eval_dataset, callbacks):
    """Create GRPOTrainer from configuration."""
    logger.info("Creating GRPOTrainer with configuration...")
    
    # Create GRPOConfig
    grpo_config = create_grpo_config_from_config(config, training_args)
    
    # Create GRPOTrainer
    grpo_trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks
    )
    
    logger.info("✅ GRPOTrainer created successfully")
    logger.info(f"   Model: {model.config.model_type}")
    logger.info(f"   Reward functions: {len(reward_functions)}")
    logger.info(f"   Callbacks: {len(callbacks)}")
    logger.info(f"   Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"   Evaluation dataset size: {len(eval_dataset)}")
    
    return grpo_trainer


def generate_unique_output_dir(config: Config) -> str:
    """Generate a unique output directory name from configuration."""
    import datetime
    import re
    
    model_name = config.model.name
    reward_funcs = config.rewards.functions
    
    # Extract model size from model name
    model_size = "unknown"
    if "0.5B" in model_name:
        model_size = "qwen2.5-0.5b"
    elif "1B" in model_name:
        model_size = "qwen2.5-1b"
    elif "3B" in model_name:
        model_size = "qwen2.5-3b"
    elif "7B" in model_name:
        model_size = "qwen2.5-7b"
    elif "14B" in model_name:
        model_size = "qwen2.5-14b"
    else:
        # Extract from model name if possible
        model_size = model_name.split('/')[-1].lower()
        model_size = re.sub(r'[^a-z0-9.-]', '-', model_size)
    
    # Create reward function string
    if len(reward_funcs) >= 4:
        reward_str = "all-rewards"
    else:
        reward_str = "-".join(reward_funcs[:3])
    
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine into output directory
    output_dir = f"saved_models/{model_size}_{reward_str}_{timestamp}"
    
    return output_dir


def setup_monitoring(config: Config):
    """Setup monitoring and logging from configuration."""
    # Configure logging level
    logging_config = config.monitoring.logging
    logger.setLevel(getattr(logging, logging_config.level))
    
    # Setup wandb if enabled
    if config.monitoring.wandb.enabled:
        os.environ["WANDB_PROJECT"] = config.monitoring.wandb.project
        if config.monitoring.wandb.run_name:
            os.environ["WANDB_RUN_NAME"] = config.monitoring.wandb.run_name
        return "wandb"
    else:
        return "none"


def main():
    """Main training function with new configuration system."""
    try:
        # Parse special arguments before creating config manager
        config_file = "config.yaml"
        profile = "default"
        cli_args = sys.argv[1:]
        
        # Extract --config and --profile from CLI args
        filtered_args = []
        i = 0
        while i < len(cli_args):
            arg = cli_args[i]
            
            if arg in ["--config", "--profile"]:
                if i + 1 < len(cli_args):
                    value = cli_args[i + 1]
                    if arg == "--config":
                        config_file = value
                    elif arg == "--profile":
                        profile = value
                    i += 2
                else:
                    raise ValueError(f"Missing value for argument: {arg}")
            else:
                filtered_args.append(arg)
                i += 1
        
        # Create configuration manager
        config_manager = ConfigManager(
            config_file=config_file,
            profile=profile,
            enable_legacy_compatibility=True
        )
        
        # Load configuration with CLI overrides
        config = config_manager.load_config(cli_args=filtered_args)
        
        # Set seed
        set_seed(config.system.seed)
        
        # Setup monitoring
        report_to = setup_monitoring(config)
        
        # Generate output directory if not specified
        if config.system.output_dir is None:
            actual_output_dir = generate_unique_output_dir(config)
            logger.info(f"Generated unique output directory: {actual_output_dir}")
        else:
            actual_output_dir = config.system.output_dir
        
        # Create temporary directory for initial setup
        import tempfile
        temp_output_dir = tempfile.mkdtemp(prefix="grpo_temp_")
        logger.info(f"Using temporary directory for initial setup: {temp_output_dir}")
        
        # Create training arguments
        training_args = create_training_arguments(config, temp_output_dir)
        training_args.report_to = report_to
        
        # Print configuration summary
        config_manager.print_config(config)
        
        # Log GPU information
        if torch.cuda.is_available():
            logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("CUDA not available - using CPU")
        
        logger.info("="*80)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Setup datasets
        train_dataset, test_dataset = setup_datasets(config)
        eval_dataset = test_dataset
        
        # Get reward functions
        reward_functions = get_reward_functions(config)
        logger.info(f"Initialized {len(reward_functions)} reward functions")
        
        # Create delayed directory creation callback
        from src.config.grpo_config import DelayedDirectoryCreationCallback
        delayed_dir_callback = DelayedDirectoryCreationCallback(actual_output_dir)
        
        # Get callbacks
        callbacks = get_callbacks(config, training_args, delayed_dir_callback)
        logger.info(f"Initialized {len(callbacks)} callbacks")
        
        # Create GRPO trainer
        grpo_trainer = create_grpo_trainer(
            config=config,
            model=model,
            tokenizer=tokenizer,
            reward_functions=reward_functions,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks
        )
        
        # Start training
        logger.info("🚀 Starting GRPO training...")
        train_result = grpo_trainer.train()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 GRPO TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"💡 Training result: {train_result}")
        logger.info("🚀 DeepSeek R1 Zero model trained with GRPO")
        logger.info("="*80)
        
    except ValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        # Clean up temporary directory if it exists
        import shutil
        if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            logger.info(f"Cleaned up temporary directory: {temp_output_dir}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        # Clean up temporary directory if it exists
        import shutil
        if 'temp_output_dir' in locals() and os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            logger.info(f"Cleaned up temporary directory: {temp_output_dir}")
        sys.exit(1)
    finally:
        # Clean up temporary directory if it exists and wasn't moved
        import shutil
        if 'temp_output_dir' in locals() and 'actual_output_dir' in locals() and \
           os.path.exists(temp_output_dir) and temp_output_dir != actual_output_dir:
            shutil.rmtree(temp_output_dir)
            logger.info(f"Final cleanup of temporary directory: {temp_output_dir}")


if __name__ == "__main__":
    main()