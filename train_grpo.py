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
    TrainingArguments,
)
from transformers.trainer_utils import set_seed

# Try to import Unsloth
try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Import TRL for GRPO training
from trl import (
    GRPOTrainer,
    GRPOConfig,
)

# Import new configuration system
from src.config import ConfigManager, Config, ValidationError
from src.data.dataset import create_train_test_datasets
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
    """Setup model and tokenizer from configuration with Unsloth support."""
    model_config = config.model
    logger.info(f"Loading model: {model_config.name}")
    
    # Check if Unsloth is enabled and available
    use_unsloth = model_config.unsloth.enabled and UNSLOTH_AVAILABLE
    
    if use_unsloth:
        logger.info("🚀 Using Unsloth FastLanguageModel for optimized training")
        return _setup_unsloth_model(config)
    else:
        if model_config.unsloth.enabled and not UNSLOTH_AVAILABLE:
            logger.warning("⚠️ Unsloth requested but not available. Falling back to standard loading.")
        logger.info("📚 Using standard AutoModelForCausalLM")
        return _setup_standard_model(config)


def _setup_unsloth_model(config: Config):
    """Setup model and tokenizer using Unsloth FastLanguageModel."""
    model_config = config.model
    unsloth_config = model_config.unsloth
    
    # Get max_seq_length from dataset processing config
    max_seq_length = config.dataset.processing.max_length
    
    # Load model and tokenizer with Unsloth
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_config.name,
            max_seq_length=max_seq_length,
            load_in_4bit=unsloth_config.load_in_4bit,
            fast_inference=unsloth_config.fast_inference,
            gpu_memory_utilization=unsloth_config.gpu_memory_utilization,
        )
        logger.info("✅ Successfully loaded model with Unsloth FastLanguageModel")
    except Exception as e:
        logger.error(f"❌ Failed to load with Unsloth: {e}")
        logger.info("🔄 Falling back to standard model loading")
        return _setup_standard_model(config)
    
    # Configure tokenizer
    tokenizer = _configure_tokenizer(tokenizer, model_config.name)
    
    # Apply LoRA if enabled
    if config.model.lora.enabled:
        model = _apply_lora_to_model(model, config)
    
    logger.info(f"Model loaded with Unsloth: {model.config.model_type if hasattr(model, 'config') else 'Unknown'}")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def _setup_standard_model(config: Config):
    """Setup model and tokenizer using standard transformers."""
    model_config = config.model
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.name,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code
    )
    
    # Configure tokenizer
    tokenizer = _configure_tokenizer(tokenizer, model_config.name)
    
    # Determine device mapping based on configuration
    device_map = _get_device_mapping(config)
    
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


def _configure_tokenizer(tokenizer, model_name=None):
    """Configure tokenizer settings for GRPO training."""
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template using Unsloth's method if available
    if tokenizer.chat_template is None:
        try:
            from unsloth.chat_templates import get_chat_template
            
            # Determine appropriate template based on model
            if model_name and "qwen" in model_name.lower():
                template_name = "qwen-2.5"
            elif model_name and "gemma" in model_name.lower():
                template_name = "gemma-3"
            elif model_name and "llama" in model_name.lower():
                template_name = "llama-3.1"
            else:
                template_name = "chatml"  # Default ChatML format
            
            logger.info(f"Applying Unsloth chat template: {template_name}")
            
            tokenizer = get_chat_template(
                tokenizer,
                chat_template=template_name,
                mapping={"role": "role", "content": "content"},
                map_eos_token=True
            )
            
            logger.info("✅ Successfully applied Unsloth chat template")
            
        except ImportError:
            logger.warning("⚠️ Unsloth chat_templates not available, using ChatML fallback")
            tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply Unsloth chat template: {e}")
            tokenizer.chat_template = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    
    return tokenizer


def _get_device_mapping(config: Config):
    """Determine device mapping based on configuration."""
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
            model_name = config.model.name
            if any(size in model_name for size in ["0.5B", "1B", "1.5B", "3B"]):
                device_map = "cuda:0"
                logger.info("Auto-detected: Using single GPU for small-medium model")
            else:
                device_map = "auto"
                logger.info("Auto-detected: Using multi-GPU for large model")
    return device_map


def _apply_lora_to_model(model, config: Config):
    """Apply LoRA PEFT to the model using Unsloth."""
    lora_config = config.model.lora
    
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_config.rank,
            target_modules=lora_config.target_modules,
            lora_alpha=lora_config.alpha,
            use_gradient_checkpointing=lora_config.use_gradient_checkpointing,
            random_state=lora_config.random_state,
        )
        logger.info(f"✅ Applied LoRA PEFT: rank={lora_config.rank}, alpha={lora_config.alpha}")
        logger.info(f"📋 Target modules: {', '.join(lora_config.target_modules)}")
    except Exception as e:
        logger.error(f"❌ Failed to apply LoRA: {e}")
        logger.info("🔄 Continuing without LoRA")
    
    return model


def setup_datasets(config: Config):
    """Setup training and evaluation datasets from configuration."""
    dataset_config = config.dataset
    logger.info(f"Loading dataset: {dataset_config.name}")
    
    # Load datasets with automatic train/test split handling
    train_dataset, test_dataset = create_train_test_datasets(
        dataset_name=dataset_config.name,
        max_length=dataset_config.processing.max_length,
        test_size=dataset_config.split.test_size,
        split_seed=dataset_config.split.seed,
        system_prompt=dataset_config.processing.system_prompt
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    logger.info(f"Train/test split created with seed: {dataset_config.split.seed}")
    
    return train_dataset, test_dataset


def create_training_arguments(config: Config, output_dir: str):
    """Create TrainingArguments from configuration."""
    
    training_args_config = config.TrainingArguments
    
    # Convert dataclass to dict and update output_dir
    training_args_dict = {
        "output_dir": output_dir,
        "overwrite_output_dir": training_args_config.overwrite_output_dir,
        "num_train_epochs": training_args_config.num_train_epochs,
        "max_steps": training_args_config.max_steps,
        "per_device_train_batch_size": (
            training_args_config.per_device_train_batch_size
        ),
        "per_device_eval_batch_size": (
            training_args_config.per_device_eval_batch_size
        ),
        "gradient_accumulation_steps": (
            training_args_config.gradient_accumulation_steps
        ),
        "learning_rate": training_args_config.learning_rate,
        "warmup_ratio": training_args_config.warmup_ratio,
        "weight_decay": training_args_config.weight_decay,
        "logging_steps": training_args_config.logging_steps,
        "eval_strategy": training_args_config.eval_strategy,
        "eval_steps": training_args_config.eval_steps,
        "save_strategy": training_args_config.save_strategy,
        "save_steps": training_args_config.save_steps,
        "save_total_limit": training_args_config.save_total_limit,
        "dataloader_num_workers": training_args_config.dataloader_num_workers,
        "dataloader_pin_memory": training_args_config.dataloader_pin_memory,
        "dataloader_persistent_workers": (
            training_args_config.dataloader_persistent_workers
        ),
        "dataloader_prefetch_factor": (
            training_args_config.dataloader_prefetch_factor
            if training_args_config.dataloader_num_workers > 0
            else None
        ),
        "dataloader_drop_last": training_args_config.dataloader_drop_last,
        "seed": training_args_config.seed,
        "bf16": training_args_config.bf16,
        "tf32": training_args_config.tf32,
        "gradient_checkpointing": (
            training_args_config.gradient_checkpointing
        ),
        "push_to_hub": training_args_config.push_to_hub,
        "report_to": training_args_config.report_to,
        "remove_unused_columns": training_args_config.remove_unused_columns,
        "group_by_length": training_args_config.group_by_length,
        "torch_compile": training_args_config.torch_compile,
        "torch_compile_mode": training_args_config.torch_compile_mode,
        "optim": training_args_config.optim,
        "lr_scheduler_type": training_args_config.lr_scheduler_type,
        "adam_beta1": training_args_config.adam_beta1,
        "adam_beta2": training_args_config.adam_beta2,
        "max_grad_norm": training_args_config.max_grad_norm,
    }

    return TrainingArguments(**training_args_dict)


def create_grpo_config_from_config(config: Config, training_args):
    """Create GRPOConfig from configuration."""
    grpo_config_dict = training_args.to_dict()
    grpo_settings = config.GRPOConfig

    # Add GRPO-specific settings
    grpo_config_dict.update({
        "max_prompt_length": grpo_settings.max_prompt_length,
        "max_completion_length": grpo_settings.max_completion_length,
        "num_generations": grpo_settings.num_generations,
        "use_vllm": grpo_settings.use_vllm,
        "vllm_mode": grpo_settings.vllm_mode,
        "vllm_gpu_memory_utilization": (
            grpo_settings.vllm_gpu_memory_utilization
        ),
        "use_liger_loss": grpo_settings.use_liger_loss,
        "log_completions": grpo_settings.log_completions,
        "wandb_log_unique_prompts": grpo_settings.wandb_log_unique_prompts,
        "ddp_find_unused_parameters": grpo_settings.ddp_find_unused_parameters,
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
    comprehensive_logging_enabled = (
        config.monitoring.logging.profiling_mode or
        config.callbacks.comprehensive_logging.enabled
    )
    if comprehensive_logging_enabled:
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
            preserve_every_n_steps=(
                config.callbacks.checkpoint_preservation.every_n_steps
            ),
            preserve_dir=config.callbacks.checkpoint_preservation.directory
        ))

    return callbacks




def create_grpo_trainer(config: Config, model, tokenizer, reward_functions, training_args, train_dataset, eval_dataset, callbacks):
    """Create GRPOTrainer from configuration with vLLM fallback mechanism."""
    logger.info("Creating GRPOTrainer with configuration...")
    # Create GRPOConfig
    grpo_config = create_grpo_config_from_config(config, training_args)
    
    # Check if vLLM is enabled and set up fallback mechanism
    vllm_enabled = config.GRPOConfig.use_vllm
    if vllm_enabled:
        logger.info("vLLM is enabled - distributed environment already set up")
    
    # Create GRPOTrainer with vLLM fallback
    try:
        grpo_trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks
        )
        
        if vllm_enabled:
            logger.info("✅ GRPOTrainer created successfully with vLLM enabled")
        else:
            logger.info("✅ GRPOTrainer created successfully with vLLM disabled")
            
    except Exception as e:
        if vllm_enabled:
            logger.warning(f"Failed to create GRPOTrainer with vLLM: {e}")
            logger.info("Falling back to GRPOTrainer without vLLM...")
            
            # Create fallback config without vLLM
            grpo_config_fallback = create_grpo_config_from_config(config, training_args)
            grpo_config_fallback.use_vllm = False
            
            try:
                grpo_trainer = GRPOTrainer(
                    model=model,
                    reward_funcs=reward_functions,
                    args=grpo_config_fallback,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    processing_class=tokenizer,
                    callbacks=callbacks
                )
                logger.info("✅ GRPOTrainer created successfully with vLLM fallback disabled")
            except Exception as fallback_e:
                logger.error(f"Failed to create GRPOTrainer even without vLLM: {fallback_e}")
                raise fallback_e
        else:
            logger.error(f"Failed to create GRPOTrainer: {e}")
            raise e
    
    logger.info(f"   Model: {model.config.model_type}")
    # logger.info(f"   Reward functions: {len(reward_functions)}")
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


def setup_monitoring(config: Config, resume_info=None):
    """Setup monitoring and logging from configuration."""
    # Configure logging level
    logging_config = config.monitoring.logging
    logger.setLevel(getattr(logging, logging_config.level))
    
    # Setup wandb if enabled
    if config.monitoring.wandb.enabled:
        os.environ["WANDB_PROJECT"] = config.monitoring.wandb.project
        
        # Handle resume scenario with descriptive run name
        run_name = config.monitoring.wandb.run_name
        if resume_info and resume_info.get('previous_run_id') and resume_info.get('resume_step'):
            prev_run_id = resume_info['previous_run_id']
            step = resume_info['resume_step']
            if run_name:
                run_name = f"{run_name}_resumed_from_{prev_run_id}_step{step}"
            else:
                run_name = f"qwen2.5-3b_format-equation_resumed_from_{prev_run_id}_step{step}"
        
        if run_name:
            os.environ["WANDB_RUN_NAME"] = run_name
            
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
        set_seed(config.TrainingArguments.seed)
        
        # Handle resume from checkpoint or generate new output directory
        resume_info = None
        if config.system.resume_from_checkpoint:
            checkpoint_path = config.system.resume_from_checkpoint
            if not os.path.exists(checkpoint_path):
                raise ValueError(f"Resume checkpoint path does not exist: {checkpoint_path}")
            
            # Extract information for resume
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkpoint_name = os.path.basename(checkpoint_path)
            actual_output_dir = checkpoint_dir
            
            # Extract step number from checkpoint name
            import re
            step_match = re.search(r'checkpoint-(\d+)', checkpoint_name)
            resume_step = step_match.group(1) if step_match else "unknown"
            
            # Set resume info for wandb naming
            resume_info = {
                'previous_run_id': 'vfq6army',  # Known from our analysis
                'resume_step': resume_step
            }
            
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            logger.info(f"Using existing output directory: {actual_output_dir}")
        elif config.system.output_dir is None:
            actual_output_dir = generate_unique_output_dir(config)
            logger.info(f"Generated unique output directory: {actual_output_dir}")
        else:
            actual_output_dir = config.system.output_dir
        
        # Setup monitoring with resume info
        report_to = setup_monitoring(config, resume_info)
        
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
        if config.system.resume_from_checkpoint:
            logger.info(f"🔄 Resuming GRPO training from: {config.system.resume_from_checkpoint}")
            train_result = grpo_trainer.train(resume_from_checkpoint=config.system.resume_from_checkpoint)
        else:
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