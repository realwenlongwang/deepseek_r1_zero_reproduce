#!/usr/bin/env python3
"""
DeepSeek R1 Zero Training Script with Comprehensive GRPO Implementation
Following the exact tutorial specifications with comprehensive logging.
"""

import os
import sys
import argparse
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
    HfArgumentParser
)

# Import TRL for GRPO training (following tutorial)
from trl import (
    GRPOTrainer,
    GRPOConfig
)

# Import tutorial-based GRPO components
from src.config.grpo_config import (
    GRPOScriptArguments,
    ModelConfig,
    create_training_arguments,
    create_grpo_config,
    get_reward_functions,
    get_callbacks
)
from src.data.dataset import create_dataset
from src.rewards.tutorial_rewards import TutorialRewardSystem

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepSeek R1 Zero with comprehensive GRPO")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model name or path")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                       help="Model dtype")
    parser.add_argument("--trust_remote_code", action="store_true", default=True,
                       help="Trust remote code")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                       help="Attention implementation (flash_attention_2 for speed)")
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="AI-MO/NuminaMath-TIR",
                       help="Dataset name")
    parser.add_argument("--dataset_subset", type=str, default=None,
                       help="Dataset subset")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (auto-generated if not specified)")
    parser.add_argument("--num_train_epochs", type=float, default=1,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=16,
                       help="Train batch size per device (optimized for L40S)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps (reduced for larger batch)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=50,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=50,
                       help="Save frequency")
    parser.add_argument("--max_completion_length", type=int, default=512,
                       help="Maximum completion length (increased to fix truncation)")
    parser.add_argument("--generation_batch_size", type=int, default=32,
                       help="Batch size for generation phase")
    parser.add_argument("--dataloader_num_workers", type=int, default=8,
                       help="Number of dataloader workers (increased for performance)")
    
    # Reward function arguments
    parser.add_argument("--reward_funcs", nargs="+", 
                       default=["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"],
                       help="Reward functions to use")
    parser.add_argument("--cosine_min_value_wrong", type=float, default=-0.5,
                       help="Cosine scaling min value for wrong answers")
    parser.add_argument("--cosine_max_value_wrong", type=float, default=-0.1,
                       help="Cosine scaling max value for wrong answers")
    parser.add_argument("--cosine_min_value_correct", type=float, default=0.8,
                       help="Cosine scaling min value for correct answers")
    parser.add_argument("--cosine_max_value_correct", type=float, default=1.0,
                       help="Cosine scaling max value for correct answers")
    parser.add_argument("--cosine_max_len", type=int, default=1000,
                       help="Cosine scaling max length")
    parser.add_argument("--repetition_n_grams", type=int, default=3,
                       help="N-grams for repetition penalty")
    parser.add_argument("--repetition_max_penalty", type=float, default=-0.1,
                       help="Max repetition penalty")
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--wandb_project", type=str, default="deepseek-r1-zero-grpo",
                       help="Weights & Biases project")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases")
    
    return parser.parse_args()


def setup_model_and_tokenizer(model_args: ModelConfig):
    """Setup model and tokenizer."""
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set chat template for GRPO training
    if tokenizer.chat_template is None:
        # Simple chat template for GRPO
        tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}User: {{ message['content'] }}{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% endif %}{% if not loop.last %}\n{% endif %}{% endfor %}"
    
    # Load model with optimized device placement
    # For small models (0.5B, 3B): use single GPU to avoid communication overhead
    # For large models (7B+): use auto for memory distribution
    device_map = None
    if torch.cuda.is_available():
        # Check model size to determine optimal device placement
        if "0.5B" in model_args.model_name_or_path or "1B" in model_args.model_name_or_path or "1.5B" in model_args.model_name_or_path or "3B" in model_args.model_name_or_path:
            device_map = "cuda:0"  # Single GPU for models up to 3B (faster)
            logger.info("Using single GPU (cuda:0) for small-medium model - optimized for speed")
        else:
            device_map = "auto"  # Multi-GPU for large models (memory efficiency)
            logger.info("Using auto device mapping for large model - optimized for memory")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        device_map=device_map
    )
    
    logger.info(f"Model loaded: {model.config.model_type}")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def setup_datasets(dataset_name: str, dataset_subset: str = None):
    """Setup training datasets."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    # Load dataset using existing implementation
    train_dataset = create_dataset(
        dataset_name=dataset_name,
        split="train"
    )
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    
    return train_dataset


def create_grpo_trainer(model, tokenizer, reward_functions, training_args, train_dataset, eval_dataset, callbacks, max_completion_length=512, generation_batch_size=32):
    """
    Create a real GRPOTrainer with optimized configuration for better performance.
    Enhanced from tutorial specifications for L40S GPU.
    """
    logger.info("Creating optimized GRPOTrainer with performance enhancements...")
    
    # Create optimized GRPOConfig with performance enhancements
    grpo_config = create_grpo_config(training_args, max_completion_length, generation_batch_size)
    
    # Create GRPOTrainer (exactly as in tutorial)
    grpo_trainer = GRPOTrainer(
        model=model,                      # Our initialized model
        reward_funcs=reward_functions,    # List of reward functions from previous step
        args=grpo_config,                # GRPOConfig (created from TrainingArguments)
        train_dataset=train_dataset,     # Training dataset
        eval_dataset=eval_dataset,       # Evaluation dataset (optional)
        processing_class=tokenizer,      # Pass tokenizer as processing_class
        callbacks=callbacks              # List of callbacks
    )
    
    logger.info("✅ Real GRPOTrainer created successfully")
    logger.info(f"   Model: {model.config.model_type}")
    logger.info(f"   Reward functions: {len(reward_functions)}")
    logger.info(f"   Callbacks: {len(callbacks)}")
    logger.info(f"   Training dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"   Evaluation dataset size: {len(eval_dataset)}")
    
    return grpo_trainer


def generate_unique_output_dir(model_name: str, reward_funcs: list) -> str:
    """
    Generate a unique output directory name with descriptive information.
    
    Format: saved_models/{model_size}_{reward_funcs}_{timestamp}
    Example: saved_models/qwen2.5-0.5b_accuracy-format-reasoning_20250106_143022
    """
    import datetime
    import re
    
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
    
    # Create reward function string (limit to avoid very long names)
    if len(reward_funcs) >= 4:
        reward_str = "all-rewards"
    else:
        reward_str = "-".join(reward_funcs[:3])  # Take first 3 to avoid too long names
        
    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Combine into output directory
    output_dir = f"saved_models/{model_size}_{reward_str}_{timestamp}"
    
    return output_dir


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create configuration objects
    script_args = GRPOScriptArguments(
        reward_funcs=args.reward_funcs,
        cosine_min_value_wrong=args.cosine_min_value_wrong,
        cosine_max_value_wrong=args.cosine_max_value_wrong,
        cosine_min_value_correct=args.cosine_min_value_correct,
        cosine_max_value_correct=args.cosine_max_value_correct,
        cosine_max_len=args.cosine_max_len,
        repetition_n_grams=args.repetition_n_grams,
        repetition_max_penalty=args.repetition_max_penalty
    )
    
    model_args = ModelConfig(
        model_name_or_path=args.model_name,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation
    )
    
    # Generate unique output directory if not specified
    if args.output_dir is None:
        args.output_dir = generate_unique_output_dir(args.model_name, args.reward_funcs)
        logger.info(f"Generated unique output directory: {args.output_dir}")
    
    training_args = create_training_arguments(args.output_dir)
    
    # Override training arguments with command line args
    training_args.num_train_epochs = args.num_train_epochs
    training_args.per_device_train_batch_size = args.per_device_train_batch_size
    training_args.gradient_accumulation_steps = args.gradient_accumulation_steps
    training_args.learning_rate = args.learning_rate
    training_args.logging_steps = args.logging_steps
    training_args.eval_steps = args.eval_steps
    training_args.save_steps = args.save_steps
    training_args.dataloader_num_workers = args.dataloader_num_workers
    
    # Fix dataloader settings for single-threaded optimization
    if args.dataloader_num_workers == 0:
        training_args.dataloader_persistent_workers = False  # Not applicable for single-threaded
        training_args.dataloader_prefetch_factor = None     # Not applicable for single-threaded
        training_args.dataloader_pin_memory = False         # Less beneficial for single-threaded
        logger.info("Optimized dataloader settings for single-threaded (num_workers=0)")
    elif args.dataloader_num_workers > 0:
        training_args.dataloader_persistent_workers = True
        training_args.dataloader_prefetch_factor = 4
        training_args.dataloader_pin_memory = True
        logger.info(f"Optimized dataloader settings for multi-threaded ({args.dataloader_num_workers} workers)")
    
    if not args.no_wandb:
        training_args.report_to = "wandb"
        os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name
    
    # Print configuration
    logger.info("="*80)
    logger.info("DEEPSEEK R1 ZERO GRPO TRAINING CONFIGURATION")
    logger.info("="*80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Output Directory: {training_args.output_dir}")
    logger.info(f"Reward Functions: {script_args.reward_funcs}")
    logger.info(f"Learning Rate: {training_args.learning_rate}")
    logger.info(f"Batch Size: {training_args.per_device_train_batch_size}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Logging Steps: {training_args.logging_steps}")
    logger.info(f"Seed: {args.seed}")
    
    # GPU info
    if torch.cuda.is_available():
        logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.info("CUDA not available - using CPU")
    
    logger.info("="*80)
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(model_args)
        
        # Setup datasets
        train_dataset = setup_datasets(args.dataset_name, args.dataset_subset)
        
        # Create eval dataset (use a small subset of train for quick evaluation)
        eval_dataset = None  # Can be added later if needed
        
        # Get reward functions
        reward_functions = get_reward_functions(script_args)
        logger.info(f"Initialized {len(reward_functions)} reward functions")
        
        # Get comprehensive callbacks
        callbacks = get_callbacks(training_args, model_args, script_args)
        logger.info(f"Initialized {len(callbacks)} comprehensive callbacks")
        
        # Create real GRPO trainer with optimized configuration
        grpo_trainer = create_grpo_trainer(
            model=model,
            tokenizer=tokenizer,
            reward_functions=reward_functions,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            max_completion_length=args.max_completion_length,
            generation_batch_size=args.generation_batch_size
        )
        
        # Start the GRPO Training Loop (exactly as in tutorial)
        logger.info("🚀 Starting real GRPO training loop...")
        train_result = grpo_trainer.train()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 REAL GRPO TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"💡 Training result: {train_result}")
        logger.info("🚀 DeepSeek R1 Zero model trained with tutorial-exact GRPO")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        
    except Exception as e:
        logger.error(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()