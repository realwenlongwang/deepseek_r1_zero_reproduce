#!/usr/bin/env python3
"""
DeepSeek R1 Zero Profiling Script
Mirrors the exact training script but uses comprehensive profiling callbacks.
Runs for a short duration to profile performance without overhead concerns.
"""

import os
import sys
import logging
import warnings

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings("ignore")
# Suppress specific Qwen2 gradient checkpointing warning
warnings.filterwarnings("ignore", 
                       message=".*Caching is incompatible with gradient checkpointing.*")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)
from transformers.trainer_utils import set_seed

# Import TRL for GRPO training (following tutorial)
from trl import GRPOTrainer

# Import tutorial-based GRPO components
from src.config.grpo_config import (
    GRPOScriptArguments,
    ModelConfig,
    create_training_arguments,
    create_grpo_config,
    get_reward_functions,
    ComprehensiveLoggingCallback
)
from src.data.dataset import create_dataset
from src.rewards.tutorial_rewards import TutorialRewardSystem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('profiling.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main profiling function - mirrors training script exactly but with comprehensive callbacks."""
    
    # Parse arguments
    parser = HfArgumentParser([GRPOScriptArguments, ModelConfig])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        script_args, model_config = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        script_args, model_config = parser.parse_args_into_dataclasses()
    
    # Override for profiling: limit to short runs
    script_args.max_steps = min(script_args.max_steps, 50)
    script_args.logging_steps = min(script_args.logging_steps, 5)
    
    logger.info("="*80)
    logger.info("DEEPSEEK R1 ZERO PROFILING - PERFORMANCE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Model: {script_args.model_name}")
    logger.info(f"Reward functions: {script_args.reward_funcs}")
    logger.info(f"Max steps: {script_args.max_steps} (limited for profiling)")
    logger.info(f"Logging steps: {script_args.logging_steps}")
    logger.info(f"Batch size: {script_args.per_device_train_batch_size}")
    logger.info(f"GPU devices: {torch.cuda.device_count()}")
    logger.info("="*80)
    
    # Set random seed
    set_seed(script_args.seed)
    
    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        trust_remote_code=True
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_dataset(script_args, tokenizer)
    
    # Create reward system
    logger.info("Setting up reward system...")
    # reward_system = TutorialRewardSystem(reward_funcs=script_args.reward_funcs)
    reward_functions = get_reward_functions(script_args)
    
    # Create training arguments
    logger.info("Creating training configuration...")
    training_args = create_training_arguments(script_args.output_dir)
    
    # Create GRPO config
    grpo_config = create_grpo_config(training_args, script_args, model_config)
    
    # Create COMPREHENSIVE callbacks for profiling (not production callbacks)
    comprehensive_callback = ComprehensiveLoggingCallback(
        script_args=script_args,
        log_examples=True
    )
    
    # Create trainer
    logger.info("Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_functions,
        train_dataset=dataset,
        callbacks=[comprehensive_callback]
    )
    
    # Run profiling training
    logger.info("Starting profiling training...")
    logger.info(f"WARNING: This will run {script_args.max_steps} steps with "
                f"HEAVY profiling overhead")
    logger.info("This is for performance analysis only - use train_grpo.py "
                "for actual training")
    
    try:
        trainer.train()
        logger.info("Profiling completed successfully!")
        
        # Log final profiling summary
        logger.info("="*80)
        logger.info("PROFILING SUMMARY")
        logger.info("="*80)
        logger.info("Use these metrics to understand actual training performance.")
        logger.info("The production training script (train_grpo.py) will run "
                    "faster without profiling overhead.")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Profiling failed: {e}")
        raise
    
    # Save profiling results
    if script_args.output_dir:
        logger.info(f"Profiling results saved to: {script_args.output_dir}")
        logger.info("Profiling log saved to: profiling.log")


if __name__ == "__main__":
    main()