"""
GRPO configuration classes based on DeepSeek R1 tutorial.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl


@dataclass
class GRPOScriptArguments:
    """
    Script arguments for GRPO training, specifically related to reward functions.
    """

    reward_funcs: List[str] = field(
        default_factory=lambda: ["equation", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty'"
        },
    )
    cosine_min_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Minimum reward for cosine scaling for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.1,
        metadata={"help": "Maximum reward for cosine scaling for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.8,
        metadata={"help": "Minimum reward for cosine scaling for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for cosine scaling for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for cosine scaling"},
    )

    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-0.1,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    
    # Additional parameters for OpenR1 rewards
    code_language: str = field(
        default="python",
        metadata={"help": "Programming language for code format reward"},
    )
    max_completion_len: int = field(
        default=512,
        metadata={"help": "Maximum completion length for soft overlong punishment"},
    )
    soft_punish_cache: int = field(
        default=50,
        metadata={"help": "Soft punishment cache for overlong completions"},
    )


@dataclass
class ModelConfig:
    """
    Configuration for the model following tutorial specifications.
    """
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-7B-Instruct", 
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_revision: Optional[str] = field(
        default="main", 
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    torch_dtype: Optional[str] = field(
        default="bfloat16", 
        metadata={"help": "Override the default `torch_dtype` and load the model under this dtype."}
    )
    trust_remote_code: bool = field(
        default=True, 
        metadata={"help": "Trust remote code when loading model and tokenizer."}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2", 
        metadata={"help": "Attention implementation to use. 'flash_attention_2' or None"}
    )


def create_training_arguments(output_dir: str = "./grpo_output") -> TrainingArguments:
    """
    Create TrainingArguments optimized for maximum GPU utilization and performance.
    Enhanced from tutorial configuration for L40S GPU with Ada architecture.
    """
    return TrainingArguments(
        output_dir=output_dir,              # Output directory for checkpoints and logs
        overwrite_output_dir=False,         # Preserve existing checkpoints (with unique names)
        num_train_epochs=1,                 # Total number of training epochs
        per_device_train_batch_size=16,     # INCREASED: Better GPU utilization (16*1=16, divisible by 8)
        per_device_eval_batch_size=32,      # INCREASED: Larger eval batch size
        gradient_accumulation_steps=1,      # REDUCED: Less accumulation with larger batch
        learning_rate=5e-5,                 # Initial learning rate for AdamW optimizer
        warmup_ratio=0.1,                   # Linear warmup over warmup_ratio fraction of training steps
        weight_decay=0.01,                  # Apply weight decay to all layers except bias and LayerNorm weights
        logging_steps=10,                   # Log every X updates steps
        eval_strategy="no",                 # Disable evaluation for now
        eval_steps=50,                      # Evaluation and logging steps
        save_strategy="steps",              # Save checkpoint every `save_steps`
        save_steps=50,                      # Save checkpoint every X updates steps
        save_total_limit=2,                 # Limit the total amount of checkpoints
        dataloader_num_workers=4,           # OPTIMAL: Match CPU core count for best performance
        dataloader_pin_memory=True,         # ENABLED: Faster CPU->GPU transfer
        dataloader_persistent_workers=True, # ENABLED: Keep workers alive between epochs
        dataloader_prefetch_factor=4,       # ADDED: Prefetch more batches for smoother training
        dataloader_drop_last=True,          # ENABLED: Consistent batch sizes for stable timing
        seed=42,                            # Random seed for reproducibility
        bf16=True,                          # Use mixed precision BFP16 training
        tf32=True,                          # ENABLED: Use TF32 for Ada architecture speedup
        push_to_hub=False,                  # Whether to push the final model to Hugging Face Hub
        gradient_checkpointing=False,       # Disabled for speed (we have abundant GPU memory)
        report_to="none",                   # Reporting to no one
        remove_unused_columns=False,        # Do not remove unused columns from the dataset
        group_by_length=True              # ENABLED: Group similar lengths for less padding
    )


def create_grpo_config(training_args, max_completion_length=512, generation_batch_size=32) -> 'GRPOConfig':
    """
    Create GRPOConfig with optimized parameters for better performance.
    """
    from trl import GRPOConfig
    
    # Create base config dict from training args
    config_dict = training_args.to_dict()
    
    # Add GRPO-specific optimizations for better GPU utilization
    config_dict.update({
        "max_completion_length": max_completion_length,    # Configurable completion length
        "generation_batch_size": generation_batch_size,    # Configurable generation batch size
        "num_generations": 8,                              # Standard GRPO setting
        "use_vllm": True,                                  # Enable vLLM with colocated mode
        "vllm_mode": "colocate",                           # Use colocated mode (same GPU)
        "vllm_gpu_memory_utilization": 0.3,                # Use 30% GPU memory for vLLM
        "use_liger_loss": False,
    })
    
    # Only add generation_kwargs if it's supported in this TRL version
    try:
        grpo_config = GRPOConfig(**config_dict, generation_kwargs={})
    except TypeError:
        # Fallback for older TRL versions that don't support generation_kwargs
        grpo_config = GRPOConfig(**config_dict)
    
    return grpo_config


def get_reward_functions(script_args: GRPOScriptArguments):
    """
    Returns a list of TRL-compatible reward functions based on the script arguments.
    These functions work with TRL's GRPOTrainer interface.
    """
    from ..rewards.openr1_rewards import get_reward_funcs
    
    # Create reward functions using OpenR1 registry pattern
    reward_functions = get_reward_funcs(script_args)
    
    return reward_functions


import time
import psutil
import torch
import numpy as np
from collections import defaultdict
import re

logger = logging.getLogger(__name__)


class ProductionLoggingCallback(TrainerCallback):
    """
    Lightweight production callback with minimal overhead.
    Focuses on essential metrics without heavy timing measurements.
    """
    
    def __init__(self, script_args: GRPOScriptArguments):
        self.script_args = script_args
        self.reward_history = defaultdict(list)
        
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log essential metrics at each step."""
        if state.global_step % args.logging_steps == 0:
            self._log_basic_metrics(args, state, **kwargs)
            self._log_reward_metrics(**kwargs)
            self._log_memory_usage()
            
    def _log_basic_metrics(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Log basic training metrics."""
        if not state.log_history:
            return
            
        latest_log = state.log_history[-1]
        
        # Basic metrics
        loss = latest_log.get('loss', None)
        lr = latest_log.get('learning_rate', None)
        
        # GRPO-specific losses
        policy_loss = latest_log.get('policy_loss', None)
        value_loss = latest_log.get('value_loss', None)
        entropy_loss = latest_log.get('entropy_loss', None)
        
        logger.info(f"Step {state.global_step:4d} | "
                   f"Loss: {loss:.4f} | "
                   f"LR: {lr:.2e} | "
                   f"Epoch: {state.epoch:.2f}")
        
        if policy_loss is not None:
            logger.info(f"  Policy Loss: {policy_loss:.4f} | "
                       f"Value Loss: {value_loss:.4f} | "
                       f"Entropy Loss: {entropy_loss:.4f}")
    
    def _log_reward_metrics(self, **kwargs):
        """Log reward function metrics."""
        try:
            rewards = kwargs.get('rewards', None)
            if rewards is None:
                return
                
            if hasattr(rewards, 'shape') and len(rewards.shape) > 1:
                # Multi-dimensional reward tensor
                mean_reward = float(rewards.mean())
                std_reward = float(rewards.std())
                
                logger.info(f"Rewards: mean={mean_reward:.4f}, std={std_reward:.4f}")
                
                # Track history for basic statistics
                self.reward_history['mean'].append(mean_reward)
                self.reward_history['std'].append(std_reward)
                
        except Exception as e:
            logger.warning(f"Error logging reward metrics: {e}")
    
    def _log_memory_usage(self):
        """Log memory usage without performance impact."""
        try:
            if torch.cuda.is_available():
                # Peak memory usage (no synchronization required)
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"GPU Memory: {gpu_memory:.2f}GB")
                
        except Exception as e:
            logger.warning(f"Error logging memory usage: {e}")


class ComprehensiveLoggingCallback(TrainerCallback):
    """
    Comprehensive callback for profiling detailed GRPO training metrics.
    Uses accurate GPU timing and detailed measurements. HIGH OVERHEAD - for profiling only.
    """
    
    def __init__(self, script_args: GRPOScriptArguments, log_examples: bool = True):
        self.script_args = script_args
        self.log_examples = log_examples
        self.step_start_time = None
        self.total_tokens_processed = 0
        self.reward_history = defaultdict(list)
        self.generation_stats = defaultdict(list)
        
        # CUDA events for accurate GPU timing
        if torch.cuda.is_available():
            self.cuda_start = torch.cuda.Event(enable_timing=True)
            self.cuda_end = torch.cuda.Event(enable_timing=True)
        else:
            self.cuda_start = None
            self.cuda_end = None
        
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Track step timing with accurate GPU measurement."""
        if self.cuda_start is not None:
            # Record CUDA event for accurate GPU timing
            self.cuda_start.record()
        self.step_start_time = time.time()
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Log comprehensive metrics at each step."""
        if state.global_step % args.logging_steps == 0:
            self._log_basic_metrics(args, state, **kwargs)
            self._log_reward_metrics(**kwargs)
            self._log_generation_quality(**kwargs)
            self._log_performance_metrics(args, state, **kwargs)
            
            if self.log_examples and state.global_step % (args.logging_steps * 5) == 0:
                self._log_generation_examples(**kwargs)
    
    def _log_basic_metrics(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Log basic training metrics."""
        if not state.log_history:
            return
            
        latest_log = state.log_history[-1]
        
        # Basic metrics
        loss = latest_log.get('loss', None)
        lr = latest_log.get('learning_rate', None)
        
        # GRPO-specific losses
        policy_loss = latest_log.get('policy_loss', None)
        value_loss = latest_log.get('value_loss', None)
        entropy_loss = latest_log.get('entropy_loss', None)
        
        logger.info(f"Step {state.global_step:4d} | "
                   f"Loss: {loss:.4f} | "
                   f"LR: {lr:.2e} | "
                   f"Epoch: {state.epoch:.2f}")
        
        if policy_loss is not None:
            logger.info(f"         GRPO | "
                       f"Policy: {policy_loss:.4f} | "
                       f"Value: {value_loss:.4f} | "
                       f"Entropy: {entropy_loss:.4f}")
    
    def _log_reward_metrics(self, **kwargs):
        """Log detailed reward function breakdown."""
        # Extract reward information if available
        rewards = kwargs.get('rewards', None)
        if rewards is None:
            return
            
        try:
            # Calculate reward statistics
            reward_stats = {}
            for reward_name in self.script_args.reward_funcs:
                if reward_name in rewards:
                    values = rewards[reward_name]
                    if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                        reward_stats[reward_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                        self.reward_history[reward_name].extend(values)
            
            if reward_stats:
                logger.info("Reward Breakdown:")
                for name, stats in reward_stats.items():
                    logger.info(f"  {name:15s}: {stats['mean']:6.3f} ± {stats['std']:5.3f} "
                               f"[{stats['min']:6.3f}, {stats['max']:6.3f}]")
                
                # Log total reward
                if 'total' in rewards:
                    total_values = rewards['total']
                    if isinstance(total_values, (list, np.ndarray)) and len(total_values) > 0:
                        total_mean = np.mean(total_values)
                        logger.info(f"  {'total':15s}: {total_mean:6.3f} (combined)")
                        
        except Exception as e:
            logger.warning(f"Error logging reward metrics: {e}")
    
    def _log_generation_quality(self, **kwargs):
        """Log generation quality metrics."""
        try:
            completions = kwargs.get('completions', None)
            if completions is None:
                return
                
            # Analyze generation quality
            format_compliance = 0
            avg_length = 0
            avg_think_length = 0
            avg_answer_length = 0
            reasoning_indicators = 0
            
            for completion in completions:
                if isinstance(completion, list) and len(completion) > 0:
                    content = completion[0].get('content', '')
                elif isinstance(completion, dict):
                    content = completion.get('content', '')
                else:
                    content = str(completion)
                
                # Check format compliance
                think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
                
                if think_match and answer_match:
                    format_compliance += 1
                    think_content = think_match.group(1).strip()
                    answer_content = answer_match.group(1).strip()
                    avg_think_length += len(think_content)
                    avg_answer_length += len(answer_content)
                    
                    # Count reasoning indicators
                    reasoning_patterns = [
                        r'step \d+', r'first[,\s]', r'second[,\s]', r'next[,\s]',
                        r'\d+\.', r'therefore', r'because', r'since'
                    ]
                    for pattern in reasoning_patterns:
                        reasoning_indicators += len(re.findall(pattern, think_content, re.IGNORECASE))
                
                avg_length += len(content)
            
            n_completions = len(completions)
            if n_completions > 0:
                format_rate = format_compliance / n_completions * 100
                avg_length /= n_completions
                avg_think_length = avg_think_length / max(format_compliance, 1)
                avg_answer_length = avg_answer_length / max(format_compliance, 1)
                reasoning_per_response = reasoning_indicators / max(format_compliance, 1)
                
                logger.info(f"Generation Quality:")
                logger.info(f"  Format compliance: {format_rate:5.1f}% "
                           f"({format_compliance}/{n_completions})")
                logger.info(f"  Avg response length: {avg_length:6.1f} chars")
                logger.info(f"  Avg think length:    {avg_think_length:6.1f} chars")
                logger.info(f"  Avg answer length:   {avg_answer_length:6.1f} chars")
                logger.info(f"  Reasoning indicators: {reasoning_per_response:5.2f} per response")
                
                # Store stats for trend analysis
                self.generation_stats['format_rate'].append(format_rate)
                self.generation_stats['avg_length'].append(avg_length)
                self.generation_stats['reasoning_indicators'].append(reasoning_per_response)
                
        except Exception as e:
            logger.warning(f"Error logging generation quality: {e}")
    
    def _log_performance_metrics(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Log training performance metrics with accurate GPU timing."""
        try:
            # Calculate accurate GPU timing
            if self.cuda_start is not None and self.cuda_end is not None:
                # Record end event and synchronize for accurate timing
                self.cuda_end.record()
                torch.cuda.synchronize()  # Wait for GPU to complete
                gpu_time = self.cuda_start.elapsed_time(self.cuda_end) / 1000.0  # Convert to seconds
            else:
                gpu_time = 0
                
            # CPU wall-clock time (for comparison)
            cpu_wall_time = time.time() - self.step_start_time if self.step_start_time else 0
            
            # Calculate effective batch size for accurate throughput
            effective_batch_size = (args.per_device_train_batch_size * 
                                  max(1, torch.cuda.device_count()) * 
                                  args.gradient_accumulation_steps)
            
            # Memory usage across all GPUs
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                gpu_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
                
                # Multi-GPU memory usage
                if torch.cuda.device_count() > 1:
                    total_gpu_memory = 0
                    for i in range(torch.cuda.device_count()):
                        total_gpu_memory += torch.cuda.max_memory_allocated(i) / 1024**3
                    gpu_memory = total_gpu_memory
            else:
                gpu_memory = gpu_reserved = 0
            
            # CPU and system metrics
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            ram_usage = memory_info.used / 1024**3  # GB
            
            # Training speed estimates (using GPU time if available)
            actual_time = gpu_time if gpu_time > 0 else cpu_wall_time
            actual_time_per_step = actual_time / args.logging_steps  # Time per individual step
            samples_per_sec = effective_batch_size / max(actual_time_per_step, 0.001)
            
            logger.info(f"Performance (over {args.logging_steps} steps):")
            if gpu_time > 0:
                logger.info(f"  GPU time:      {gpu_time:6.2f}s")
                logger.info(f"  CPU wall time: {cpu_wall_time:6.2f}s")
            else:
                logger.info(f"  Step time:     {cpu_wall_time:6.2f}s")
            logger.info(f"  Samples/sec:   {samples_per_sec:6.1f}")
            logger.info(f"  Effective batch size: {effective_batch_size}")
            logger.info(f"  GPU memory:    {gpu_memory:6.2f}GB (reserved: {gpu_reserved:6.2f}GB)")
            logger.info(f"  RAM usage:     {ram_usage:6.2f}GB ({memory_info.percent:5.1f}%)")
            logger.info(f"  CPU usage:     {cpu_percent:5.1f}%")
            
        except Exception as e:
            logger.warning(f"Error logging performance metrics: {e}")
    
    def _log_generation_examples(self, **kwargs):
        """Log examples of model generations for qualitative analysis."""
        try:
            completions = kwargs.get('completions', None)
            rewards = kwargs.get('rewards', None)
            
            if completions is None or rewards is None:
                return
                
            logger.info("="*80)
            logger.info("GENERATION EXAMPLES")
            logger.info("="*80)
            
            # Find best and worst examples based on total reward
            if 'total' in rewards:
                total_rewards = rewards['total']
                if len(total_rewards) > 0:
                    best_idx = np.argmax(total_rewards)
                    worst_idx = np.argmin(total_rewards)
                    
                    self._log_single_example(completions[best_idx], rewards, best_idx, "BEST")
                    logger.info("-" * 40)
                    self._log_single_example(completions[worst_idx], rewards, worst_idx, "WORST")
            
            logger.info("="*80)
            
        except Exception as e:
            logger.warning(f"Error logging generation examples: {e}")
    
    def _log_single_example(self, completion, rewards, idx, label):
        """Log a single generation example with reward breakdown."""
        if isinstance(completion, list) and len(completion) > 0:
            content = completion[0].get('content', '')
        elif isinstance(completion, dict):
            content = completion.get('content', '')
        else:
            content = str(completion)
        
        logger.info(f"{label} EXAMPLE (idx {idx}):")
        logger.info(f"Content: {content[:200]}{'...' if len(content) > 200 else ''}")
        
        # Log reward breakdown for this example
        logger.info("Reward breakdown:")
        for reward_name in self.script_args.reward_funcs:
            if reward_name in rewards:
                reward_value = rewards[reward_name][idx] if idx < len(rewards[reward_name]) else 0
                logger.info(f"  {reward_name}: {reward_value:.3f}")
        
        if 'total' in rewards and idx < len(rewards['total']):
            logger.info(f"  total: {rewards['total'][idx]:.3f}")


class RewardTrendCallback(TrainerCallback):
    """
    Callback for tracking reward trends and detecting training issues.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.reward_history = defaultdict(list)
        self.step_count = 0
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Track reward trends and detect issues."""
        self.step_count += 1
        
        rewards = kwargs.get('rewards', None)
        if rewards is None:
            return
            
        # Store reward history
        for reward_name, values in rewards.items():
            if isinstance(values, (list, np.ndarray)) and len(values) > 0:
                mean_reward = np.mean(values)
                self.reward_history[reward_name].append(mean_reward)
        
        # Check for trends every window_size steps
        if self.step_count % self.window_size == 0:
            self._analyze_trends(state.global_step)
    
    def _analyze_trends(self, step):
        """Analyze reward trends and log warnings if needed."""
        for reward_name, history in self.reward_history.items():
            if len(history) >= self.window_size:
                recent = history[-self.window_size:]
                older = history[-2*self.window_size:-self.window_size] if len(history) >= 2*self.window_size else history[:-self.window_size]
                
                if len(older) > 0:
                    recent_mean = np.mean(recent)
                    older_mean = np.mean(older)
                    change = (recent_mean - older_mean) / abs(older_mean + 1e-8) * 100
                    
                    if abs(change) > 10:  # Significant change threshold
                        direction = "↗️" if change > 0 else "↘️"
                        logger.info(f"Trend Alert Step {step}: {reward_name} {direction} {change:+.1f}% "
                                   f"({older_mean:.3f} → {recent_mean:.3f})")


class DelayedDirectoryCreationCallback(TrainerCallback):
    """
    Callback that delays the creation of the actual output directory until the first checkpoint save.
    This prevents empty folders from being created when training fails early.
    """
    
    def __init__(self, actual_output_dir: str):
        self.actual_output_dir = actual_output_dir
        self.directory_created = False
        
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Create the actual output directory on first save."""
        if not self.directory_created:
            import os
            import shutil
            
            # Create the actual output directory
            os.makedirs(self.actual_output_dir, exist_ok=True)
            logger.info(f"Created output directory: {self.actual_output_dir}")
            
            # Move any existing files from temp directory to actual directory
            if os.path.exists(args.output_dir) and args.output_dir != self.actual_output_dir:
                # Copy contents from temp to actual directory
                for item in os.listdir(args.output_dir):
                    src_path = os.path.join(args.output_dir, item)
                    dst_path = os.path.join(self.actual_output_dir, item)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path)
                    else:
                        shutil.copy2(src_path, dst_path)
                
                # Update the args to point to the actual directory
                args.output_dir = self.actual_output_dir
                logger.info(f"Moved checkpoint to: {self.actual_output_dir}")
            
            self.directory_created = True


def get_callbacks(training_args: TrainingArguments, model_args: ModelConfig, script_args: GRPOScriptArguments, delayed_dir_callback=None, profiling_mode=False):
    """
    Returns a list of callbacks for GRPO training monitoring.
    
    Args:
        profiling_mode: If True, uses ComprehensiveLoggingCallback with heavy profiling.
                       If False, uses ProductionLoggingCallback with minimal overhead.
    """
    if profiling_mode:
        # Heavy profiling with accurate GPU timing - for profiling script only
        callbacks = [
            ComprehensiveLoggingCallback(script_args, log_examples=True),
            RewardTrendCallback(window_size=50)
        ]
    else:
        # Lightweight production callbacks - for actual training
        callbacks = [
            ProductionLoggingCallback(script_args),
            RewardTrendCallback(window_size=50)
        ]
    
    # Add delayed directory creation callback if provided
    if delayed_dir_callback is not None:
        callbacks.append(delayed_dir_callback)
    
    return callbacks