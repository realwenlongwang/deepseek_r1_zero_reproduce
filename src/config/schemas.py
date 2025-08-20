"""
Configuration schemas for DeepSeek R1 Zero GRPO training.
Defines the complete hierarchical configuration structure using dataclasses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
import os


@dataclass
class ProjectConfig:
    """Project metadata configuration."""
    name: str = "deepseek-r1-zero-grpo"
    version: str = "1.0.0"
    description: str = "DeepSeek R1 Zero reproduction with GRPO training"


@dataclass
class UnslothConfig:
    """Unsloth FastLanguageModel configuration."""
    enabled: bool = False
    load_in_4bit: bool = True
    fast_inference: bool = True
    gpu_memory_utilization: float = 0.5


@dataclass
class LoRAConfig:
    """LoRA PEFT configuration."""
    enabled: bool = False
    rank: int = 16
    alpha: int = 16
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    device_placement: str = "auto"  # auto, single, multi
    unsloth: UnslothConfig = field(default_factory=UnslothConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass
class BatchSizeConfig:
    """Batch size configuration."""
    per_device_train: int = 16
    per_device_eval: int = 32
    gradient_accumulation_steps: int = 1
    generation_batch_size: int = 32


@dataclass
class OptimizationConfig:
    """Optimization configuration."""
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01


@dataclass
class PrecisionConfig:
    """Mixed precision configuration."""
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    torch_compile: bool = False


@dataclass
class SchedulingConfig:
    """Training scheduling configuration."""
    logging_steps: int = 10
    eval_strategy: str = "no"
    eval_steps: int = 50
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2


@dataclass
class DataloaderConfig:
    """Dataloader configuration."""
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    drop_last: bool = True
    group_by_length: bool = True


@dataclass
class TrainingArgumentsConfig:
    """Training Arguments configuration - exact parameter names from transformers.TrainingArguments."""
    output_dir: Optional[str] = None
    overwrite_output_dir: bool = False
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 2
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = False
    dataloader_persistent_workers: bool = False
    dataloader_prefetch_factor: int = 2
    dataloader_drop_last: bool = True
    seed: int = 42
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    push_to_hub: bool = False
    report_to: str = "none"
    remove_unused_columns: bool = False
    group_by_length: bool = True
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    
    # Optimizer configuration
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    lr_scheduler_type: str = "linear"


@dataclass
class DatasetSplitConfig:
    """Dataset split configuration."""
    test_size: int = 128
    seed: int = 42


@dataclass
class DatasetProcessingConfig:
    """Dataset processing configuration."""
    max_length: int = 2048
    system_prompt: str = "You are a helpful assistant that provides step-by-step reasoning."


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "Jiayi-Pan/Countdown-Tasks-3to4"
    subset: Optional[str] = None
    split: DatasetSplitConfig = field(default_factory=DatasetSplitConfig)
    processing: DatasetProcessingConfig = field(default_factory=DatasetProcessingConfig)


@dataclass
class VLLMConfig:
    """vLLM configuration."""
    enabled: bool = True
    mode: str = "colocate"
    gpu_memory_utilization: float = 0.3


@dataclass
class GRPOConfigConfig:
    """GRPO configuration - additional parameters beyond TrainingArguments."""
    max_prompt_length: int = 256
    max_completion_length: int = 1024
    num_generations: int = 8
    use_vllm: bool = False
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.3
    use_liger_loss: bool = False
    log_completions: bool = True
    wandb_log_unique_prompts: bool = True
    ddp_find_unused_parameters: bool = False


@dataclass
class CosineRewardConfig:
    """Cosine reward function configuration."""
    min_value_wrong: float = -0.5
    max_value_wrong: float = -0.1
    min_value_correct: float = 0.8
    max_value_correct: float = 1.0
    max_len: int = 1000


@dataclass
class RepetitionRewardConfig:
    """Repetition penalty reward configuration."""
    n_grams: int = 3
    max_penalty: float = -0.1


@dataclass
class CodeRewardConfig:
    """Code reward configuration."""
    language: str = "python"


@dataclass
class SoftPunishConfig:
    """Soft punishment configuration."""
    max_completion_len: int = 512
    cache: int = 50


@dataclass
class RewardsConfig:
    """Reward functions configuration."""
    functions: List[str] = field(default_factory=lambda: ["format", "equation"])
    cosine: CosineRewardConfig = field(default_factory=CosineRewardConfig)
    repetition: RepetitionRewardConfig = field(default_factory=RepetitionRewardConfig)
    code: CodeRewardConfig = field(default_factory=CodeRewardConfig)
    soft_punish: SoftPunishConfig = field(default_factory=SoftPunishConfig)


@dataclass
class SystemConfig:
    """System configuration."""
    output_dir: Optional[str] = None  # auto-generated if None
    resume_from_checkpoint: Optional[str] = None  # path to checkpoint directory to resume from


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "deepseek-r1-zero-grpo"
    run_name: Optional[str] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file: str = "training.log"
    profiling_mode: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


@dataclass
class ComprehensiveLoggingCallbackConfig:
    """Comprehensive logging callback configuration."""
    enabled: bool = False
    log_examples: bool = True


@dataclass
class RewardTrendCallbackConfig:
    """Reward trend callback configuration."""
    window_size: int = 50


@dataclass
class CheckpointPreservationCallbackConfig:
    """Checkpoint preservation callback configuration."""
    enabled: bool = True
    every_n_steps: int = 2000
    directory: str = "permanent_checkpoints"


@dataclass
class CallbacksConfig:
    """Callbacks configuration."""
    comprehensive_logging: ComprehensiveLoggingCallbackConfig = field(
        default_factory=ComprehensiveLoggingCallbackConfig
    )
    reward_trend: RewardTrendCallbackConfig = field(
        default_factory=RewardTrendCallbackConfig
    )
    checkpoint_preservation: CheckpointPreservationCallbackConfig = field(
        default_factory=CheckpointPreservationCallbackConfig
    )


@dataclass
class Config:
    """Complete configuration schema."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    TrainingArguments: TrainingArgumentsConfig = field(default_factory=TrainingArgumentsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    GRPOConfig: GRPOConfigConfig = field(default_factory=GRPOConfigConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)


def get_config_field_types() -> Dict[str, str]:
    """
    Get mapping of configuration field paths to their expected types.
    Used for type validation and CLI argument parsing.
    """
    return {
        # Project
        "project.name": "str",
        "project.version": "str",
        "project.description": "str",
        
        # Model
        "model.name": "str",
        "model.revision": "str",
        "model.torch_dtype": "str",
        "model.trust_remote_code": "bool",
        "model.attn_implementation": "str",
        "model.device_placement": "str",
        "model.unsloth.enabled": "bool",
        "model.unsloth.load_in_4bit": "bool",
        "model.unsloth.fast_inference": "bool",
        "model.unsloth.gpu_memory_utilization": "float",
        "model.lora.enabled": "bool",
        "model.lora.rank": "int",
        "model.lora.alpha": "int",
        "model.lora.target_modules": "list",
        "model.lora.use_gradient_checkpointing": "str",
        "model.lora.random_state": "int",
        
        # TrainingArguments
        "TrainingArguments.output_dir": "str",
        "TrainingArguments.overwrite_output_dir": "bool",
        "TrainingArguments.num_train_epochs": "float",
        "TrainingArguments.max_steps": "int",
        "TrainingArguments.per_device_train_batch_size": "int",
        "TrainingArguments.per_device_eval_batch_size": "int",
        "TrainingArguments.gradient_accumulation_steps": "int",
        "TrainingArguments.learning_rate": "float",
        "TrainingArguments.warmup_ratio": "float",
        "TrainingArguments.weight_decay": "float",
        "TrainingArguments.logging_steps": "int",
        "TrainingArguments.eval_strategy": "str",
        "TrainingArguments.eval_steps": "int",
        "TrainingArguments.save_strategy": "str",
        "TrainingArguments.save_steps": "int",
        "TrainingArguments.save_total_limit": "int",
        "TrainingArguments.dataloader_num_workers": "int",
        "TrainingArguments.dataloader_pin_memory": "bool",
        "TrainingArguments.dataloader_persistent_workers": "bool",
        "TrainingArguments.dataloader_prefetch_factor": "int",
        "TrainingArguments.dataloader_drop_last": "bool",
        "TrainingArguments.seed": "int",
        "TrainingArguments.bf16": "bool",
        "TrainingArguments.tf32": "bool",
        "TrainingArguments.gradient_checkpointing": "bool",
        "TrainingArguments.push_to_hub": "bool",
        "TrainingArguments.report_to": "str",
        "TrainingArguments.remove_unused_columns": "bool",
        "TrainingArguments.group_by_length": "bool",
        "TrainingArguments.torch_compile": "bool",
        "TrainingArguments.torch_compile_mode": "str",
        "TrainingArguments.optim": "str",
        "TrainingArguments.adam_beta1": "float",
        "TrainingArguments.adam_beta2": "float",
        "TrainingArguments.max_grad_norm": "float",
        "TrainingArguments.lr_scheduler_type": "str",
        
        # Dataset
        "dataset.name": "str",
        "dataset.subset": "str",
        "dataset.split.test_size": "int",
        "dataset.split.seed": "int",
        "dataset.processing.max_length": "int",
        "dataset.processing.system_prompt": "str",
        
        # GRPOConfig
        "GRPOConfig.max_prompt_length": "int",
        "GRPOConfig.max_completion_length": "int",
        "GRPOConfig.num_generations": "int",
        "GRPOConfig.use_vllm": "bool",
        "GRPOConfig.vllm_mode": "str",
        "GRPOConfig.vllm_gpu_memory_utilization": "float",
        "GRPOConfig.use_liger_loss": "bool",
        "GRPOConfig.log_completions": "bool",
        "GRPOConfig.wandb_log_unique_prompts": "bool",
        "GRPOConfig.ddp_find_unused_parameters": "bool",
        
        # Rewards
        "rewards.functions": "list",
        "rewards.cosine.min_value_wrong": "float",
        "rewards.cosine.max_value_wrong": "float",
        "rewards.cosine.min_value_correct": "float",
        "rewards.cosine.max_value_correct": "float",
        "rewards.cosine.max_len": "int",
        "rewards.repetition.n_grams": "int",
        "rewards.repetition.max_penalty": "float",
        "rewards.code.language": "str",
        "rewards.soft_punish.max_completion_len": "int",
        "rewards.soft_punish.cache": "int",
        
        # System
        "system.output_dir": "str",
        "system.resume_from_checkpoint": "str",
        
        # Monitoring
        "monitoring.wandb.enabled": "bool",
        "monitoring.wandb.project": "str",
        "monitoring.wandb.run_name": "str",
        "monitoring.logging.level": "str",
        "monitoring.logging.file": "str",
        "monitoring.logging.profiling_mode": "bool",
        
        # Callbacks
        "callbacks.comprehensive_logging.enabled": "bool",
        "callbacks.comprehensive_logging.log_examples": "bool",
        "callbacks.reward_trend.window_size": "int",
        "callbacks.checkpoint_preservation.enabled": "bool",
        "callbacks.checkpoint_preservation.every_n_steps": "int",
        "callbacks.checkpoint_preservation.directory": "str",
    }


def get_array_fields() -> List[str]:
    """Get list of configuration fields that expect array values."""
    return [
        "rewards.functions",
        "model.lora.target_modules",
    ]


def get_boolean_fields() -> List[str]:
    """Get list of configuration fields that expect boolean values."""
    field_types = get_config_field_types()
    return [field for field, field_type in field_types.items() if field_type == "bool"]


def get_numeric_fields() -> List[str]:
    """Get list of configuration fields that expect numeric values."""
    field_types = get_config_field_types()
    return [field for field, field_type in field_types.items() if field_type in ["int", "float"]]