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
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    torch_dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"
    device_placement: str = "auto"  # auto, single, multi


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
class TrainingConfig:
    """Training configuration."""
    epochs: float = 1.0
    batch_size: BatchSizeConfig = field(default_factory=BatchSizeConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)


@dataclass
class DatasetSplitConfig:
    """Dataset split configuration."""
    test_size: float = 0.1
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
class GRPOConfig:
    """GRPO-specific configuration."""
    num_generations: int = 8
    max_completion_length: int = 512
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    liger_loss: bool = False


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
    seed: int = 42
    output_dir: Optional[str] = None  # auto-generated if None


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
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
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
        
        # Training - Batch Size
        "training.batch_size.per_device_train": "int",
        "training.batch_size.per_device_eval": "int",
        "training.batch_size.gradient_accumulation_steps": "int",
        "training.batch_size.generation_batch_size": "int",
        
        # Training - Optimization
        "training.optimization.learning_rate": "float",
        "training.optimization.warmup_ratio": "float",
        "training.optimization.weight_decay": "float",
        
        # Training - Precision
        "training.precision.bf16": "bool",
        "training.precision.tf32": "bool",
        "training.precision.gradient_checkpointing": "bool",
        
        # Training - Scheduling
        "training.scheduling.logging_steps": "int",
        "training.scheduling.eval_strategy": "str",
        "training.scheduling.eval_steps": "int",
        "training.scheduling.save_strategy": "str",
        "training.scheduling.save_steps": "int",
        "training.scheduling.save_total_limit": "int",
        
        # Training - Dataloader
        "training.dataloader.num_workers": "int",
        "training.dataloader.pin_memory": "bool",
        "training.dataloader.persistent_workers": "bool",
        "training.dataloader.prefetch_factor": "int",
        "training.dataloader.drop_last": "bool",
        "training.dataloader.group_by_length": "bool",
        
        # Training - Epochs
        "training.epochs": "float",
        
        # Dataset
        "dataset.name": "str",
        "dataset.subset": "str",
        "dataset.split.test_size": "float",
        "dataset.split.seed": "int",
        "dataset.processing.max_length": "int",
        "dataset.processing.system_prompt": "str",
        
        # GRPO
        "grpo.num_generations": "int",
        "grpo.max_completion_length": "int",
        "grpo.vllm.enabled": "bool",
        "grpo.vllm.mode": "str",
        "grpo.vllm.gpu_memory_utilization": "float",
        "grpo.liger_loss": "bool",
        
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
        "system.seed": "int",
        "system.output_dir": "str",
        
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
    ]


def get_boolean_fields() -> List[str]:
    """Get list of configuration fields that expect boolean values."""
    field_types = get_config_field_types()
    return [field for field, field_type in field_types.items() if field_type == "bool"]


def get_numeric_fields() -> List[str]:
    """Get list of configuration fields that expect numeric values."""
    field_types = get_config_field_types()
    return [field for field, field_type in field_types.items() if field_type in ["int", "float"]]