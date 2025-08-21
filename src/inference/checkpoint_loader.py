#!/usr/bin/env python3
"""
Automatic checkpoint detection and loading for DeepSeek R1 Zero models.
Supports both full model and LoRA adapter checkpoints with proper chat template handling.
"""

import os
import json
import logging
from typing import Tuple, Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

# Official Qwen2.5 ChatML template from tokenizer_config.json
OFFICIAL_QWEN25_TEMPLATE = """{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- "\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>" }}
    {%- for tool in tools %}
        {{- "\\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n" }}
{%- else %}
    {%- if messages[0]['role'] == 'system' %}
        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) or (message.role == "assistant" and not message.tool_calls) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role }}
        {%- if message.content %}
            {{- '\\n' + message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '\\n<tool_call>\\n{"name": "' }}
            {{- tool_call.name }}
            {{- '", "arguments": ' }}
            {{- tool_call.arguments | tojson }}
            {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}
    {%- elif message.role == "tool" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\\n<tool_response>\\n' }}
        {{- message.content }}
        {{- '\\n</tool_response>' }}
        {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\\n' }}
{%- endif %}"""

# Training-compatible ChatML template (matches exactly what training uses)
TRAINING_CHATML_TEMPLATE = "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"


def detect_checkpoint_type(checkpoint_path: str) -> str:
    """
    Auto-detect if checkpoint is full model or LoRA adapter.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        "lora_adapter" or "full_model"
        
    Raises:
        ValueError: If checkpoint type cannot be determined
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    # Check for LoRA adapter files
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    adapter_model_path = os.path.join(checkpoint_path, "adapter_model.safetensors")
    
    # Check for full model files
    model_safetensors_path = os.path.join(checkpoint_path, "model.safetensors")
    config_json_path = os.path.join(checkpoint_path, "config.json")
    
    if os.path.exists(adapter_config_path) and os.path.exists(adapter_model_path):
        return "lora_adapter"
    elif os.path.exists(model_safetensors_path) and os.path.exists(config_json_path):
        return "full_model"
    else:
        raise ValueError(f"Unknown checkpoint format at {checkpoint_path}. "
                        f"Expected either LoRA adapter files (adapter_config.json, adapter_model.safetensors) "
                        f"or full model files (model.safetensors, config.json)")


def get_base_model_from_adapter_config(checkpoint_path: str) -> str:
    """
    Extract base model name from LoRA adapter config.
    
    Args:
        checkpoint_path: Path to LoRA adapter checkpoint
        
    Returns:
        Base model name/path
    """
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        raise ValueError(f"adapter_config.json not found at {checkpoint_path}")
    
    with open(adapter_config_path, 'r') as f:
        config = json.load(f)
    
    base_model = config.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(f"base_model_name_or_path not found in adapter_config.json")
    
    return base_model


def configure_tokenizer_for_qwen25(tokenizer, checkpoint_type: str, model_name: str = None):
    """
    Configure tokenizer with appropriate chat template for Qwen2.5.
    Uses the same logic as training script to ensure consistency.
    
    Args:
        tokenizer: HuggingFace tokenizer
        checkpoint_type: "lora_adapter" or "full_model"
        model_name: Model name for template detection
        
    Returns:
        Configured tokenizer
    """
    # Ensure pad token is set (same as training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Set chat template using Unsloth's method if available (same logic as training)
    if tokenizer.chat_template is None:
        try:
            from unsloth.chat_templates import get_chat_template
            
            # Determine appropriate template based on model (same as training)
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
                # mapping={"role": "role", "content": "content"},
                map_eos_token=True
            )
            
            logger.info("✅ Successfully applied Unsloth chat template")
            
        except ImportError:
            logger.warning("⚠️ Unsloth chat_templates not available, using ChatML fallback")
            tokenizer.chat_template = TRAINING_CHATML_TEMPLATE
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply Unsloth chat template: {e}")
            tokenizer.chat_template = TRAINING_CHATML_TEMPLATE
    else:
        logger.info("Using existing chat template from tokenizer")
    
    return tokenizer


class AutoCheckpointLoader:
    """
    Automatic checkpoint loader that handles both full models and LoRA adapters.
    """
    
    def __init__(self, device_map: str = "auto", torch_dtype = torch.bfloat16):
        """
        Initialize the checkpoint loader.
        
        Args:
            device_map: Device mapping strategy ("auto", "cuda:0", etc.)
            torch_dtype: Torch data type for model loading
        """
        self.device_map = device_map
        self.torch_dtype = torch_dtype
    
    def load_checkpoint(self, checkpoint_path: str, use_local_files_only: bool = False) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load checkpoint with automatic type detection.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            use_local_files_only: Whether to use only local files
            
        Returns:
            Tuple of (model, tokenizer, metadata)
        """
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        
        # Detect checkpoint type
        checkpoint_type = detect_checkpoint_type(checkpoint_path)
        logger.info(f"Detected checkpoint type: {checkpoint_type}")
        # return self._load_full_model(checkpoint_path, use_local_files_only)
        if checkpoint_type == "lora_adapter":
            return self._load_lora_adapter(checkpoint_path, use_local_files_only)
        else:
            return self._load_full_model(checkpoint_path, use_local_files_only)
        
    
    def _load_lora_adapter(self, checkpoint_path: str, use_local_files_only: bool) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load LoRA adapter checkpoint using Unsloth if available."""
        # Get base model info
        base_model_name = get_base_model_from_adapter_config(checkpoint_path)
        logger.info(f"Base model for LoRA adapter: {base_model_name}")
        
        # Resolve base model path
        base_model_path = self._resolve_base_model_path(base_model_name)
        logger.info(f"Loading base model from: {base_model_path}")
        
        # Try to use Unsloth for optimal inference performance
        model, tokenizer = self._try_load_with_unsloth(base_model_path, use_local_files_only)

        
        if model is None:
            # Fallback to standard loading
            logger.info("🔄 Falling back to standard model loading")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                local_files_only=use_local_files_only,
                trust_remote_code=True
            )
            
            # Load tokenizer from base model
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                local_files_only=use_local_files_only,
                trust_remote_code=True
            )
            
            # Apply LoRA adapter
            model = PeftModel.from_pretrained(
                base_model,
                checkpoint_path,
                local_files_only=use_local_files_only
            )
        else:
            # Load LoRA adapter on Unsloth model
            logger.info("📎 Loading LoRA adapter on Unsloth model")
            try:
                model = PeftModel.from_pretrained(
                    model,
                    checkpoint_path,
                    local_files_only=use_local_files_only
                )
                logger.info("✅ LoRA adapter loaded on Unsloth model")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load LoRA on Unsloth model: {e}")
                logger.info("🔄 Retrying with standard approach")
                
                # Reload with standard approach
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    local_files_only=use_local_files_only,
                    trust_remote_code=True
                )
                
                model = PeftModel.from_pretrained(
                    base_model,
                    checkpoint_path,
                    local_files_only=use_local_files_only
                )
        
        # Configure tokenizer
        tokenizer = configure_tokenizer_for_qwen25(tokenizer, "lora_adapter", base_model_name)
        
        # Enable fast inference for Unsloth models
        try:
            from unsloth import FastLanguageModel
            # Try to enable fast inference mode
            model = FastLanguageModel.for_inference(model)
            logger.info("🚀 Enabled Unsloth fast inference mode")
        except Exception as e:
            logger.info(f"ℹ️ Unsloth fast inference not available: {e}")
        
        metadata = {
            "checkpoint_type": "lora_adapter",
            "base_model": base_model_name,
            "adapter_path": checkpoint_path,
            "model_parameters": model.num_parameters() if hasattr(model, 'num_parameters') else "unknown",
        }
        
        logger.info(f"✅ Successfully loaded LoRA adapter")
        logger.info(f"Base model: {base_model_name}")
        logger.info(f"Adapter: {checkpoint_path}")
        
        return model, tokenizer, metadata
    
    def _try_load_with_unsloth(self, base_model_path: str, use_local_files_only: bool) -> Tuple[Any, Any]:
        """
        Try to load base model with Unsloth for optimal inference performance.
        
        Returns:
            Tuple of (model, tokenizer) or (None, None) if Unsloth not available
        """
        try:
            from unsloth import FastLanguageModel
            
            logger.info("🚀 Attempting to load with Unsloth FastLanguageModel")
            
            # Use similar parameters to training script
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model_path,
                max_seq_length=2048,  # Reasonable default for inference
                load_in_4bit=True,    # Enable 4-bit for memory efficiency
            )
            logger.info("✅ Successfully loaded base model with Unsloth")
            return model, tokenizer
            
        except ImportError:
            logger.info("ℹ️ Unsloth not available, will use standard loading")
            return None, None
        except Exception as e:
            logger.warning(f"⚠️ Failed to load with Unsloth: {e}")
            logger.info("🔄 Will fallback to standard loading")
            return None, None
    
    def _load_full_model(self, checkpoint_path: str, use_local_files_only: bool) -> Tuple[Any, Any, Dict[str, Any]]:
        """Load full model checkpoint, optionally with Unsloth optimization."""
        
        # Standard loading
        logger.info("📚 Loading with standard AutoModelForCausalLM")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            local_files_only=use_local_files_only,
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint_path,
            local_files_only=use_local_files_only,
            trust_remote_code=True
        )

        tokenizer = configure_tokenizer_for_qwen25(tokenizer, "full_model", checkpoint_path)
        
        metadata = {
            "checkpoint_type": "full_model",
            "model_path": checkpoint_path,
            "model_type": model.config.model_type if hasattr(model, 'config') else "unknown",
            "model_parameters": model.num_parameters() if hasattr(model, 'num_parameters') else "unknown",
        }
        
        logger.info(f"✅ Successfully loaded full model")
        logger.info(f"Model: {checkpoint_path}")
        logger.info(f"Type: {metadata['model_type']}")
        logger.info(f"Parameters: {metadata['model_parameters']:,}")
        
        return model, tokenizer, metadata
    
    def _resolve_base_model_path(self, base_model_name: str) -> str:
        """
        Resolve base model path, checking local models directory first.
        
        Args:
            base_model_name: Base model name from adapter config
            
        Returns:
            Resolved model path
        """
        # Check if it's already a local path
        if os.path.exists(base_model_name):
            return base_model_name
        
        # Check for local models directory
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        
        # Try to find matching model in local models directory
        if os.path.exists(models_dir):
            for model_dir in os.listdir(models_dir):
                model_path = os.path.join(models_dir, model_dir)
                if os.path.isdir(model_path):
                    # Check if this matches the base model name
                    if base_model_name.endswith(model_dir) or model_dir in base_model_name:
                        logger.info(f"Found local model: {model_path}")
                        return model_path
        
        # Fall back to original name (will download from HuggingFace)
        logger.info(f"Using HuggingFace model: {base_model_name}")
        return base_model_name


def find_latest_checkpoint(base_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory structure.
    
    Args:
        base_dir: Base directory to search (e.g., "saved_models")
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(base_dir):
        return None
    
    latest_checkpoint = None
    latest_time = 0
    
    # Search in base directory
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        if os.path.isdir(item_path):
            # Check if this is a checkpoint directory
            try:
                checkpoint_type = detect_checkpoint_type(item_path)
                modification_time = os.path.getmtime(item_path)
                
                if modification_time > latest_time:
                    latest_time = modification_time
                    latest_checkpoint = item_path
            except ValueError:
                # Not a valid checkpoint, check subdirectories
                for subitem in os.listdir(item_path):
                    subitem_path = os.path.join(item_path, subitem)
                    
                    if os.path.isdir(subitem_path):
                        try:
                            checkpoint_type = detect_checkpoint_type(subitem_path)
                            modification_time = os.path.getmtime(subitem_path)
                            
                            if modification_time > latest_time:
                                latest_time = modification_time
                                latest_checkpoint = subitem_path
                        except ValueError:
                            continue
    
    return latest_checkpoint


def format_messages_for_qwen25(prompt: str, use_system_message: bool = True, 
                              custom_system_message: str = None) -> list:
    """
    Format messages using Qwen2.5 best practices with DeepSeek R1 system prompt.
    
    Args:
        prompt: User prompt
        use_system_message: Whether to include system message
        custom_system_message: Custom system message (uses DeepSeek R1 default if None)
        
    Returns:
        Formatted messages list
    """
    messages = []
    
    if use_system_message:
        # Use the exact same system prompt as training (from src/data/dataset.py)
        if custom_system_message is None:
            system_msg = (
                "A conversation between User and Assistant. The user asks a question, "
                "and the Assistant solves it. The assistant "
                "first thinks about the reasoning process in the mind and "
                "then provides the user with the answer. The reasoning "
                "process and answer are enclosed within <think> </think> "
                "and <answer> </answer> tags, respectively, i.e., "
                "<think> reasoning process here </think>"
                "<answer> answer here </answer>"
            )
        else:
            system_msg = custom_system_message
        messages.append({"role": "system", "content": system_msg})
    
    messages.append({"role": "user", "content": prompt})
    
    return messages