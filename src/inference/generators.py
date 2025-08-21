#!/usr/bin/env python3
"""
Generation engines for DeepSeek R1 Zero models.
Provides flexible text generation with various sampling strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import torch
from transformers import GenerationConfig

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Unified inference engine for text generation with DeepSeek R1 Zero models.
    """
    
    def __init__(self, model, tokenizer, device: str = "auto"):
        """
        Initialize the inference engine.
        
        Args:
            model: Loaded model (full model or LoRA adapter)
            tokenizer: Configured tokenizer
            device: Target device for inference
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Auto-detect device if needed
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        logger.info(f"Inference engine initialized on device: {self.device}")
    
    def generate(self, 
                prompt: str,
                max_new_tokens: int = 512,
                temperature: float = 0.7,
                top_p: float = 0.9,
                top_k: int = 50,
                repetition_penalty: float = 1.1,
                do_sample: bool = True,
                use_system_message: bool = True,
                custom_system_message: str = None,
                return_full_text: bool = False,
                **kwargs) -> str:
        """
        Generate response for a single prompt.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_p: Top-p (nucleus) sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling (vs greedy decoding)
            use_system_message: Whether to include system message
            custom_system_message: Custom system message
            return_full_text: Whether to return full text including prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Import here to avoid circular imports
        from .checkpoint_loader import format_messages_for_qwen25
        
        # Format messages
        messages = format_messages_for_qwen25(
            prompt, 
            use_system_message=use_system_message,
            custom_system_message=custom_system_message
        )
        
        # Apply chat template
        try:
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            # Fallback to simple prompt
            formatted_prompt = prompt
        
        # Tokenize input
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Move to device
        if torch.cuda.is_available() and self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Prepare generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        # Generate
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
        
        # Decode response
        if return_full_text:
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # Only return the new tokens
            input_length = inputs['input_ids'].shape[1]
            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        return response
    
    def generate_batch(self,
                      prompts: List[str],
                      max_new_tokens: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      batch_size: int = 4,
                      **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts in batches.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            batch_size: Batch size for processing
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            batch_responses = []
            for prompt in batch_prompts:
                try:
                    response = self.generate(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        **kwargs
                    )
                    batch_responses.append(response)
                except Exception as e:
                    logger.error(f"Failed to generate for prompt: {prompt[:50]}... Error: {e}")
                    batch_responses.append(f"[Generation failed: {str(e)}]")
            
            responses.extend(batch_responses)
        
        return responses
    
    def generate_with_multiple_attempts(self,
                                      prompt: str,
                                      num_attempts: int = 3,
                                      temperature: float = 0.8,
                                      **kwargs) -> List[str]:
        """
        Generate multiple responses for the same prompt with different sampling.
        
        Args:
            prompt: Input prompt
            num_attempts: Number of generation attempts
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        responses = []
        
        for attempt in range(num_attempts):
            try:
                # Vary temperature slightly for diversity
                attempt_temp = temperature + (attempt * 0.1)
                response = self.generate(
                    prompt=prompt,
                    temperature=attempt_temp,
                    do_sample=True,
                    **kwargs
                )
                responses.append(response)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                responses.append(f"[Attempt {attempt + 1} failed: {str(e)}]")
        
        return responses
    
    def generate_interactive(self, 
                           system_message: str = None,
                           max_turns: int = 10) -> None:
        """
        Interactive chat mode.
        
        Args:
            system_message: Optional system message
            max_turns: Maximum number of conversation turns
        """
        print("="*80)
        print("🤖 DeepSeek R1 Zero Interactive Chat")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to reset conversation history")
        print("="*80)
        
        conversation_history = []
        if system_message:
            conversation_history.append({"role": "system", "content": system_message})
        
        turn = 0
        while turn < max_turns:
            try:
                # Get user input
                user_input = input("\n👤 User: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'clear':
                    conversation_history = []
                    if system_message:
                        conversation_history.append({"role": "system", "content": system_message})
                    print("🧹 Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})
                
                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                
                try:
                    # Use conversation history for context
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        conversation_history,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    # Tokenize and generate
                    inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
                    if torch.cuda.is_available() and self.device == "cuda":
                        inputs = {k: v.cuda() for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=512,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    input_length = inputs['input_ids'].shape[1]
                    response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    print(response)
                    
                    # Add assistant response to history
                    conversation_history.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    print(f"[Generation failed: {e}]")
                
                turn += 1
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except EOFError:
                print("\n👋 Goodbye!")
                break


class GenerationPresets:
    """
    Predefined generation presets for different use cases.
    """
    
    @staticmethod
    def creative() -> Dict[str, Any]:
        """Creative writing preset with high diversity."""
        return {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "do_sample": True
        }
    
    @staticmethod
    def balanced() -> Dict[str, Any]:
        """Balanced preset for general use."""
        return {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "do_sample": True
        }
    
    @staticmethod
    def precise() -> Dict[str, Any]:
        """Precise preset for factual/mathematical content."""
        return {
            "temperature": 0.3,
            "top_p": 0.8,
            "top_k": 40,
            "repetition_penalty": 1.15,
            "do_sample": True
        }
    
    @staticmethod
    def deterministic() -> Dict[str, Any]:
        """Deterministic preset (greedy decoding)."""
        return {
            "temperature": 0.0,
            "do_sample": False,
            "repetition_penalty": 1.1
        }
    
    @staticmethod
    def reasoning() -> Dict[str, Any]:
        """Optimized for step-by-step reasoning tasks."""
        return {
            "temperature": 0.4,
            "top_p": 0.85,
            "top_k": 30,
            "repetition_penalty": 1.2,
            "do_sample": True
        }
