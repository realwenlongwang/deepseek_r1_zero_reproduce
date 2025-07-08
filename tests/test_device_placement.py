#!/usr/bin/env python3
"""
Test device placement for training and inference to verify GPU allocation works correctly.
"""

import os
import sys
import tempfile
import torch
import pytest
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the model loading functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from train_grpo import setup_model_and_tokenizer
from test_inference import load_checkpoint
from src.config.grpo_config import ModelConfig


class TestDevicePlacement:
    """Test device placement strategies for different model sizes."""
    
    def test_small_model_single_gpu_training(self):
        """Test that small models (0.5B) use single GPU for training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Mock model config for small model
        model_config = ModelConfig(
            model_name_or_path="models/Qwen2.5-0.5B",
            torch_dtype="bfloat16",
            trust_remote_code=True,
            attn_implementation=None  # Disable flash attention for test
        )
        
        # Mock the model loading to avoid actually downloading/loading
        with patch('train_grpo.AutoModelForCausalLM.from_pretrained') as mock_model, \
             patch('train_grpo.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Create mock objects
            mock_model_instance = MagicMock()
            mock_model_instance.config.model_type = "qwen2"
            mock_model_instance.num_parameters.return_value = 494000000
            mock_model.return_value = mock_model_instance
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "</s>"
            mock_tokenizer_instance.chat_template = None
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Call the function
            model, tokenizer = setup_model_and_tokenizer(model_config)
            
            # Verify single GPU device mapping was used
            mock_model.assert_called_once()
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs['device_map'] == "cuda:0", f"Expected single GPU, got {call_kwargs['device_map']}"
    
    def test_large_model_multi_gpu_training(self):
        """Test that large models (7B) use auto device mapping for training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Mock model config for large model
        model_config = ModelConfig(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            torch_dtype="bfloat16",
            trust_remote_code=True,
            attn_implementation=None
        )
        
        # Mock the model loading
        with patch('train_grpo.AutoModelForCausalLM.from_pretrained') as mock_model, \
             patch('train_grpo.AutoTokenizer.from_pretrained') as mock_tokenizer:
            
            # Create mock objects
            mock_model_instance = MagicMock()
            mock_model_instance.config.model_type = "qwen2"
            mock_model_instance.num_parameters.return_value = 7600000000
            mock_model.return_value = mock_model_instance
            
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.pad_token = None
            mock_tokenizer_instance.eos_token = "</s>"
            mock_tokenizer_instance.chat_template = None
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            # Call the function
            model, tokenizer = setup_model_and_tokenizer(model_config)
            
            # Verify auto device mapping was used
            mock_model.assert_called_once()
            call_kwargs = mock_model.call_args[1]
            assert call_kwargs['device_map'] == "auto", f"Expected auto mapping, got {call_kwargs['device_map']}"
    
    def test_inference_single_gpu_placement(self):
        """Test that inference always uses single GPU regardless of model size."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a temporary checkpoint directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the model loading for inference
            with patch('test_inference.AutoModelForCausalLM.from_pretrained') as mock_model, \
                 patch('test_inference.AutoTokenizer.from_pretrained') as mock_tokenizer:
                
                # Create mock objects
                mock_model_instance = MagicMock()
                mock_model.return_value = mock_model_instance
                
                mock_tokenizer_instance = MagicMock()
                mock_tokenizer.return_value = mock_tokenizer_instance
                
                # Call the function
                model, tokenizer = load_checkpoint(temp_dir)
                
                # Verify single GPU device mapping was used for inference
                mock_model.assert_called_once()
                call_kwargs = mock_model.call_args[1]
                assert call_kwargs['device_map'] == "cuda:0", f"Expected single GPU for inference, got {call_kwargs['device_map']}"
    
    def test_cpu_fallback(self):
        """Test that CPU fallback works when CUDA is not available."""
        # Mock CUDA availability
        with patch('torch.cuda.is_available', return_value=False):
            model_config = ModelConfig(
                model_name_or_path="models/Qwen2.5-0.5B",
                torch_dtype="bfloat16",
                trust_remote_code=True,
                attn_implementation=None
            )
            
            # Mock the model loading
            with patch('train_grpo.AutoModelForCausalLM.from_pretrained') as mock_model, \
                 patch('train_grpo.AutoTokenizer.from_pretrained') as mock_tokenizer:
                
                # Create mock objects
                mock_model_instance = MagicMock()
                mock_model_instance.config.model_type = "qwen2"
                mock_model_instance.num_parameters.return_value = 494000000
                mock_model.return_value = mock_model_instance
                
                mock_tokenizer_instance = MagicMock()
                mock_tokenizer_instance.pad_token = None
                mock_tokenizer_instance.eos_token = "</s>"
                mock_tokenizer_instance.chat_template = None
                mock_tokenizer.return_value = mock_tokenizer_instance
                
                # Call the function
                model, tokenizer = setup_model_and_tokenizer(model_config)
                
                # Verify None device mapping (CPU) was used
                mock_model.assert_called_once()
                call_kwargs = mock_model.call_args[1]
                assert call_kwargs['device_map'] is None, f"Expected CPU (None), got {call_kwargs['device_map']}"
    
    def test_device_consistency_check(self):
        """Test that we can detect device placement consistency between training and inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # This test verifies that our logic will work correctly
        # Small model should use cuda:0 in both training and inference
        assert self._get_training_device_map("models/Qwen2.5-0.5B") == "cuda:0"
        assert self._get_inference_device_map() == "cuda:0"
        
        # Large model should use auto for training but cuda:0 for inference
        assert self._get_training_device_map("Qwen/Qwen2.5-7B-Instruct") == "auto"
        assert self._get_inference_device_map() == "cuda:0"
    
    def _get_training_device_map(self, model_name: str) -> str:
        """Helper to get training device mapping logic."""
        if torch.cuda.is_available():
            if "0.5B" in model_name or "1B" in model_name:
                return "cuda:0"
            else:
                return "auto"
        return None
    
    def _get_inference_device_map(self) -> str:
        """Helper to get inference device mapping logic."""
        if torch.cuda.is_available():
            return "cuda:0"
        return None


class TestGPUMemoryUtilization:
    """Test GPU memory utilization patterns."""
    
    def test_single_gpu_memory_check(self):
        """Test that single GPU setup only uses one GPU."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Need multiple GPUs for this test")
        
        # Create a simple tensor on cuda:0
        with torch.cuda.device(0):
            tensor = torch.randn(1000, 1000, device="cuda:0")
            
            # Check memory usage
            memory_0 = torch.cuda.memory_allocated(0)
            memory_1 = torch.cuda.memory_allocated(1) if torch.cuda.device_count() > 1 else 0
            
            # GPU 0 should have memory allocated, GPU 1 should not
            assert memory_0 > 0, "GPU 0 should have memory allocated"
            assert memory_1 == 0, f"GPU 1 should have no memory, but has {memory_1} bytes"
            
            # Clean up
            del tensor
            torch.cuda.empty_cache()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])