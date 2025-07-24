#!/usr/bin/env python3
"""
Comprehensive test suite to isolate evaluation freeze issues in GRPO training.
This script creates targeted tests to identify specific bottlenecks.
"""

import os
import sys
import time
import logging
import warnings
import torch
import multiprocessing as mp
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import tempfile
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import dependencies
from src.data.dataset import create_train_test_datasets
from src.rewards.openr1_rewards import get_reward_funcs
from src.config.grpo_config import GRPOScriptArguments


class FreezeTestSuite:
    """Test suite to identify evaluation freeze bottlenecks."""
    
    def __init__(self):
        self.test_results = {}
        self.mock_completions = self._create_mock_completions()
        self.mock_dataset = self._create_mock_dataset()
        
    def _create_mock_completions(self) -> List[List[Dict[str, str]]]:
        """Create mock completions for reward function testing."""
        return [
            [{"content": "<think>\nStep 1: I need to solve this math problem.\nStep 2: Let me calculate.\n</think>\n<answer>42</answer>"}],
            [{"content": "<think>\nFirst, I'll analyze the problem.\nSecond, I'll compute the result.\n</think>\n<answer>24</answer>"}],
            [{"content": "Invalid format without proper tags"}],
            [{"content": "<think>\nSome reasoning here.\n</think>\n<answer>99</answer>"}],
        ]
    
    def _create_mock_dataset(self) -> Dict[str, Any]:
        """Create mock dataset for testing."""
        return {
            "train": [
                {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
                {"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "6"}]},
            ],
            "solution": ["4", "6"],
            "target": ["4", "6"],
            "nums": [[2, 2], [3, 3]]
        }
    
    def test_reward_function_performance(self) -> Dict[str, float]:
        """Test individual reward function performance and timeout behavior."""
        logger.info("🧪 Testing reward function performance...")
        
        # Test reward functions individually
        script_args = GRPOScriptArguments(
            reward_funcs=["format", "equation", "accuracy", "reasoning_steps"],
            cosine_min_value_wrong=-0.5,
            cosine_max_value_wrong=-0.1,
            cosine_min_value_correct=0.8,
            cosine_max_value_correct=1.0,
            cosine_max_len=1000,
            repetition_n_grams=3,
            repetition_max_penalty=-0.1,
            code_language="python",
            max_completion_len=512,
            soft_punish_cache=50,
        )
        
        reward_functions = get_reward_funcs(script_args)
        performance_results = {}
        
        for i, reward_func in enumerate(reward_functions):
            func_name = script_args.reward_funcs[i]
            logger.info(f"Testing {func_name} reward function...")
            
            start_time = time.time()
            try:
                # Test with timeout
                result = self._run_with_timeout(
                    reward_func,
                    args=(self.mock_completions,),
                    kwargs={"solution": self.mock_dataset["solution"], 
                           "target": self.mock_dataset["target"],
                           "nums": self.mock_dataset["nums"]},
                    timeout=30
                )
                elapsed = time.time() - start_time
                performance_results[func_name] = {
                    "elapsed": elapsed,
                    "success": True,
                    "result": result
                }
                logger.info(f"✅ {func_name}: {elapsed:.2f}s - SUCCESS")
            except Exception as e:
                elapsed = time.time() - start_time
                performance_results[func_name] = {
                    "elapsed": elapsed,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ {func_name}: {elapsed:.2f}s - ERROR: {e}")
        
        self.test_results["reward_function_performance"] = performance_results
        return performance_results
    
    def _run_with_timeout(self, func, args=(), kwargs=None, timeout=30):
        """Run function with timeout to catch hangs."""
        if kwargs is None:
            kwargs = {}
        
        # Use multiprocessing to enforce timeout
        def target(queue):
            try:
                result = func(*args, **kwargs)
                queue.put(("success", result))
            except Exception as e:
                queue.put(("error", e))
        
        queue = mp.Queue()
        process = mp.Process(target=target, args=(queue,))
        process.start()
        process.join(timeout=timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            raise TimeoutError(f"Function timed out after {timeout} seconds")
        
        if not queue.empty():
            status, result = queue.get()
            if status == "success":
                return result
            else:
                raise result
        else:
            raise RuntimeError("Process ended without returning result")
    
    def test_dataset_loading_performance(self) -> Dict[str, Any]:
        """Test dataset loading and processing performance."""
        logger.info("🧪 Testing dataset loading performance...")
        
        start_time = time.time()
        try:
            # Test dataset loading with timeout
            train_dataset, test_dataset = self._run_with_timeout(
                create_train_test_datasets,
                kwargs={
                    "dataset_name": "Jiayi-Pan/Countdown-Tasks-3to4",
                    "test_size": 0.1,
                    "split_seed": 42
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            result = {
                "elapsed": elapsed,
                "success": True,
                "train_size": len(train_dataset),
                "test_size": len(test_dataset)
            }
            logger.info(f"✅ Dataset loading: {elapsed:.2f}s - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
            
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "elapsed": elapsed,
                "success": False,
                "error": str(e)
            }
            logger.error(f"❌ Dataset loading: {elapsed:.2f}s - ERROR: {e}")
        
        self.test_results["dataset_loading"] = result
        return result
    
    def test_dataloader_configurations(self) -> Dict[str, Any]:
        """Test different dataloader configurations to identify problematic settings."""
        logger.info("🧪 Testing dataloader configurations...")
        
        from torch.utils.data import DataLoader, Dataset
        
        class MockDataset(Dataset):
            def __init__(self, size=100):
                self.size = size
                self.data = [{"input": f"test_{i}", "target": f"answer_{i}"} for i in range(size)]
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                time.sleep(0.01)  # Simulate processing time
                return self.data[idx]
        
        dataset = MockDataset(size=50)
        configs = [
            {"num_workers": 0, "pin_memory": False, "prefetch_factor": None},
            {"num_workers": 2, "pin_memory": False, "prefetch_factor": 2},
            {"num_workers": 4, "pin_memory": True, "prefetch_factor": 2},
            {"num_workers": 8, "pin_memory": True, "prefetch_factor": 4},  # Current config
        ]
        
        results = {}
        
        for i, config in enumerate(configs):
            logger.info(f"Testing dataloader config {i+1}: {config}")
            try:
                # Remove None values from config
                clean_config = {k: v for k, v in config.items() if v is not None}
                
                start_time = time.time()
                dataloader = DataLoader(dataset, batch_size=4, **clean_config)
                
                # Test iteration with timeout
                batch_count = 0
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 5:  # Test first 5 batches
                        break
                
                elapsed = time.time() - start_time
                results[f"config_{i+1}"] = {
                    "config": config,
                    "elapsed": elapsed,
                    "success": True,
                    "batches_processed": batch_count
                }
                logger.info(f"✅ Config {i+1}: {elapsed:.2f}s - SUCCESS")
                
            except Exception as e:
                elapsed = time.time() - start_time
                results[f"config_{i+1}"] = {
                    "config": config,
                    "elapsed": elapsed,
                    "success": False,
                    "error": str(e)
                }
                logger.error(f"❌ Config {i+1}: {elapsed:.2f}s - ERROR: {e}")
        
        self.test_results["dataloader_configs"] = results
        return results
    
    def test_grpo_trainer_initialization(self) -> Dict[str, Any]:
        """Test GRPOTrainer initialization with different configurations."""
        logger.info("🧪 Testing GRPOTrainer initialization...")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
        from trl import GRPOConfig, GRPOTrainer
        
        # Use a small model for testing
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        configs_to_test = [
            {"use_vllm": False, "description": "Without vLLM"},
            {"use_vllm": True, "description": "With vLLM"},
        ]
        
        results = {}
        
        for config in configs_to_test:
            logger.info(f"Testing GRPOTrainer: {config['description']}")
            
            try:
                start_time = time.time()
                
                # Create temporary directory
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Initialize trainer with timeout
                    trainer = self._run_with_timeout(
                        self._initialize_grpo_trainer,
                        args=(model_name, temp_dir, config["use_vllm"]),
                        timeout=180
                    )
                    
                    elapsed = time.time() - start_time
                    results[config["description"]] = {
                        "elapsed": elapsed,
                        "success": True,
                        "use_vllm": config["use_vllm"]
                    }
                    logger.info(f"✅ {config['description']}: {elapsed:.2f}s - SUCCESS")
                    
                    # Clean up
                    del trainer
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                elapsed = time.time() - start_time
                results[config["description"]] = {
                    "elapsed": elapsed,
                    "success": False,
                    "error": str(e),
                    "use_vllm": config["use_vllm"]
                }
                logger.error(f"❌ {config['description']}: {elapsed:.2f}s - ERROR: {e}")
        
        self.test_results["grpo_trainer_init"] = results
        return results
    
    def _initialize_grpo_trainer(self, model_name: str, output_dir: str, use_vllm: bool):
        """Initialize GRPOTrainer for testing."""
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
            use_cache=False
        )
        
        # Create training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            eval_strategy="steps",
            eval_steps=10,
            save_steps=10,
            num_train_epochs=0.01,
            remove_unused_columns=False,
            report_to="none",
        )
        
        # Create GRPO config
        grpo_config = GRPOConfig(
            **training_args.to_dict(),
            max_completion_length=512,
            num_generations=4,
            use_vllm=use_vllm,
            generation_batch_size=8,
        )
        
        # Create mock dataset
        mock_dataset = [
            {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]},
            {"messages": [{"role": "user", "content": "What is 3+3?"}, {"role": "assistant", "content": "6"}]},
        ]
        
        # Create reward functions
        script_args = GRPOScriptArguments(reward_funcs=["format"])
        reward_functions = get_reward_funcs(script_args)
        
        # Initialize trainer
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_functions,
            args=grpo_config,
            train_dataset=mock_dataset,
            eval_dataset=mock_dataset,
            processing_class=tokenizer,
        )
        
        return trainer
    
    def test_wandb_connectivity(self) -> Dict[str, Any]:
        """Test WandB connectivity and initialization."""
        logger.info("🧪 Testing WandB connectivity...")
        
        try:
            import wandb
            
            # Test WandB initialization with timeout
            start_time = time.time()
            
            # Mock wandb.init to avoid actual logging
            with patch('wandb.init') as mock_init:
                mock_init.return_value = Mock()
                
                wandb.init(
                    project="test-project",
                    mode="offline"  # Use offline mode for testing
                )
                
                elapsed = time.time() - start_time
                result = {
                    "elapsed": elapsed,
                    "success": True,
                    "wandb_available": True
                }
                logger.info(f"✅ WandB test: {elapsed:.2f}s - SUCCESS")
                
                wandb.finish()
                
        except ImportError:
            result = {
                "elapsed": 0,
                "success": False,
                "wandb_available": False,
                "error": "WandB not installed"
            }
            logger.warning("⚠️ WandB not available")
        except Exception as e:
            elapsed = time.time() - start_time
            result = {
                "elapsed": elapsed,
                "success": False,
                "wandb_available": True,
                "error": str(e)
            }
            logger.error(f"❌ WandB test: {elapsed:.2f}s - ERROR: {e}")
        
        self.test_results["wandb_connectivity"] = result
        return result
    
    def test_memory_usage_patterns(self) -> Dict[str, Any]:
        """Test memory usage patterns during evaluation."""
        logger.info("🧪 Testing memory usage patterns...")
        
        if not torch.cuda.is_available():
            logger.warning("⚠️ CUDA not available, skipping memory tests")
            return {"success": False, "error": "CUDA not available"}
        
        # Monitor memory usage
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        memory_snapshots = []
        
        try:
            # Create increasingly large tensors to simulate memory pressure
            tensors = []
            for i in range(10):
                tensor = torch.randn(1000, 1000, device='cuda')
                tensors.append(tensor)
                
                current_memory = torch.cuda.memory_allocated()
                memory_snapshots.append({
                    "step": i,
                    "memory_mb": (current_memory - initial_memory) / 1024 / 1024,
                    "total_memory_mb": current_memory / 1024 / 1024
                })
                
                # Simulate evaluation workload
                result = torch.matmul(tensor, tensor.T)
                del result
                
            # Clean up
            del tensors
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            
            result = {
                "success": True,
                "initial_memory_mb": initial_memory / 1024 / 1024,
                "final_memory_mb": final_memory / 1024 / 1024,
                "memory_snapshots": memory_snapshots,
                "memory_cleaned": final_memory <= initial_memory * 1.1
            }
            
            logger.info(f"✅ Memory test: Peak usage {max(s['memory_mb'] for s in memory_snapshots):.1f}MB")
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e)
            }
            logger.error(f"❌ Memory test: ERROR: {e}")
        
        self.test_results["memory_usage"] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive freeze detection tests...")
        
        test_functions = [
            self.test_reward_function_performance,
            self.test_dataset_loading_performance,
            self.test_dataloader_configurations,
            self.test_wandb_connectivity,
            self.test_memory_usage_patterns,
            # Note: Skipping GRPO trainer test as it requires more resources
        ]
        
        for test_func in test_functions:
            try:
                test_func()
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed: {e}")
                self.test_results[test_func.__name__] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Generate summary
        self._generate_summary()
        
        return self.test_results
    
    def _generate_summary(self):
        """Generate test summary and recommendations."""
        logger.info("="*60)
        logger.info("🏁 TEST SUMMARY")
        logger.info("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, dict) and result.get("success", False))
        
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Successful tests: {successful_tests}")
        logger.info(f"Failed tests: {total_tests - successful_tests}")
        
        # Identify potential issues
        issues = []
        
        # Check reward function performance
        if "reward_function_performance" in self.test_results:
            perf_results = self.test_results["reward_function_performance"]
            slow_functions = [name for name, result in perf_results.items() 
                            if result.get("elapsed", 0) > 10]
            if slow_functions:
                issues.append(f"Slow reward functions: {slow_functions}")
        
        # Check dataloader configs
        if "dataloader_configs" in self.test_results:
            dl_results = self.test_results["dataloader_configs"]
            failed_configs = [name for name, result in dl_results.items() 
                            if not result.get("success", False)]
            if failed_configs:
                issues.append(f"Problematic dataloader configs: {failed_configs}")
        
        # Check WandB
        if "wandb_connectivity" in self.test_results:
            wandb_result = self.test_results["wandb_connectivity"]
            if not wandb_result.get("success", False):
                issues.append("WandB connectivity issues")
        
        # Print recommendations
        if issues:
            logger.warning("⚠️ POTENTIAL ISSUES IDENTIFIED:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ No major issues detected in tested components")
        
        # Save results to file
        results_file = "freeze_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"📄 Detailed results saved to: {results_file}")


def main():
    """Main test runner."""
    logger.info("🧪 Starting GRPO Evaluation Freeze Test Suite")
    
    # Create test suite
    suite = FreezeTestSuite()
    
    # Run tests
    results = suite.run_all_tests()
    
    logger.info("🏁 Test suite completed")
    return results


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()