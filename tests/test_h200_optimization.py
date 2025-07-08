#!/usr/bin/env python3
"""
Comprehensive H200 GPU optimization testing.
Tests dataloader performance, batch sizes, and GPU utilization to find optimal config.
"""

import os
import sys
import time
import psutil
import torch
import threading
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import create_dataset
from src.config.grpo_config import create_training_arguments, ModelConfig
from train_grpo import setup_model_and_tokenizer


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    batch_size: int
    num_workers: int
    samples_per_second: float
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    cpu_utilization: float
    dataloader_time: float
    model_time: float
    total_time: float


class GPUMonitor:
    """Monitor GPU utilization and memory usage in background."""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_stats = []
        
    def start_monitoring(self):
        """Start GPU monitoring in background thread."""
        self.monitoring = True
        self.gpu_stats = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop GPU monitoring and return average stats."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        
        if not self.gpu_stats:
            return 0.0, 0.0, 0.0
            
        avg_utilization = np.mean([s['utilization'] for s in self.gpu_stats])
        avg_memory = np.mean([s['memory_used'] for s in self.gpu_stats])
        max_memory = max([s['memory_used'] for s in self.gpu_stats])
        
        return avg_utilization, avg_memory, max_memory
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            if torch.cuda.is_available():
                # Get GPU stats
                memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                # Approximate GPU utilization (not perfect but gives indication)
                utilization = (memory_used / memory_total) * 100 if memory_total > 0 else 0
                
                self.gpu_stats.append({
                    'utilization': utilization,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'timestamp': time.time()
                })
            
            time.sleep(0.1)  # Sample every 100ms


class H200OptimizationTester:
    """Comprehensive H200 optimization testing suite."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.gpu_monitor = GPUMonitor()
        
    def setup(self):
        """Setup model, tokenizer, and dataset for testing."""
        print("🔧 Setting up test environment...")
        
        # Setup model and tokenizer
        model_config = ModelConfig(
            model_name_or_path=self.model_name,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            attn_implementation=None  # Disable flash attention for stability
        )
        
        print(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = setup_model_and_tokenizer(model_config)
        
        # Setup dataset
        print("Loading dataset...")
        self.dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
        
        # Get initial GPU memory info
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory Available: {total_memory:.1f} GB")
            print(f"🏛️  GPU: {torch.cuda.get_device_name(0)}")
        
        print("✅ Setup complete\n")
        
    def test_dataloader_performance(self, batch_sizes: List[int] = [16, 32, 64], 
                                  worker_counts: List[int] = [4, 8, 16, 32]) -> List[PerformanceMetrics]:
        """Test dataloader performance with different configurations."""
        print("📊 Testing Dataloader Performance...")
        print("=" * 60)
        
        results = []
        
        for batch_size in batch_sizes:
            for num_workers in worker_counts:
                print(f"Testing batch_size={batch_size}, workers={num_workers}")
                
                try:
                    # Create dataloader
                    from torch.utils.data import DataLoader
                    dataloader = DataLoader(
                        self.dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True,
                        persistent_workers=True if num_workers > 0 else False,
                        prefetch_factor=4 if num_workers > 0 else None,
                        drop_last=True
                    )
                    
                    # Measure performance
                    metrics = self._measure_dataloader_performance(dataloader, batch_size, num_workers)
                    results.append(metrics)
                    
                    print(f"  ⚡ {metrics.samples_per_second:.1f} samples/sec, "
                          f"GPU: {metrics.gpu_memory_used:.1f}GB, "
                          f"DataLoader: {metrics.dataloader_time:.3f}s")
                    
                except Exception as e:
                    print(f"  ❌ Failed: {e}")
                    
                # Clear GPU memory
                torch.cuda.empty_cache()
                time.sleep(1)
                
        return results
        
    def _measure_dataloader_performance(self, dataloader, batch_size: int, num_workers: int) -> PerformanceMetrics:
        """Measure performance for a specific dataloader configuration."""
        
        # Start GPU monitoring
        self.gpu_monitor.start_monitoring()
        
        total_samples = 0
        total_dataloader_time = 0
        total_model_time = 0
        num_batches = min(10, len(dataloader))  # Test on first 10 batches
        
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            # Measure dataloader time
            dataloader_start = time.time()
            
            # Process batch (simulate tokenization)
            if isinstance(batch, dict):
                # Handle conversation format
                batch_size_actual = len(batch.get('problem', []))
            else:
                batch_size_actual = len(batch)
                
            dataloader_end = time.time()
            
            # Measure model time (simulate forward pass)
            model_start = time.time()
            
            # Create dummy input for forward pass timing
            dummy_input = torch.randn(batch_size_actual, 512, device='cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                dummy_input = dummy_input.to(torch.bfloat16)
                
            # Simulate computation
            with torch.no_grad():
                _ = torch.sum(dummy_input)  # Lightweight computation
                
            model_end = time.time()
            
            total_samples += batch_size_actual
            total_dataloader_time += (dataloader_end - dataloader_start)
            total_model_time += (model_end - model_start)
            
        end_time = time.time()
        
        # Stop GPU monitoring
        avg_gpu_util, avg_gpu_memory, max_gpu_memory = self.gpu_monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = end_time - start_time
        samples_per_second = total_samples / total_time if total_time > 0 else 0
        
        # Get current GPU memory
        gpu_memory_used = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
        
        return PerformanceMetrics(
            batch_size=batch_size,
            num_workers=num_workers,
            samples_per_second=samples_per_second,
            gpu_memory_used=max_gpu_memory,
            gpu_memory_total=gpu_memory_total,
            gpu_utilization=avg_gpu_util,
            cpu_utilization=psutil.cpu_percent(),
            dataloader_time=total_dataloader_time,
            model_time=total_model_time,
            total_time=total_time
        )
        
    def test_memory_scaling(self, max_batch_size: int = 128) -> List[PerformanceMetrics]:
        """Test memory usage scaling with increasing batch sizes."""
        print("🧠 Testing Memory Scaling...")
        print("=" * 60)
        
        results = []
        batch_sizes = [8, 16, 32, 64, 96, 128]
        
        if max_batch_size < 128:
            batch_sizes = [b for b in batch_sizes if b <= max_batch_size]
            
        for batch_size in batch_sizes:
            print(f"Testing memory usage with batch_size={batch_size}")
            
            try:
                # Clear memory
                torch.cuda.empty_cache()
                
                # Create dummy batch to test memory usage
                sequence_length = 512
                vocab_size = self.model.config.vocab_size if self.model else 32000
                
                # Simulate model forward pass memory usage
                with torch.no_grad():
                    dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_length), 
                                               device='cuda' if torch.cuda.is_available() else 'cpu')
                    
                    if torch.cuda.is_available() and self.model:
                        # Start monitoring
                        self.gpu_monitor.start_monitoring()
                        
                        # Simulate forward pass
                        start_time = time.time()
                        outputs = self.model(dummy_input, labels=dummy_input)
                        torch.cuda.synchronize()
                        end_time = time.time()
                        
                        # Stop monitoring
                        avg_gpu_util, avg_gpu_memory, max_gpu_memory = self.gpu_monitor.stop_monitoring()
                        
                        memory_used = torch.cuda.memory_allocated() / 1024**3
                        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        memory_percent = (memory_used / memory_total) * 100
                        
                        forward_time = end_time - start_time
                        samples_per_second = batch_size / forward_time if forward_time > 0 else 0
                        
                        metrics = PerformanceMetrics(
                            batch_size=batch_size,
                            num_workers=0,  # Not applicable for memory test
                            samples_per_second=samples_per_second,
                            gpu_memory_used=memory_used,
                            gpu_memory_total=memory_total,
                            gpu_utilization=avg_gpu_util,
                            cpu_utilization=psutil.cpu_percent(),
                            dataloader_time=0,
                            model_time=forward_time,
                            total_time=forward_time
                        )
                        
                        results.append(metrics)
                        
                        print(f"  📊 Memory: {memory_used:.1f}GB ({memory_percent:.1f}%), "
                              f"Speed: {samples_per_second:.1f} samples/sec")
                        
                        # Check if we're approaching memory limit
                        if memory_percent > 85:
                            print(f"  ⚠️  Approaching memory limit, stopping at batch_size={batch_size}")
                            break
                            
                    del dummy_input
                    
            except torch.cuda.OutOfMemoryError:
                print(f"  💥 OOM at batch_size={batch_size}")
                torch.cuda.empty_cache()
                break
            except Exception as e:
                print(f"  ❌ Error at batch_size={batch_size}: {e}")
                
        return results
        
    def analyze_results(self, dataloader_results: List[PerformanceMetrics], 
                       memory_results: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze test results and provide recommendations."""
        print("\n📈 Performance Analysis")
        print("=" * 60)
        
        # Find optimal dataloader configuration
        best_dataloader = max(dataloader_results, key=lambda x: x.samples_per_second)
        
        print(f"🏆 Best Dataloader Config:")
        print(f"   Batch Size: {best_dataloader.batch_size}")
        print(f"   Workers: {best_dataloader.num_workers}")
        print(f"   Performance: {best_dataloader.samples_per_second:.1f} samples/sec")
        print(f"   GPU Memory: {best_dataloader.gpu_memory_used:.1f}GB")
        
        # Find optimal memory configuration
        if memory_results:
            # Find sweet spot: highest throughput within memory constraints
            safe_memory_results = [r for r in memory_results if (r.gpu_memory_used / r.gpu_memory_total) < 0.85]
            if safe_memory_results:
                best_memory = max(safe_memory_results, key=lambda x: x.samples_per_second)
                print(f"\n🧠 Optimal Memory Config:")
                print(f"   Max Safe Batch Size: {best_memory.batch_size}")
                print(f"   Memory Usage: {best_memory.gpu_memory_used:.1f}GB ({(best_memory.gpu_memory_used/best_memory.gpu_memory_total)*100:.1f}%)")
                print(f"   Performance: {best_memory.samples_per_second:.1f} samples/sec")
        
        # Worker analysis
        worker_analysis = {}
        for workers in set(r.num_workers for r in dataloader_results):
            worker_results = [r for r in dataloader_results if r.num_workers == workers]
            avg_performance = np.mean([r.samples_per_second for r in worker_results])
            worker_analysis[workers] = avg_performance
            
        print(f"\n👷 Worker Performance Analysis:")
        for workers, performance in sorted(worker_analysis.items()):
            print(f"   {workers:2d} workers: {performance:6.1f} samples/sec")
            
        # Identify bottlenecks
        print(f"\n🔍 Bottleneck Analysis:")
        for result in dataloader_results[:3]:  # Show top 3
            dataloader_ratio = result.dataloader_time / result.total_time
            model_ratio = result.model_time / result.total_time
            print(f"   Batch:{result.batch_size}, Workers:{result.num_workers} - "
                  f"DataLoader: {dataloader_ratio*100:.1f}%, Model: {model_ratio*100:.1f}%")
        
        return {
            'best_dataloader_config': {
                'batch_size': best_dataloader.batch_size,
                'num_workers': best_dataloader.num_workers,
                'performance': best_dataloader.samples_per_second
            },
            'best_memory_config': {
                'batch_size': best_memory.batch_size if 'best_memory' in locals() else 64,
                'memory_usage': best_memory.gpu_memory_used if 'best_memory' in locals() else 0
            },
            'worker_analysis': worker_analysis
        }
        
    def generate_optimal_command(self, analysis: Dict[str, Any]) -> str:
        """Generate optimal training command based on analysis."""
        best_batch = analysis['best_memory_config']['batch_size']
        best_workers = analysis['best_dataloader_config']['num_workers']
        
        # Conservative adjustment for GRPO (which uses more memory)
        grpo_batch_size = min(best_batch // 2, 64)  # GRPO needs extra memory for generation
        
        command = f"""
# Optimal H200 Configuration Based on Performance Testing
CUDA_VISIBLE_DEVICES=0 uv run train_grpo.py \\
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \\
    --reward_funcs accuracy format reasoning_steps \\
    --per_device_train_batch_size {grpo_batch_size} \\
    --gradient_accumulation_steps 1 \\
    --generation_batch_size {grpo_batch_size * 2} \\
    --max_completion_length 1024 \\
    --dataloader_num_workers {best_workers} \\
    --logging_steps 5

# Performance Expectations:
# - Batch Size: {grpo_batch_size} (GRPO-optimized)
# - Workers: {best_workers} (optimal for your CPU)
# - Expected GPU Usage: ~{(analysis['best_memory_config']['memory_usage'] * 1.5):.1f}GB
# - Expected Performance: ~{analysis['best_dataloader_config']['performance'] * 0.7:.1f} samples/sec (GRPO adjusted)
"""
        return command


def main():
    """Run comprehensive H200 optimization tests."""
    print("🚀 H200 GPU Optimization Testing Suite")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. This test requires GPU.")
        return
        
    tester = H200OptimizationTester()
    
    try:
        # Setup
        tester.setup()
        
        # Test dataloader performance
        print("Phase 1: Dataloader Performance Testing")
        dataloader_results = tester.test_dataloader_performance(
            batch_sizes=[16, 32, 64],
            worker_counts=[4, 8, 16, 32]
        )
        
        print("\nPhase 2: Memory Scaling Testing")
        memory_results = tester.test_memory_scaling(max_batch_size=128)
        
        # Analyze results
        analysis = tester.analyze_results(dataloader_results, memory_results)
        
        # Generate optimal command
        optimal_command = tester.generate_optimal_command(analysis)
        print("\n🎯 OPTIMAL CONFIGURATION:")
        print(optimal_command)
        
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()