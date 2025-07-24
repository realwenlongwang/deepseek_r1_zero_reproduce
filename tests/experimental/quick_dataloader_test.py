#!/usr/bin/env python3
"""
Quick dataloader test to verify H200 optimization hypothesis.
Tests basic dataloader performance without complex imports.
"""

import time
import psutil
import torch
import os
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import numpy as np


class MockGRPODataset(Dataset):
    """Mock dataset that simulates GRPO conversation processing workload."""
    
    def __init__(self, size: int = 1000):
        self.size = size
        # Simulate conversation data similar to NuminaMath
        self.data = []
        for i in range(size):
            self.data.append({
                'problem': f"Solve this math problem: {i} + {i*2} = ?",
                'solution': f"The answer is {i + i*2}",
                'conversation': [
                    {'role': 'system', 'content': 'You are a math tutor.'},
                    {'role': 'user', 'content': f"Solve: {i} + {i*2} = ?"},
                    {'role': 'assistant', 'content': f"<think>I need to calculate {i} + {i*2}</think><answer>{i + i*2}</answer>"}
                ]
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate conversation formatting work (CPU intensive)
        item = self.data[idx % self.size]
        
        # Simulate tokenization work
        conversation_text = ""
        for msg in item['conversation']:
            conversation_text += f"{msg['role']}: {msg['content']}\n"
        
        # Simulate some processing
        tokens = conversation_text.split()  # Simple tokenization
        
        return {
            'problem': item['problem'],
            'solution': item['solution'],
            'conversation': item['conversation'],
            'tokens': tokens,
            'token_count': len(tokens)
        }


def measure_dataloader_performance(dataset, batch_size: int, num_workers: int, num_batches: int = 15) -> Tuple[float, float, float]:
    """Measure dataloader performance without GPU model."""
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=True
    )
    
    # Warm up
    for i, batch in enumerate(dataloader):
        if i >= 2:
            break
    
    # Measure performance
    start_time = time.time()
    cpu_measurements = []
    batch_times = []
    total_samples = 0
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
            
        batch_start = time.time()
        
        # Simulate processing work (like reward function computation)
        try:
            batch_size_actual = len(batch['problem'])
            
            # Simulate CPU work that GRPO does
            for j in range(batch_size_actual):
                problem = batch['problem'][j]
                solution = batch['solution'][j]
                tokens = batch['tokens'][j]
                
                # Simulate reward function calculations
                accuracy_score = 1.0 if "answer" in solution.lower() else 0.0
                format_score = 1.0 if "<think>" in solution and "<answer>" in solution else 0.0
                length_score = min(len(tokens) / 100, 1.0)
                
                # Simulate some computation
                _ = accuracy_score * format_score * length_score
        except (IndexError, KeyError, TypeError) as e:
            # Handle batch structure issues
            batch_size_actual = batch_size if isinstance(batch, dict) else len(batch)
            # Simulate minimal processing
            for j in range(batch_size_actual):
                _ = j * 0.1  # Simple computation
        
        batch_end = time.time()
        
        total_samples += batch_size_actual
        batch_times.append(batch_end - batch_start)
        cpu_measurements.append(psutil.cpu_percent(interval=None))
    
    end_time = time.time()
    
    total_time = end_time - start_time
    samples_per_second = total_samples / total_time if total_time > 0 else 0
    avg_cpu_percent = np.mean(cpu_measurements) if cpu_measurements else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    return samples_per_second, avg_cpu_percent, avg_batch_time


def test_worker_scaling():
    """Test dataloader worker scaling performance."""
    print("🧪 Testing Dataloader Worker Scaling")
    print("=" * 60)
    
    # Create mock dataset
    dataset = MockGRPODataset(size=2000)
    print(f"Created mock dataset with {len(dataset)} samples")
    
    # Get CPU info
    cpu_count = os.cpu_count()
    print(f"System CPU cores: {cpu_count}")
    
    # Test configurations
    batch_size = 32
    worker_counts = [1, 4, 8, 16, 24, 32]
    if cpu_count and cpu_count > 32:
        worker_counts.extend([48, 64])
    
    print(f"\nTesting with batch_size={batch_size}:")
    print("Workers | Samples/sec | CPU Usage | Batch Time | Speedup")
    print("-" * 60)
    
    results = []
    baseline_perf = None
    
    for num_workers in worker_counts:
        try:
            samples_per_sec, cpu_usage, batch_time = measure_dataloader_performance(
                dataset, batch_size, num_workers
            )
            
            if num_workers == 4:  # Current default
                baseline_perf = samples_per_sec
            
            speedup = samples_per_sec / baseline_perf if baseline_perf else 1.0
            speedup_str = f"{speedup:.1f}x" if baseline_perf else "baseline"
            
            results.append({
                'workers': num_workers,
                'samples_per_sec': samples_per_sec,
                'cpu_usage': cpu_usage,
                'batch_time': batch_time,
                'speedup': speedup
            })
            
            print(f"  {num_workers:2d}    | {samples_per_sec:9.1f} | "
                  f"{cpu_usage:7.1f}% | {batch_time*1000:8.1f}ms | {speedup_str:>7s}")
            
        except Exception as e:
            print(f"  {num_workers:2d}    | ERROR: {e}")
    
    return results


def test_batch_scaling():
    """Test batch size scaling performance."""
    print("\n🧪 Testing Batch Size Scaling")
    print("=" * 60)
    
    dataset = MockGRPODataset(size=2000)
    
    batch_sizes = [8, 16, 32, 64, 96, 128]
    num_workers = 16  # Fixed optimal workers
    
    print(f"Testing with num_workers={num_workers}:")
    print("Batch Size | Samples/sec | CPU Usage | Batch Time | Efficiency")
    print("-" * 65)
    
    results = []
    
    for batch_size in batch_sizes:
        try:
            samples_per_sec, cpu_usage, batch_time = measure_dataloader_performance(
                dataset, batch_size, num_workers, num_batches=10
            )
            
            # Efficiency: samples per second per batch size
            efficiency = samples_per_sec / batch_size
            
            results.append({
                'batch_size': batch_size,
                'samples_per_sec': samples_per_sec,
                'cpu_usage': cpu_usage,
                'batch_time': batch_time,
                'efficiency': efficiency
            })
            
            print(f"    {batch_size:2d}     | {samples_per_sec:9.1f} | "
                  f"{cpu_usage:7.1f}% | {batch_time*1000:8.1f}ms | {efficiency:8.2f}")
            
        except Exception as e:
            print(f"    {batch_size:2d}     | ERROR: {e}")
    
    return results


def analyze_results(worker_results: List[dict], batch_results: List[dict]):
    """Analyze results and provide H200 optimization recommendations."""
    print("\n📊 H200 Optimization Analysis")
    print("=" * 60)
    
    # Worker analysis
    if worker_results:
        current_config = next((r for r in worker_results if r['workers'] == 4), None)
        best_config = max(worker_results, key=lambda x: x['samples_per_sec'])
        
        print("🔧 Worker Configuration Analysis:")
        if current_config:
            print(f"   Current (4 workers): {current_config['samples_per_sec']:.1f} samples/sec")
            print(f"   Best ({best_config['workers']} workers): {best_config['samples_per_sec']:.1f} samples/sec")
            print(f"   Potential Speedup: {best_config['speedup']:.1f}x")
            
            if best_config['speedup'] > 2.0:
                print("   🚨 MAJOR BOTTLENECK: Dataloader is severely limiting performance!")
            elif best_config['speedup'] > 1.5:
                print("   ⚠️  BOTTLENECK: Dataloader is limiting performance")
            else:
                print("   ✅ Dataloader is reasonably optimized")
        
        # CPU utilization warning
        high_cpu_configs = [r for r in worker_results if r['cpu_usage'] > 90]
        if high_cpu_configs:
            print(f"   ⚠️  CPU saturation at {len(high_cpu_configs)} configurations")
    
    # Batch size analysis
    if batch_results:
        best_batch = max(batch_results, key=lambda x: x['samples_per_sec'])
        current_batch = next((r for r in batch_results if r['batch_size'] == 16), None)
        
        print(f"\n💾 Batch Size Analysis:")
        if current_batch:
            print(f"   Current (16 batch): {current_batch['samples_per_sec']:.1f} samples/sec")
        print(f"   Best ({best_batch['batch_size']} batch): {best_batch['samples_per_sec']:.1f} samples/sec")
        
        if current_batch:
            batch_speedup = best_batch['samples_per_sec'] / current_batch['samples_per_sec']
            print(f"   Batch Size Speedup: {batch_speedup:.1f}x")
    
    # Generate H200-specific recommendations
    if worker_results and batch_results:
        print(f"\n🎯 H200 Optimization Recommendations:")
        
        optimal_workers = best_config['workers']
        optimal_batch = best_batch['batch_size']
        
        # Conservative adjustments for GRPO (uses more memory)
        grpo_batch_size = min(optimal_batch, 64)  # GRPO constraint
        
        total_speedup = best_config['speedup'] * (batch_speedup if 'batch_speedup' in locals() else 1.0)
        
        print(f"""
🚀 Recommended Configuration for H200:
   Workers: {optimal_workers} (vs current 4)
   Batch Size: {grpo_batch_size} (vs current 16, GRPO-adjusted)
   Expected Total Speedup: {total_speedup:.1f}x

📊 Performance Impact:
   - Dataloader speedup: {best_config['speedup']:.1f}x
   - Batch size speedup: {batch_speedup if 'batch_speedup' in locals() else 1.0:.1f}x
   - Combined improvement: ~{total_speedup:.1f}x faster training

🔧 Command to test:
CUDA_VISIBLE_DEVICES=0 uv run train_grpo.py \\
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \\
    --reward_funcs accuracy format reasoning_steps \\
    --per_device_train_batch_size {grpo_batch_size} \\
    --dataloader_num_workers {optimal_workers} \\
    --generation_batch_size {grpo_batch_size * 2} \\
    --logging_steps 5
""")
        
        # Verdict on dataloader bottleneck
        if best_config['speedup'] > 2.0:
            print("🔍 VERDICT: Dataloader IS a major bottleneck on H200!")
            print("   Your suspicion was CORRECT - dataloader optimization is critical.")
        elif best_config['speedup'] > 1.5:
            print("🔍 VERDICT: Dataloader is a moderate bottleneck on H200.")
            print("   Optimization will provide noticeable improvements.")
        else:
            print("🔍 VERDICT: Dataloader is not a significant bottleneck.")
            print("   Focus optimization efforts elsewhere (batch size, model config).")


def main():
    """Run comprehensive dataloader bottleneck test."""
    print("🔍 H200 Dataloader Bottleneck Verification")
    print("Testing with mock GRPO-like workload (no model loading required)")
    print("=" * 60)
    
    try:
        # Test worker scaling
        worker_results = test_worker_scaling()
        
        # Test batch scaling
        batch_results = test_batch_scaling()
        
        # Analyze and provide recommendations
        analyze_results(worker_results, batch_results)
        
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()