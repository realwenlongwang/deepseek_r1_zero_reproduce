#!/usr/bin/env python3
"""
Quick dataloader bottleneck test for H200 optimization.
Tests dataloader performance without loading full model.
"""

import os
import sys
import time
import psutil
import torch
from typing import List, Tuple
import numpy as np
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.dataset import create_dataset


def measure_dataloader_performance(dataset, batch_size: int, num_workers: int, num_batches: int = 20) -> Tuple[float, float, float]:
    """
    Measure dataloader performance: samples/sec, CPU usage, and time per batch.
    
    Returns:
        (samples_per_second, avg_cpu_percent, avg_batch_time)
    """
    
    # Create dataloader
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
        if i >= 2:  # 2 warm-up batches
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
        
        # Process batch (simulate tokenization work)
        if isinstance(batch, dict):
            batch_size_actual = len(batch.get('problem', []))
            # Simulate processing the conversation format
            for problem in batch.get('problem', []):
                _ = len(str(problem))  # Simulate string processing
        else:
            batch_size_actual = len(batch)
        
        batch_end = time.time()
        
        total_samples += batch_size_actual
        batch_times.append(batch_end - batch_start)
        cpu_measurements.append(psutil.cpu_percent(interval=None))
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    samples_per_second = total_samples / total_time if total_time > 0 else 0
    avg_cpu_percent = np.mean(cpu_measurements) if cpu_measurements else 0
    avg_batch_time = np.mean(batch_times) if batch_times else 0
    
    return samples_per_second, avg_cpu_percent, avg_batch_time


def test_worker_scaling():
    """Test how dataloader performance scales with worker count."""
    print("🧪 Testing Dataloader Worker Scaling on H200")
    print("=" * 60)
    
    # Load dataset
    print("Loading dataset...")
    dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Test configurations
    batch_size = 32  # Fixed batch size for worker testing
    worker_counts = [1, 2, 4, 8, 16, 32, 48]  # Test up to 48 workers
    
    results = []
    
    print(f"\nTesting with batch_size={batch_size}:")
    print("Workers | Samples/sec | CPU Usage | Batch Time | Efficiency")
    print("-" * 60)
    
    baseline_performance = None
    
    for num_workers in worker_counts:
        try:
            # Measure performance
            start_time = time.time()
            samples_per_sec, cpu_usage, batch_time = measure_dataloader_performance(
                dataset, batch_size, num_workers, num_batches=15
            )
            test_time = time.time() - start_time
            
            # Calculate efficiency (performance per worker)
            efficiency = samples_per_sec / max(num_workers, 1)
            
            # Store baseline for comparison
            if num_workers == 4:  # Current default
                baseline_performance = samples_per_sec
            
            results.append({
                'workers': num_workers,
                'samples_per_sec': samples_per_sec,
                'cpu_usage': cpu_usage,
                'batch_time': batch_time,
                'efficiency': efficiency,
                'test_time': test_time
            })
            
            # Format output
            improvement = ""
            if baseline_performance and num_workers != 4:
                speedup = samples_per_sec / baseline_performance
                improvement = f" ({speedup:.1f}x)"
            
            print(f"  {num_workers:2d}    | {samples_per_sec:9.1f}{improvement} | "
                  f"{cpu_usage:7.1f}% | {batch_time*1000:8.1f}ms | {efficiency:8.1f}")
            
        except Exception as e:
            print(f"  {num_workers:2d}    | ERROR: {e}")
            
        # Clean up
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        time.sleep(0.5)
    
    return results


def test_batch_size_scaling():
    """Test how performance scales with batch size."""
    print("\n🧪 Testing Batch Size Scaling")
    print("=" * 60)
    
    # Load dataset
    dataset = create_dataset("AI-MO/NuminaMath-TIR", split="train")
    
    # Test configurations
    batch_sizes = [8, 16, 32, 64, 96, 128]
    num_workers = 16  # Fixed worker count for batch size testing
    
    results = []
    
    print(f"Testing with num_workers={num_workers}:")
    print("Batch Size | Samples/sec | CPU Usage | Batch Time | Memory Impact")
    print("-" * 65)
    
    for batch_size in batch_sizes:
        try:
            # Measure performance
            start_time = time.time()
            samples_per_sec, cpu_usage, batch_time = measure_dataloader_performance(
                dataset, batch_size, num_workers, num_batches=10
            )
            
            # Estimate memory impact (very rough)
            memory_impact = batch_size * 0.1  # Rough estimate: 100MB per sample
            
            results.append({
                'batch_size': batch_size,
                'samples_per_sec': samples_per_sec,
                'cpu_usage': cpu_usage,
                'batch_time': batch_time,
                'memory_impact': memory_impact
            })
            
            print(f"    {batch_size:2d}     | {samples_per_sec:9.1f} | "
                  f"{cpu_usage:7.1f}% | {batch_time*1000:8.1f}ms | ~{memory_impact:5.1f}GB")
            
        except Exception as e:
            print(f"    {batch_size:2d}     | ERROR: {e}")
            
        # Clean up
        time.sleep(0.5)
    
    return results


def analyze_bottlenecks(worker_results: List[dict], batch_results: List[dict]):
    """Analyze results to identify bottlenecks and provide recommendations."""
    print("\n📊 Bottleneck Analysis")
    print("=" * 60)
    
    # Worker analysis
    if worker_results:
        best_worker = max(worker_results, key=lambda x: x['samples_per_sec'])
        current_config = next((r for r in worker_results if r['workers'] == 4), None)
        
        print(f"🎯 Worker Analysis:")
        print(f"   Current config (4 workers): {current_config['samples_per_sec']:.1f} samples/sec")
        print(f"   Best config ({best_worker['workers']} workers): {best_worker['samples_per_sec']:.1f} samples/sec")
        
        if current_config:
            speedup = best_worker['samples_per_sec'] / current_config['samples_per_sec']
            print(f"   Potential speedup: {speedup:.1f}x")
            
            if speedup > 1.5:
                print(f"   🚨 DATALOADER IS A BOTTLENECK! Use {best_worker['workers']} workers")
            else:
                print(f"   ✅ Dataloader is reasonably optimized")
    
    # Batch size analysis
    if batch_results:
        best_batch = max(batch_results, key=lambda x: x['samples_per_sec'])
        current_batch = next((r for r in batch_results if r['batch_size'] == 16), None)
        
        print(f"\n💾 Batch Size Analysis:")
        if current_batch:
            print(f"   Current config (16 batch): {current_batch['samples_per_sec']:.1f} samples/sec")
        print(f"   Best config ({best_batch['batch_size']} batch): {best_batch['samples_per_sec']:.1f} samples/sec")
        
        if current_batch:
            batch_speedup = best_batch['samples_per_sec'] / current_batch['samples_per_sec']
            print(f"   Potential speedup: {batch_speedup:.1f}x")
    
    # CPU utilization analysis
    high_cpu_results = [r for r in worker_results if r['cpu_usage'] > 80]
    if high_cpu_results:
        print(f"\n⚠️  High CPU Usage Warning:")
        print(f"   {len(high_cpu_results)} configurations exceeded 80% CPU")
        print(f"   Consider reducing workers if CPU becomes bottleneck")
    
    # Generate recommendation
    print(f"\n🎯 H200 Optimization Recommendations:")
    
    if worker_results and batch_results:
        recommended_workers = best_worker['workers']
        recommended_batch = min(best_batch['batch_size'], 64)  # Conservative for GRPO
        
        print(f"""
Recommended Configuration:
- Workers: {recommended_workers} (vs current 4)
- Batch Size: {recommended_batch} (vs current 16)
- Expected Performance Gain: {speedup:.1f}x from workers, {batch_speedup:.1f}x from batch size

Command to test:
CUDA_VISIBLE_DEVICES=0 uv run train_grpo.py \\
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \\
    --reward_funcs accuracy format reasoning_steps \\
    --per_device_train_batch_size {recommended_batch} \\
    --dataloader_num_workers {recommended_workers} \\
    --logging_steps 5
""")


def main():
    """Run dataloader bottleneck analysis."""
    print("🔍 H200 Dataloader Bottleneck Analysis")
    print("This test measures ONLY dataloader performance (no model loading)")
    print("=" * 60)
    
    try:
        # Test worker scaling
        worker_results = test_worker_scaling()
        
        # Test batch size scaling  
        batch_results = test_batch_size_scaling()
        
        # Analyze results
        analyze_bottlenecks(worker_results, batch_results)
        
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()