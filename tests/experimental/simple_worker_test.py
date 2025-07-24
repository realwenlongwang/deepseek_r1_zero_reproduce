#!/usr/bin/env python3
"""
Simple worker scaling test to verify H200 dataloader bottleneck hypothesis.
"""

import time
import psutil
import torch
import os
from torch.utils.data import DataLoader, Dataset
import numpy as np


class SimpleGRPODataset(Dataset):
    """Simple dataset that simulates GRPO workload."""
    
    def __init__(self, size: int = 2000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate conversation processing (CPU-intensive)
        problem = f"Solve this equation: {idx} * 2 + 5 = ?"
        solution = f"<think>I need to calculate {idx} * 2 + 5 = {idx * 2 + 5}</think><answer>{idx * 2 + 5}</answer>"
        
        # Simulate tokenization (list of strings)
        tokens = solution.split()
        
        # Simulate some CPU work
        processed_tokens = [token.lower() for token in tokens]
        
        return {
            'problem': problem,
            'solution': solution,
            'token_count': len(tokens),
            'has_think': '<think>' in solution,
            'has_answer': '<answer>' in solution
        }


def test_workers_simple():
    """Simple test of worker scaling."""
    print("🧪 Simple Worker Scaling Test")
    print("=" * 50)
    
    dataset = SimpleGRPODataset(size=1000)
    batch_size = 32
    
    print(f"Dataset size: {len(dataset)}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Batch size: {batch_size}")
    print()
    
    worker_counts = [0, 1, 4, 8, 16, 32]
    
    print("Workers | Samples/sec | CPU% | Time/batch")
    print("-" * 45)
    
    results = []
    
    for num_workers in worker_counts:
        try:
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=False,  # Simplified
                drop_last=True
            )
            
            # Measure performance
            start_time = time.time()
            cpu_before = psutil.cpu_percent()
            
            samples_processed = 0
            num_batches = 10  # Test fewer batches for speed
            
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                # Simulate processing the batch
                batch_size_actual = len(batch['problem'])
                samples_processed += batch_size_actual
                
                # Simulate some computation on the batch
                for j in range(batch_size_actual):
                    problem = batch['problem'][j]
                    solution = batch['solution'][j]
                    token_count = batch['token_count'][j].item()  # Convert tensor to int
                    
                    # Simulate reward calculation
                    accuracy = 1.0 if 'answer' in solution else 0.0
                    format_score = 1.0 if batch['has_think'][j] and batch['has_answer'][j] else 0.0
                    length_penalty = max(0, 1.0 - token_count / 100)
                    
                    total_reward = accuracy * format_score * length_penalty
                    # Simulate storing result
                    _ = total_reward
            
            end_time = time.time()
            cpu_after = psutil.cpu_percent()
            
            total_time = end_time - start_time
            samples_per_sec = samples_processed / total_time if total_time > 0 else 0
            avg_cpu = (cpu_before + cpu_after) / 2
            time_per_batch = total_time / num_batches * 1000  # ms
            
            results.append({
                'workers': num_workers,
                'samples_per_sec': samples_per_sec,
                'cpu_percent': avg_cpu,
                'time_per_batch_ms': time_per_batch
            })
            
            print(f"  {num_workers:2d}    | {samples_per_sec:8.1f} | {avg_cpu:4.1f} | {time_per_batch:8.1f}ms")
            
        except Exception as e:
            print(f"  {num_workers:2d}    | ERROR: {str(e)[:30]}")
    
    return results


def analyze_simple_results(results):
    """Analyze worker scaling results."""
    print(f"\n📊 Analysis")
    print("=" * 50)
    
    if len(results) < 2:
        print("❌ Not enough data for analysis")
        return
    
    # Find baseline (4 workers, current default)
    baseline = next((r for r in results if r['workers'] == 4), results[0])
    best = max(results, key=lambda x: x['samples_per_sec'])
    
    print(f"Baseline (4 workers): {baseline['samples_per_sec']:.1f} samples/sec")
    print(f"Best ({best['workers']} workers): {best['samples_per_sec']:.1f} samples/sec")
    
    speedup = best['samples_per_sec'] / baseline['samples_per_sec']
    print(f"Potential speedup: {speedup:.1f}x")
    
    print(f"\n🎯 Verdict:")
    if speedup > 2.0:
        print(f"🚨 DATALOADER IS A MAJOR BOTTLENECK!")
        print(f"   Increasing workers from 4 to {best['workers']} gives {speedup:.1f}x speedup")
    elif speedup > 1.5:
        print(f"⚠️  Dataloader is a moderate bottleneck")
        print(f"   Optimization recommended: use {best['workers']} workers")
    else:
        print(f"✅ Dataloader is reasonably optimized")
        print(f"   Current 4 workers are adequate")
    
    # Generate recommendation
    print(f"\n🔧 H200 Recommendation:")
    optimal_workers = best['workers']
    
    print(f"""
For H200 with Qwen2.5-1.5B:
  
CUDA_VISIBLE_DEVICES=0 uv run train_grpo.py \\
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \\
    --reward_funcs accuracy format reasoning_steps \\
    --per_device_train_batch_size 64 \\
    --dataloader_num_workers {optimal_workers} \\
    --generation_batch_size 128 \\
    --logging_steps 5

Expected improvement: {speedup:.1f}x faster data loading
""")


def main():
    """Run simple worker test."""
    print("🔍 H200 Dataloader Bottleneck Test")
    print("Quick test to verify worker scaling hypothesis")
    print("=" * 50)
    
    try:
        results = test_workers_simple()
        analyze_simple_results(results)
        
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()