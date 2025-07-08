#!/usr/bin/env python3
"""
Test with heavier processing to simulate real GRPO conversation formatting workload.
"""

import time
import psutil
import torch
import os
import json
import re
from torch.utils.data import DataLoader, Dataset
import numpy as np


class HeavyGRPODataset(Dataset):
    """Dataset with heavy processing to simulate real GRPO workload."""
    
    def __init__(self, size: int = 1000):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate heavy conversation processing
        
        # 1. Generate complex conversation data
        problem = f"Find the derivative of f(x) = x^{idx % 10 + 1} + {idx % 5 + 1}x + {idx % 3 + 1}"
        
        # 2. Simulate complex solution generation
        power = idx % 10 + 1
        coeff = idx % 5 + 1
        constant = idx % 3 + 1
        
        solution = f"""<think>
I need to find the derivative of f(x) = x^{power} + {coeff}x + {constant}.

Using the power rule:
- The derivative of x^{power} is {power}x^{power-1}
- The derivative of {coeff}x is {coeff}
- The derivative of the constant {constant} is 0

So f'(x) = {power}x^{power-1} + {coeff}
</think>

<answer>f'(x) = {power}x^{power-1} + {coeff}</answer>"""
        
        # 3. Simulate heavy tokenization and conversation formatting
        conversation = [
            {"role": "system", "content": "You are a helpful math tutor that shows step-by-step solutions."},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": solution}
        ]
        
        # 4. Simulate tokenization (heavy string processing)
        formatted_conversation = ""
        for msg in conversation:
            formatted_conversation += f"<|{msg['role']}|>\n{msg['content']}\n<|end|>\n"
        
        # 5. Simulate complex tokenization
        tokens = []
        words = formatted_conversation.split()
        for word in words:
            # Simulate subword tokenization
            if len(word) > 6:
                # Split long words
                for i in range(0, len(word), 4):
                    tokens.append(word[i:i+4])
            else:
                tokens.append(word)
        
        # 6. Simulate reward function preprocessing
        has_think = '<think>' in solution
        has_answer = '<answer>' in solution
        has_steps = 'step' in solution.lower() or 'derivative' in solution.lower()
        reasoning_length = len(solution.split()) if has_think else 0
        
        # 7. Simulate format checking (regex processing)
        think_match = re.search(r'<think>(.*?)</think>', solution, re.DOTALL)
        answer_match = re.search(r'<answer>(.*?)</answer>', solution, re.DOTALL)
        
        format_valid = bool(think_match and answer_match)
        
        # 8. Simulate heavy computation
        # This simulates the CPU-intensive work that GRPO does
        complexity_score = 0
        for token in tokens:
            complexity_score += len(token) * 0.1
        
        # 9. Simulate JSON serialization/deserialization (common in GRPO)
        metadata = {
            'problem_id': idx,
            'problem_type': 'derivative',
            'difficulty': idx % 5 + 1,
            'tokens': tokens[:50],  # Limit to avoid too much data
            'conversation': conversation
        }
        
        # Serialize and deserialize to simulate real processing
        json_str = json.dumps(metadata)
        parsed_metadata = json.loads(json_str)
        
        return {
            'problem': problem,
            'solution': solution,
            'formatted_conversation': formatted_conversation,
            'token_count': len(tokens),
            'has_think': has_think,
            'has_answer': has_answer,
            'has_steps': has_steps,
            'reasoning_length': reasoning_length,
            'format_valid': format_valid,
            'complexity_score': complexity_score,
            'metadata': parsed_metadata
        }


def test_heavy_workers():
    """Test worker scaling with heavy processing."""
    print("🧪 Heavy Processing Worker Scaling Test")
    print("Simulating real GRPO conversation formatting workload")
    print("=" * 60)
    
    dataset = HeavyGRPODataset(size=500)  # Smaller dataset due to heavy processing
    batch_size = 16  # Smaller batch due to heavy processing
    
    print(f"Dataset size: {len(dataset)}")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"Batch size: {batch_size}")
    print()
    
    worker_counts = [0, 1, 2, 4, 8, 16, 24, 32]
    
    print("Workers | Samples/sec | CPU% | Time/batch | Efficiency")
    print("-" * 60)
    
    results = []
    
    for num_workers in worker_counts:
        try:
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=True if num_workers > 0 else False,
                persistent_workers=True if num_workers > 0 else False,
                prefetch_factor=2 if num_workers > 0 else None,
                drop_last=True
            )
            
            # Measure performance
            start_time = time.time()
            cpu_measurements = []
            
            samples_processed = 0
            num_batches = 8  # Fewer batches due to heavy processing
            
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                batch_start = time.time()
                
                # Simulate heavy batch processing (like GRPO reward computation)
                batch_size_actual = len(batch['problem'])
                samples_processed += batch_size_actual
                
                # Simulate complex reward function computations
                total_rewards = []
                for j in range(batch_size_actual):
                    # Extract batch data
                    problem = batch['problem'][j]
                    solution = batch['solution'][j]
                    token_count = batch['token_count'][j].item()
                    complexity = batch['complexity_score'][j].item()
                    
                    # Simulate accuracy reward (string matching)
                    accuracy_score = 1.0 if 'derivative' in solution and 'x^' in solution else 0.5
                    
                    # Simulate format reward (regex processing)
                    format_score = 1.0 if batch['format_valid'][j] else 0.0
                    
                    # Simulate reasoning reward (length-based)
                    reasoning_score = min(batch['reasoning_length'][j].item() / 50, 1.0)
                    
                    # Simulate cosine similarity (expensive computation)
                    cosine_score = np.cos(complexity * 0.1) * 0.5 + 0.5
                    
                    # Simulate repetition penalty (n-gram analysis)
                    words = solution.split()
                    repetition_penalty = 0.0
                    if len(words) > 3:
                        # Check for 3-grams
                        for k in range(len(words) - 2):
                            trigram = ' '.join(words[k:k+3])
                            if solution.count(trigram) > 1:
                                repetition_penalty += 0.1
                    
                    # Combine rewards (like GRPO does)
                    total_reward = (accuracy_score + format_score + reasoning_score + 
                                   cosine_score - repetition_penalty) / 4.0
                    total_rewards.append(total_reward)
                
                # Simulate storing results
                avg_reward = np.mean(total_rewards)
                _ = avg_reward  # Store result
                
                cpu_measurements.append(psutil.cpu_percent(interval=None))
            
            end_time = time.time()
            
            total_time = end_time - start_time
            samples_per_sec = samples_processed / total_time if total_time > 0 else 0
            avg_cpu = np.mean(cpu_measurements) if cpu_measurements else 0
            time_per_batch = total_time / num_batches * 1000  # ms
            efficiency = samples_per_sec / max(num_workers, 1)  # samples per worker
            
            results.append({
                'workers': num_workers,
                'samples_per_sec': samples_per_sec,
                'cpu_percent': avg_cpu,
                'time_per_batch_ms': time_per_batch,
                'efficiency': efficiency
            })
            
            print(f"  {num_workers:2d}    | {samples_per_sec:8.1f} | {avg_cpu:4.1f} | "
                  f"{time_per_batch:8.1f}ms | {efficiency:8.2f}")
            
        except Exception as e:
            print(f"  {num_workers:2d}    | ERROR: {str(e)[:40]}")
        
        # Clean up between tests
        time.sleep(0.5)
    
    return results


def analyze_heavy_results(results):
    """Analyze heavy processing results."""
    print(f"\n📊 Heavy Processing Analysis")
    print("=" * 60)
    
    if len(results) < 2:
        print("❌ Not enough data for analysis")
        return
    
    # Find configurations
    single_thread = next((r for r in results if r['workers'] == 0), None)
    current_default = next((r for r in results if r['workers'] == 4), None)
    best = max(results, key=lambda x: x['samples_per_sec'])
    
    print(f"Single-threaded (0 workers): {single_thread['samples_per_sec']:.1f} samples/sec" if single_thread else "N/A")
    print(f"Current default (4 workers): {current_default['samples_per_sec']:.1f} samples/sec" if current_default else "N/A")
    print(f"Best ({best['workers']} workers): {best['samples_per_sec']:.1f} samples/sec")
    
    # Calculate speedups
    if single_thread and current_default:
        speedup_vs_single = best['samples_per_sec'] / single_thread['samples_per_sec']
        speedup_vs_current = best['samples_per_sec'] / current_default['samples_per_sec']
        
        print(f"\nSpeedup vs single-threaded: {speedup_vs_single:.1f}x")
        print(f"Speedup vs current default: {speedup_vs_current:.1f}x")
        
        print(f"\n🎯 Heavy Processing Verdict:")
        if speedup_vs_current > 2.0:
            print(f"🚨 MAJOR BOTTLENECK with heavy processing!")
            print(f"   Use {best['workers']} workers for {speedup_vs_current:.1f}x speedup")
        elif speedup_vs_current > 1.3:
            print(f"⚠️  Moderate bottleneck with heavy processing")
            print(f"   Recommended: {best['workers']} workers")
        else:
            print(f"✅ Current config adequate for heavy processing")
        
        if speedup_vs_single > 1.2:
            print(f"✅ Multiprocessing beneficial for heavy workloads")
            print(f"   {best['workers']} workers > single-threaded by {speedup_vs_single:.1f}x")
        else:
            print(f"⚠️  Multiprocessing overhead still high even with heavy processing")
    
    # Efficiency analysis
    print(f"\n📈 Efficiency Analysis (samples/sec per worker):")
    for r in sorted(results, key=lambda x: x['workers']):
        if r['workers'] > 0:
            print(f"   {r['workers']:2d} workers: {r['efficiency']:6.2f} samples/sec/worker")


def main():
    """Run heavy processing test."""
    print("🔍 H200 Heavy Processing Dataloader Test")
    print("Testing with realistic GRPO conversation processing workload")
    print("=" * 60)
    
    try:
        results = test_heavy_workers()
        analyze_heavy_results(results)
        
        print(f"\n🎯 FINAL H200 RECOMMENDATION:")
        print("=" * 60)
        
        if results:
            best = max(results, key=lambda x: x['samples_per_sec'])
            current = next((r for r in results if r['workers'] == 4), None)
            
            if current:
                speedup = best['samples_per_sec'] / current['samples_per_sec']
                
                print(f"""
Based on heavy processing simulation:

Optimal Configuration:
- Workers: {best['workers']} (vs current 4)
- Expected speedup: {speedup:.1f}x
- Batch size: 64-96 (H200 can handle larger)

CUDA_VISIBLE_DEVICES=0 uv run train_grpo.py \\
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \\
    --reward_funcs accuracy format reasoning_steps \\
    --per_device_train_batch_size 64 \\
    --dataloader_num_workers {best['workers']} \\
    --generation_batch_size 128 \\
    --max_completion_length 1024 \\
    --logging_steps 5
""")
        
    except Exception as e:
        print(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()