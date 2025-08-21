#!/usr/bin/env python3
"""
DeepSeek R1 Zero Comprehensive Inference Script
Supports automatic checkpoint detection, multiple generation modes, and evaluation.
"""
import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any
import random

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.inference import AutoCheckpointLoader, InferenceEngine, ResponseValidator
from src.inference.checkpoint_loader import find_latest_checkpoint, format_messages_for_qwen25
from src.inference.generators import GenerationPresets
from src.inference.interactive import run_interactive_chat

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DeepSeek R1 Zero Inference with automatic checkpoint detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with latest checkpoint
  python inference.py --interactive
  
  # Single prompt with specific checkpoint
  python inference.py --checkpoint saved_models/permanent_checkpoints/checkpoint-20000 \
                      --prompt "Solve: x^2 + 5x + 6 = 0"
  
  # Evaluate on test dataset
  python inference.py --eval_dataset countdown --samples 50 --output results.json
  
  # Batch prompts from file
  python inference.py --batch_file prompts.txt --output responses.json
        """)
    
    # Checkpoint options
    checkpoint_group = parser.add_argument_group('Checkpoint Options')
    checkpoint_group.add_argument('--checkpoint', type=str, 
                                 help='Path to specific checkpoint (auto-detects latest if not specified)')
    checkpoint_group.add_argument('--list_checkpoints', action='store_true',
                                 help='List available checkpoints and exit')
    
    # Mode selection
    mode_group = parser.add_argument_group('Inference Modes')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Start interactive chat mode')
    mode_group.add_argument('--prompt', type=str,
                           help='Single prompt for generation')
    mode_group.add_argument('--batch_file', type=str,
                           help='File containing prompts (one per line)')
    mode_group.add_argument('--eval_dataset', choices=['countdown', 'numina'],
                           help='Evaluate on test dataset')
    
    # Generation parameters
    gen_group = parser.add_argument_group('Generation Parameters')
    gen_group.add_argument('--max_new_tokens', type=int, default=1024,
                          help='Maximum new tokens to generate (default: 1024)')
    gen_group.add_argument('--temperature', type=float, default=0.7,
                          help='Sampling temperature (default: 0.7)')
    gen_group.add_argument('--top_p', type=float, default=0.9,
                          help='Top-p sampling (default: 0.9)')
    gen_group.add_argument('--top_k', type=int, default=50,
                          help='Top-k sampling (default: 50)')
    gen_group.add_argument('--repetition_penalty', type=float, default=1.1,
                          help='Repetition penalty (default: 1.1)')
    gen_group.add_argument('--preset', choices=['creative', 'balanced', 'precise', 'deterministic', 'reasoning'],
                          help='Use predefined generation preset')
    gen_group.add_argument('--no_system_message', action='store_true',
                          help='Disable default system message')
    gen_group.add_argument('--system_message', type=str,
                          help='Custom system message')
    
    # Evaluation options
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument('--samples', type=int, default=10,
                           help='Number of samples to evaluate (default: 10)')
    eval_group.add_argument('--sample_offset', type=int, default=0,
                           help='Starting sample index (default: 0)')
    eval_group.add_argument('--validate_responses', action='store_true',
                           help='Run response validation and scoring')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('--output', type=str,
                             help='Output file for results (JSON format)')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Verbose output')
    output_group.add_argument('--show_metadata', action='store_true',
                             help='Show model metadata')
    
    # Device options
    device_group = parser.add_argument_group('Device Options')
    device_group.add_argument('--device', type=str, default='auto',
                             help='Device for inference (auto, cpu, cuda, cuda:0, etc.)')
    
    return parser.parse_args()


def list_available_checkpoints():
    """List all available checkpoints."""
    print("🔍 Available Checkpoints:")
    print("=" * 80)
    
    base_dirs = ["saved_models", "grpo_output"]
    found_checkpoints = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        
        print(f"\n📁 {base_dir}/")
        
        # Check direct checkpoints
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                try:
                    from src.inference.checkpoint_loader import detect_checkpoint_type
                    checkpoint_type = detect_checkpoint_type(item_path)
                    found_checkpoints.append(item_path)
                    
                    # Get modification time
                    mod_time = os.path.getmtime(item_path)
                    import datetime
                    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"  ✅ {item} ({checkpoint_type}) - {mod_time_str}")
                    
                except:
                    # Check subdirectories
                    if os.path.isdir(item_path):
                        for subitem in os.listdir(item_path):
                            subitem_path = os.path.join(item_path, subitem)
                            if os.path.isdir(subitem_path):
                                try:
                                    checkpoint_type = detect_checkpoint_type(subitem_path)
                                    found_checkpoints.append(subitem_path)
                                    
                                    mod_time = os.path.getmtime(subitem_path)
                                    mod_time_str = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                                    
                                    print(f"    ✅ {item}/{subitem} ({checkpoint_type}) - {mod_time_str}")
                                except:
                                    continue
    
    if found_checkpoints:
        # Find and highlight latest
        latest_checkpoint = find_latest_checkpoint("saved_models")
        if latest_checkpoint:
            print(f"\n🎯 Latest checkpoint: {latest_checkpoint}")
        
        print(f"\n📊 Total checkpoints found: {len(found_checkpoints)}")
    else:
        print("\n❌ No valid checkpoints found.")
        print("Run training first with: uv run train_grpo.py --reward_funcs accuracy format reasoning_steps")


def load_prompts_from_file(filepath: str) -> List[str]:
    """Load prompts from text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {filepath}")
        return prompts
    except Exception as e:
        logger.error(f"Failed to load prompts from {filepath}: {e}")
        return []


def get_test_samples(dataset_type: str, num_samples: int, sample_offset: int = 0) -> List[Dict]:
    """Get test samples from dataset."""
    samples = []
    
    if dataset_type == "countdown":
        try:
            from datasets import load_dataset
            from src.data.dataset import format_countdown_problem
            
            # Load dataset and create test split (same logic as in test_inference.py)
            dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
            
            # Create deterministic test split
            COUNTDOWN_SEED = 42
            TEST_RATIO = 0.1
            
            random.seed(COUNTDOWN_SEED)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            test_size = int(len(dataset) * TEST_RATIO)
            train_split_point = len(dataset) - test_size
            test_indices = indices[train_split_point:]
            
            # Get requested samples
            end_idx = min(sample_offset + num_samples, len(test_indices))
            for i in range(sample_offset, end_idx):
                if i < len(test_indices):
                    actual_idx = test_indices[i]
                    sample = dataset[actual_idx]
                    
                    problem_text = format_countdown_problem(sample["target"], sample["nums"])
                    
                    samples.append({
                        "index": i,
                        "problem": problem_text,
                        "reference": f"Target: {sample['target']}",
                        "metadata": {
                            "target": sample["target"],
                            "nums": sample["nums"],
                            "dataset_type": "countdown"
                        }
                    })
            
        except Exception as e:
            logger.error(f"Failed to load countdown dataset: {e}")
    
    elif dataset_type == "numina":
        try:
            from datasets import load_dataset
            
            dataset = load_dataset("AI-MO/NuminaMath-TIR", split="test")
            
            end_idx = min(sample_offset + num_samples, len(dataset))
            for i in range(sample_offset, end_idx):
                if i < len(dataset):
                    sample = dataset[i]
                    samples.append({
                        "index": i,
                        "problem": sample["problem"],
                        "reference": sample["solution"],
                        "metadata": {
                            "dataset_type": "numina"
                        }
                    })
                    
        except Exception as e:
            logger.error(f"Failed to load numina dataset: {e}")
    
    return samples


def run_single_prompt(engine: InferenceEngine, prompt: str, args) -> Dict[str, Any]:
    """Run inference on single prompt."""
    # Apply generation preset if specified
    gen_kwargs = {}
    if args.preset:
        presets = {
            'creative': GenerationPresets.creative(),
            'balanced': GenerationPresets.balanced(),
            'precise': GenerationPresets.precise(),
            'deterministic': GenerationPresets.deterministic(),
            'reasoning': GenerationPresets.reasoning()
        }
        gen_kwargs.update(presets[args.preset])
    
    # Override with explicit arguments
    gen_kwargs.update({
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'repetition_penalty': args.repetition_penalty,
        'use_system_message': not args.no_system_message,
        'custom_system_message': args.system_message
    })
    
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    try:
        response = engine.generate(prompt, **gen_kwargs)
        
        result = {
            "prompt": prompt,
            "response": response,
            "generation_params": gen_kwargs,
            "success": True
        }
        
        print(f"Response:\n{response}")
        
        # Validate if requested
        if args.validate_responses:
            validator = ResponseValidator()
            validation = validator.validate_response(response)
            result["validation"] = validation
            
            print(f"\n📊 Validation Results:")
            print(f"  Format valid: {'✅' if validation['format_valid'] else '❌'}")
            print(f"  Format score: {validation['format_score']:.2f}")
            print(f"  Quality score: {validation['quality_score']:.2f}")
            if validation['issues']:
                print(f"  Issues: {', '.join(validation['issues'])}")
        
        return result
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return {
            "prompt": prompt,
            "response": f"[Generation failed: {str(e)}]",
            "generation_params": gen_kwargs,
            "success": False,
            "error": str(e)
        }


def run_batch_inference(engine: InferenceEngine, prompts: List[str], args) -> List[Dict[str, Any]]:
    """Run inference on batch of prompts."""
    results = []
    
    print(f"🔄 Processing {len(prompts)} prompts...")
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Prompt {i}/{len(prompts)} ---")
        result = run_single_prompt(engine, prompt, args)
        results.append(result)
        
        if not result["success"]:
            logger.warning(f"Prompt {i} failed")
    
    return results


def evaluate_with_reward_functions(responses: List[str], samples: List[Dict], dataset_type: str) -> Dict[str, Any]:
    """Evaluate responses using format and equation reward functions."""
    # Import reward functions
    from src.rewards.openr1_rewards import format_reward, equation_reward
    
    # Convert responses to TRL format (list of list of dict)
    trl_format_responses = [[{"content": response}] for response in responses]
    
    # Calculate format rewards
    format_rewards = format_reward(trl_format_responses)
    
    # Calculate equation rewards if countdown dataset
    equation_rewards = None
    if dataset_type == "countdown":
        # Extract target and nums from samples metadata
        targets = [str(sample["metadata"]["target"]) for sample in samples]
        nums_list = [sample["metadata"]["nums"] for sample in samples]
        
        equation_rewards = equation_reward(trl_format_responses, targets, nums_list)
    
    return {
        "format_rewards": format_rewards,
        "equation_rewards": equation_rewards,
        "avg_format_reward": sum(format_rewards) / len(format_rewards) if format_rewards else 0.0,
        "avg_equation_reward": sum(equation_rewards) / len(equation_rewards) if equation_rewards else 0.0
    }


def run_dataset_evaluation(engine: InferenceEngine, dataset_type: str, args) -> Dict[str, Any]:
    """Run evaluation on test dataset."""
    samples = get_test_samples(dataset_type, args.samples, args.sample_offset)
    
    if not samples:
        logger.error(f"No samples loaded from {dataset_type} dataset")
        return {"error": "No samples loaded"}
    
    print(f"🔄 Evaluating on {len(samples)} {dataset_type} samples...")
    
    results = []
    responses = []
    for i, sample in enumerate(samples, 1):
        print(f"\n--- Sample {i}/{len(samples)} ---")
        
        result = run_single_prompt(engine, sample["problem"], args)
        result.update({
            "sample_index": sample["index"],
            "reference": sample["reference"],
            "metadata": sample["metadata"]
        })
        
        # Always validate for dataset evaluation
        validator = ResponseValidator()
        problem_type = "countdown" if dataset_type == "countdown" else "math"
        validation = validator.validate_response(
            result["response"], 
            sample["reference"], 
            problem_type
        )
        result["validation"] = validation
        
        # Calculate reward function scores for this individual sample
        from src.rewards.openr1_rewards import format_reward, equation_reward
        
        # Convert single response to TRL format
        trl_response = [{"content": result["response"]}]
        
        # Calculate format reward
        format_score = format_reward([trl_response])[0]
        
        # Calculate equation reward if countdown dataset
        equation_score = None
        if dataset_type == "countdown":
            target = str(sample["metadata"]["target"])
            nums = sample["metadata"]["nums"]
            equation_score = equation_reward([trl_response], [target], [nums])[0]
        
        results.append(result)
        responses.append(result["response"])
        
        # Show validation summary using reward functions
        format_emoji = '✅' if format_score == 1.0 else '❌'
        if equation_score is not None:
            equation_emoji = '✅' if equation_score == 1.0 else '❌'
            print(f"📊 Format: {format_emoji} ({format_score:.1f}) | "
                  f"Equation: {equation_emoji} ({equation_score:.1f})")
        else:
            print(f"📊 Format: {format_emoji} ({format_score:.1f})")
    
    # Calculate batch statistics
    validator = ResponseValidator()
    batch_stats = validator.validate_batch(
        [r["response"] for r in results],
        [r["reference"] for r in results],
        problem_type
    )
    
    # Calculate reward function scores
    reward_evaluation = evaluate_with_reward_functions(responses, samples, dataset_type)
    
    # Print reward function results
    print(f"\n🎯 Reward Function Evaluation:")
    print(f"  Format Reward (avg): {reward_evaluation['avg_format_reward']:.3f}")
    for i, reward in enumerate(reward_evaluation['format_rewards']):
        print(f"    Sample {i+1}: {reward:.3f}")
    
    if reward_evaluation['equation_rewards'] is not None:
        print(f"  Equation Reward (avg): {reward_evaluation['avg_equation_reward']:.3f}")
        for i, reward in enumerate(reward_evaluation['equation_rewards']):
            print(f"    Sample {i+1}: {reward:.3f}")
    
    return {
        "dataset_type": dataset_type,
        "samples_evaluated": len(samples),
        "individual_results": results,
        "batch_statistics": batch_stats,
        "reward_evaluation": reward_evaluation,
        "success": True
    }


def main():
    """Main function."""
    args = parse_args()
    
    # List checkpoints and exit
    if args.list_checkpoints:
        list_available_checkpoints()
        return
    
    # Interactive mode
    if args.interactive:
        run_interactive_chat(
            checkpoint_path=args.checkpoint,
            device=args.device
        )
        return
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint("saved_models")
        if checkpoint_path is None:
            print("❌ No checkpoints found. Use --list_checkpoints to see available options.")
            return
        print(f"🔍 Using latest checkpoint: {checkpoint_path}")
    
    # Load model
    print(f"🔄 Loading checkpoint from path {checkpoint_path}")
    loader = AutoCheckpointLoader(device_map=args.device)
    model, tokenizer, metadata = loader.load_checkpoint(checkpoint_path)
    
    # Show metadata if requested
    if args.show_metadata or args.verbose:
        print("\n📋 Model Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        print()
    
    # Create inference engine
    engine = InferenceEngine(model, tokenizer, args.device)
    
    # Run inference based on mode
    results = None
    
    if args.prompt:
        # Single prompt
        results = run_single_prompt(engine, args.prompt, args)
        
    elif args.batch_file:
        # Batch from file
        prompts = load_prompts_from_file(args.batch_file)
        if prompts:
            results = run_batch_inference(engine, prompts, args)
        else:
            print("❌ No prompts loaded from file.")
            return
            
    elif args.eval_dataset:
        # Dataset evaluation
        results = run_dataset_evaluation(engine, args.eval_dataset, args)
        
    else:
        print("❌ Please specify an inference mode (--prompt, --batch_file, --eval_dataset, or --interactive)")
        return
    
    # Save results if output file specified
    if args.output and results:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {args.output}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()