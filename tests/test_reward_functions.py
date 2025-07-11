#!/usr/bin/env python3
"""
Comprehensive test script for evaluating reward functions from openr1_rewards.py.

This test:
1. Loads the dataset
2. Loads Qwen2.5/3B-instruct base model
3. Picks three questions from the dataset
4. Generates responses using the base model
5. Evaluates all reward functions on the responses
6. Analyzes reward correctness
"""

import os
import sys
import json
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)

from src.data.dataset import create_dataset, SYSTEM_PROMPT
from src.rewards.openr1_rewards import (
    accuracy_reward,
    format_reward,
    tag_count_reward,
    reasoning_steps_reward,
    len_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    get_soft_overlong_punishment,
    equation_reward
)

@dataclass
class MockScriptArgs:
    """Mock script arguments for reward function configuration."""
    cosine_min_value_wrong: float = -0.5
    cosine_max_value_wrong: float = -0.1
    cosine_min_value_correct: float = 0.8
    cosine_max_value_correct: float = 1.0
    cosine_max_len: int = 1000
    repetition_n_grams: int = 3
    repetition_max_penalty: float = -0.1
    max_completion_len: int = 512
    soft_punish_cache: int = 50

def load_model_and_tokenizer(model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
    """Load the base model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    print(f"Model loaded successfully: {model.config.model_type}")
    print(f"Model parameters: {model.num_parameters():,}")
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_length: int = 512) -> str:
    """Generate response from the model."""
    
    # Format the prompt as a conversation with the system prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize and generate
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode the response (excluding the input prompt)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    return response.strip()

def evaluate_reward_functions(completions: List[List[Dict]], solutions: List[str], problems: List[str], targets: List[str] = None, nums_list: List[List[int]] = None) -> Dict[str, List[float]]:
    """Evaluate all reward functions on the completions."""
    print("\n" + "="*80)
    print("EVALUATING REWARD FUNCTIONS")
    print("="*80)
    
    # Create mock script args for parameterized reward functions
    script_args = MockScriptArgs()
    
    # Initialize reward functions
    reward_functions = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "tag_count": tag_count_reward,
        "reasoning_steps": reasoning_steps_reward,
        "length": len_reward,
        "cosine_scaled": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
        "equation": equation_reward
    }
    
    results = {}
    
    for func_name, func in reward_functions.items():
        print(f"\nEvaluating {func_name}...")
        
        try:
            if func_name == "soft_overlong_punishment":
                # This function expects token IDs instead of text
                # For simplicity, we'll estimate token length
                completion_ids = [
                    [list(range(len(comp[0]["content"])))]  # Mock token IDs based on character length
                    for comp in completions
                ]
                rewards = func(completion_ids)
            elif func_name == "equation":
                # This function needs target and nums parameters
                if targets is not None and nums_list is not None:
                    rewards = func(completions, target=targets, nums=nums_list)
                else:
                    print(f"  Skipping {func_name} - missing targets or nums")
                    results[func_name] = [None] * len(completions)
                    continue
            else:
                rewards = func(completions, solution=solutions)
            
            results[func_name] = rewards
            print(f"  Rewards: {rewards}")
            
        except Exception as e:
            print(f"  Error evaluating {func_name}: {e}")
            results[func_name] = [None] * len(completions)
    
    return results

def analyze_reward_correctness(completions: List[List[Dict]], solutions: List[str], problems: List[str], reward_results: Dict[str, List[float]]):
    """Analyze the correctness of reward functions."""
    print("\n" + "="*80)
    print("REWARD ANALYSIS")
    print("="*80)
    
    for i, (completion, solution, problem) in enumerate(zip(completions, solutions, problems)):
        print(f"\n--- QUESTION {i+1} ---")
        print(f"Problem: {problem[:100]}...")
        print(f"Model Response: {completion[0]['content'][:200]}...")
        print(f"Ground Truth Solution: {solution[:100]}...")
        
        print("\nReward Scores:")
        for func_name, rewards in reward_results.items():
            reward = rewards[i] if i < len(rewards) else None
            print(f"  {func_name:20}: {reward}")
        
        # Analyze specific reward functions
        response_text = completion[0]['content']
        
        print("\nDetailed Analysis:")
        
        # Format analysis
        has_think_tags = '<think>' in response_text and '</think>' in response_text
        has_answer_tags = '<answer>' in response_text and '</answer>' in response_text
        print(f"  Has <think> tags: {has_think_tags}")
        print(f"  Has <answer> tags: {has_answer_tags}")
        print(f"  Format reward correct: {reward_results['format'][i] == 1.0 if has_think_tags and has_answer_tags else reward_results['format'][i] == 0.0}")
        
        # Reasoning steps analysis
        import re
        reasoning_patterns = [
            r"Step \d+:",
            r"^\d+\.",
            r"\n-",
            r"\n\*",
            r"First,",
            r"Second,",
            r"Next,",
            r"Finally,"
        ]
        step_count = sum(len(re.findall(pattern, response_text)) for pattern in reasoning_patterns)
        print(f"  Reasoning steps found: {step_count}")
        print(f"  Reasoning steps reward: {reward_results['reasoning_steps'][i]}")
        
        # Length analysis
        response_length = len(response_text)
        print(f"  Response length: {response_length} characters")
        print(f"  Length reward: {reward_results['length'][i]}")
        
        print("-" * 60)

def main():
    """Main test function."""
    print("="*80)
    print("REWARD FUNCTION EVALUATION TEST")
    print("="*80)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load dataset - using Countdown dataset for better accuracy testing
    print("Loading dataset...")
    dataset = create_dataset(
        dataset_name="Jiayi-Pan/Countdown-Tasks-3to4", 
        split="train"
    )
    print(f"Countdown dataset loaded with {len(dataset)} examples")
    
    # Pick three questions
    print("\nSelecting three questions...")
    selected_indices = [0, 10, 20]  # Select specific indices for consistency
    selected_examples = [dataset[i] for i in selected_indices]
    
    problems = [ex["problem"] for ex in selected_examples]
    solutions = [ex.get("reference_solution", "") for ex in selected_examples]
    
    # Extract target numbers and available numbers for equation reward
    targets = []
    nums_list = []
    for ex in selected_examples:
        # Debug: print the keys available in the dataset
        print(f"Dataset example keys: {list(ex.keys())}")
        targets.append(ex.get("target", ""))
        nums_list.append(ex.get("nums", []))
    
    print(f"Selected {len(problems)} questions")
    for i, problem in enumerate(problems):
        print(f"  Question {i+1}: {problem[:100]}...")
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer("Qwen/Qwen2.5-3B-Instruct")
    
    # Generate responses
    print("\nGenerating responses...")
    responses = []
    for i, problem in enumerate(problems):
        print(f"  Generating response for question {i+1}...")
        response = generate_response(model, tokenizer, problem)
        responses.append(response)
        print(f"    Generated {len(response)} characters")
    
    # Format completions for reward functions
    # Most reward functions expect format: List[List[Dict]]
    completions = [
        [{"content": response, "role": "assistant"}]
        for response in responses
    ]
    
    print("\nGenerated Responses:")
    for i, response in enumerate(responses):
        print(f"\n--- RESPONSE {i+1} ---")
        print(response[:500] + "..." if len(response) > 500 else response)
    
    # Evaluate reward functions
    reward_results = evaluate_reward_functions(completions, solutions, problems, targets, nums_list)
    
    # Analyze reward correctness
    analyze_reward_correctness(completions, solutions, problems, reward_results)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("This test evaluated the following reward functions:")
    for func_name in reward_results.keys():
        print(f"  - {func_name}")
    
    print(f"\nTested on {len(problems)} questions from Countdown dataset")
    print("All reward functions were evaluated and analyzed for correctness.")
    
    print("\nKey findings:")
    print("- Format rewards should be 1.0 for properly formatted responses with <think> and <answer> tags")
    print("- Accuracy rewards compare mathematical correctness against ground truth")
    print("- Reasoning steps rewards encourage step-by-step problem solving")
    print("- Length and cosine rewards balance response length and correctness")
    print("- Repetition penalty discourages repetitive content")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()