#!/usr/bin/env python3
"""
Test inference script to sample from the trained model using a specified checkpoint.
"""

import os
import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import the correct system prompt used during training
from src.data.dataset import SYSTEM_PROMPT

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test inference with a trained model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to the model checkpoint directory")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--sample_idx", type=int, default=0,
                       help="Which test sample to use (default: 0)")
    return parser.parse_args()

def load_checkpoint(checkpoint_path):
    """Load the specified checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        local_files_only=True,
        trust_remote_code=True
    )
    
    # Load model with consistent device placement
    # Use single GPU for inference to avoid device mismatch issues
    device_map = None
    if torch.cuda.is_available():
        device_map = "cuda:0"  # Force entire model on single GPU for inference
        print("Using single GPU (cuda:0) for inference - avoids device mismatch")
    
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
        trust_remote_code=True
    )
    
    return model, tokenizer

def get_test_sample(sample_idx=0):
    """Get a sample from the test dataset."""
    print("Loading dataset...")
    dataset = load_dataset("AI-MO/NuminaMath-TIR", split="test")
    
    # Get specified sample
    sample = dataset[sample_idx]
    print(f"Using test sample #{sample_idx}")
    return sample

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """Generate response from the model."""
    
    # Format the prompt as a conversation with the SAME system prompt used in training
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
    
    print(f"Formatted prompt:\n{formatted_prompt}")
    print("="*80)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response (only the new tokens)
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response

def main():
    """Main function."""
    args = parse_args()
    
    print("=== INFERENCE TEST WITH CONFIGURABLE CHECKPOINT ===")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"System prompt being used:\n{SYSTEM_PROMPT}")
    print("="*80)
    
    print("Loading checkpoint...")
    model, tokenizer = load_checkpoint(args.checkpoint_path)
    
    print("Getting test sample...")
    sample = get_test_sample(args.sample_idx)
    
    print("Sample problem:")
    print(f"Problem: {sample['problem']}")
    print(f"Ground truth solution: {sample['solution']}")
    print("="*80)
    
    print(f"Generating response (max_length={args.max_length}, temp={args.temperature}, top_p={args.top_p})...")
    response = generate_response(
        model, tokenizer, sample['problem'], 
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    print("Generated response:")
    print(response)
    print("="*80)
    
    # Check if it follows the expected format
    import re
    think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
    
    if think_match and answer_match:
        print("✅ Response follows expected format!")
        print(f"Think section: {think_match.group(1).strip()[:200]}...")
        print(f"Answer section: {answer_match.group(1).strip()}")
    else:
        print("❌ Response does NOT follow expected <think>...</think><answer>...</answer> format")
    
    # Check for reasoning indicators
    reasoning_pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    reasoning_matches = len(re.findall(reasoning_pattern, response, re.MULTILINE))
    print(f"🧠 Reasoning indicators found: {reasoning_matches}")
    
    # Check if it's garbage like the trained model
    if "-t-t-t" in response or response.count("-t") > 10:
        print("💀 Model generating garbage patterns!")
    else:
        print("✅ Model generating coherent text (not garbage)")
        
    print("="*80)

if __name__ == "__main__":
    main()