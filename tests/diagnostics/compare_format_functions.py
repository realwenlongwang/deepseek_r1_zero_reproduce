#!/usr/bin/env python3
"""
Compare the soft_format_reward_func with the flexible format_reward implementation.
"""

import sys
import re
sys.path.append('src')
from src.rewards.openr1_rewards import format_reward

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def compare_format_functions():
    """Compare the two format reward functions."""
    print("🔍 Comparing Format Reward Functions")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Project format (think/answer)",
            "text": "<think>Step by step reasoning</think><answer>Final answer</answer>",
            "description": "What this project expects"
        },
        {
            "name": "Project format with spaces",
            "text": "<think> Step by step reasoning </think><answer> Final answer </answer>",
            "description": "Project format with spaces inside tags"
        },
        {
            "name": "Soft format (reasoning/answer)",
            "text": "<reasoning>Step by step reasoning</reasoning><answer>Final answer</answer>",
            "description": "What soft_format_reward_func expects"
        },
        {
            "name": "Assistant prefix + project format",
            "text": "Assistant: <think>reasoning</think><answer>answer</answer>",
            "description": "Unsloth output with project format"
        },
        {
            "name": "Assistant prefix + soft format",
            "text": "Assistant: <reasoning>reasoning</reasoning><answer>answer</answer>",
            "description": "Unsloth output with soft format"
        },
        {
            "name": "Case variations",
            "text": "<THINK>reasoning</THINK><ANSWER>answer</ANSWER>",
            "description": "Different case"
        },
        {
            "name": "Extra whitespace",
            "text": "  <think>reasoning</think>  <answer>answer</answer>  ",
            "description": "Leading/trailing whitespace"
        },
        {
            "name": "Missing think/reasoning",
            "text": "<answer>Just an answer</answer>",
            "description": "Invalid format"
        }
    ]
    
    print("Comparing function behaviors:")
    print(f"{'Test Case':<35} {'Flexible':<10} {'Soft':<10} {'Notes'}")
    print("-" * 80)
    
    for case in test_cases:
        completion = [[{"content": case["text"]}]]
        
        # Test flexible format_reward
        flexible_reward = format_reward(completion)[0]
        
        # Test soft_format_reward_func
        soft_reward = soft_format_reward_func(completion)[0]
        
        # Determine notes
        notes = ""
        if flexible_reward != soft_reward:
            notes = "DIFFERENT"
        if flexible_reward == 0 and soft_reward == 0:
            notes = "Both fail"
        elif flexible_reward > 0 and soft_reward > 0:
            notes = "Both pass"
            
        print(f"{case['name']:<35} {flexible_reward:<10} {soft_reward:<10} {notes}")
    
    print("\n" + "=" * 60)
    print("📊 Analysis Summary:")
    
    print("\n🔧 Technical Differences:")
    print("1. TAG NAMES:")
    print("   • Flexible: <think>...</think> and <answer>...</answer>")
    print("   • Soft: <reasoning>...</reasoning> and <answer>...</answer>")
    
    print("\n2. PATTERN MATCHING:")
    print("   • Flexible: Uses re.search() - finds tags anywhere in text")
    print("   • Soft: Uses re.match() - requires tags at start of string")
    
    print("\n3. REWARD VALUES:")
    print("   • Flexible: 1.0 for valid format, 0.0 for invalid")
    print("   • Soft: 0.5 for valid format, 0.0 for invalid")
    
    print("\n4. FLEXIBILITY:")
    print("   • Flexible: Handles case variations, whitespace, Assistant: prefix")
    print("   • Soft: Strict matching, no case tolerance, no prefix handling")
    
    print("\n5. CONTENT VALIDATION:")
    print("   • Flexible: Checks for non-empty content in tags")
    print("   • Soft: No content validation (empty tags would pass)")
    
    print("\n🎯 Key Takeaway:")
    print("These are completely different functions for different tag formats!")
    print("The soft function expects <reasoning>/<answer>, while this project uses <think>/<answer>.")

def show_pattern_differences():
    """Show the regex pattern differences in detail."""
    print("\n🔍 Regex Pattern Analysis:")
    print("=" * 60)
    
    print("Soft function pattern:")
    print("   r\"<reasoning>.*?</reasoning>\\s*<answer>.*?</answer>\"")
    print("   • Uses re.match() - must start at beginning")
    print("   • Expects <reasoning> tag specifically")
    print("   • Allows optional whitespace between tags")
    print("   • No case insensitivity")
    
    print("\nFlexible function patterns:")
    print("   think_pattern = r\"<think\\s*>(.*?)</think\\s*>\"")
    print("   answer_pattern = r\"<answer\\s*>(.*?)</answer\\s*>\"")
    print("   • Uses re.search() - can find anywhere in text")
    print("   • Expects <think> tag specifically")
    print("   • Allows optional spaces inside tags")
    print("   • Case insensitive matching")
    print("   • Validates content is non-empty")
    print("   • Checks tag order")

if __name__ == "__main__":
    compare_format_functions()
    show_pattern_differences()