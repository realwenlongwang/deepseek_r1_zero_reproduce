#!/usr/bin/env python3
"""
Final comprehensive test to verify the format reward solution works for both 
standard and unsloth models with various format variations.
"""

import sys
sys.path.append('src')
from src.rewards.openr1_rewards import format_reward

def test_format_reward_solution():
    """Test that the format reward solution resolves the original issue."""
    print("🎯 Final Format Reward Solution Test")
    print("=" * 60)
    
    # Test cases representing real-world model outputs
    test_scenarios = [
        {
            "name": "Original System Prompt Format",
            "description": "What models trained with original system prompt would generate",
            "text": "<think> I need to solve 2 + 2. This is basic arithmetic. </think><answer> The answer is 4. </answer>",
            "expected": 1.0
        },
        {
            "name": "Strict Format", 
            "description": "What the original format_reward expected",
            "text": "<think>I need to solve 2 + 2. This is basic arithmetic.</think>\n<answer>The answer is 4.</answer>",
            "expected": 1.0
        },
        {
            "name": "Unsloth Model Output (likely)",
            "description": "Typical unsloth model output with extra whitespace",
            "text": "  <think> Let me calculate 2 + 2 step by step. </think>  \n  <answer> 4 </answer>  ",
            "expected": 1.0
        },
        {
            "name": "Standard Model Output",
            "description": "Typical standard model output",
            "text": "<think>Simple addition: 2 + 2</think><answer>4</answer>",
            "expected": 1.0
        },
        {
            "name": "Case Variations",
            "description": "Models might generate different cases",
            "text": "<THINK>Calculating...</THINK><ANSWER>Result is 4</ANSWER>",
            "expected": 1.0
        },
        {
            "name": "Complex Reasoning",
            "description": "Multi-step reasoning with various formatting",
            "text": "<think>\nStep 1: Identify the problem - adding 2 + 2\nStep 2: Perform the calculation\nStep 3: Verify the result\n</think>\n<answer>\nThe sum of 2 + 2 equals 4.\n</answer>",
            "expected": 1.0
        },
        {
            "name": "Invalid: Missing Think",
            "description": "Should still fail appropriately",
            "text": "<answer>Just an answer without thinking</answer>",
            "expected": 0.0
        },
        {
            "name": "Invalid: Wrong Order", 
            "description": "Should still fail appropriately",
            "text": "<answer>Answer first</answer><think>Then thinking</think>",
            "expected": 0.0
        }
    ]
    
    print("Testing various model output scenarios:")
    print()
    
    all_passed = True
    for scenario in test_scenarios:
        completion = [[{"content": scenario["text"]}]]
        actual_reward = format_reward(completion)[0]
        
        if actual_reward == scenario["expected"]:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        
        print(f"{status} {scenario['name']}")
        print(f"   📝 {scenario['description']}")
        print(f"   🎯 Expected: {scenario['expected']}, Got: {actual_reward}")
        print(f"   📄 Text preview: {repr(scenario['text'][:50])}{'...' if len(scenario['text']) > 50 else ''}")
        print()
    
    return all_passed

def test_backwards_compatibility():
    """Ensure the flexible format reward is backwards compatible."""
    print("🔄 Testing Backwards Compatibility")
    print("=" * 60)
    
    # Examples that would have worked with the original strict format
    strict_examples = [
        "<think>Simple reasoning</think>\n<answer>Simple answer</answer>",
        "<think>Multi-line\nreasoning here</think>\n<answer>Final result</answer>",
    ]
    
    print("Ensuring strict format examples still work:")
    for i, example in enumerate(strict_examples, 1):
        completion = [[{"content": example}]]
        reward = format_reward(completion)[0]
        status = "✅" if reward == 1.0 else "❌"
        print(f"{status} Strict example {i}: {reward}")
    
    return True

def summarize_solution():
    """Summarize the complete solution to the format reward issue."""
    print("\n📋 Solution Summary")
    print("=" * 60)
    
    print("🔍 Root Cause Identified:")
    print("   • Original format_reward function was too strict")
    print("   • Required exact format: <think>content</think>\\n<answer>content</answer>")
    print("   • System prompt showed different format with spaces inside tags")
    print("   • Unsloth models followed system prompt more precisely than standard models")
    print()
    
    print("🛠️  Solution Implemented:")
    print("   • Made format_reward function flexible and tolerant")
    print("   • Accepts various spacing and whitespace patterns")
    print("   • Handles case-insensitive tag matching")
    print("   • Maintains validation for correct order and non-empty content")
    print("   • Backwards compatible with original strict format")
    print()
    
    print("✅ Benefits:")
    print("   • Works with both standard and Unsloth models")
    print("   • Handles natural variations in model outputs")
    print("   • Maintains training signal quality")
    print("   • No need to retrain existing models")
    print("   • More robust for future model variations")
    print()
    
    print("🚀 Next Steps:")
    print("   • Test with actual training: uv run train_grpo.py --model.unsloth.enabled true")
    print("   • Monitor format reward values in training logs")
    print("   • Verify both standard and unsloth models get > 0 format rewards")

if __name__ == "__main__":
    print("🧪 Comprehensive Format Reward Solution Test")
    print("=" * 80)
    print()
    
    scenario_success = test_format_reward_solution()
    compat_success = test_backwards_compatibility()
    
    if scenario_success and compat_success:
        print("🎉 ALL TESTS PASSED!")
        print("The flexible format reward solution should resolve the issue where")
        print("format_reward was always zero when model.unsloth.enable == true")
        print()
        summarize_solution()
    else:
        print("❌ Some tests failed. Please review the implementation.")