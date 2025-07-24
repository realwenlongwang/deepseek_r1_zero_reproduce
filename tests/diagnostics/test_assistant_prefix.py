#!/usr/bin/env python3
"""
Test to verify if the format reward handles the "Assistant: " prefix correctly.
"""

import sys
sys.path.append('src')
from src.rewards.openr1_rewards import format_reward

def test_assistant_prefix():
    """Test format reward with Assistant: prefix that unsloth models generate."""
    print("🧪 Testing Assistant Prefix Handling")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Standard format (no prefix)",
            "text": "<think> I need to solve this step by step </think><answer> The answer is 42 </answer>",
            "expected": 1.0,
            "description": "What standard models generate"
        },
        {
            "name": "Unsloth format (with Assistant: prefix)",
            "text": "Assistant: <think> I need to solve this step by step </think><answer> The answer is 42 </answer>",
            "expected": 1.0,
            "description": "What unsloth models generate"
        },
        {
            "name": "Unsloth format with extra whitespace",
            "text": "Assistant:  <think> Let me work through this </think><answer> Final result </answer>",
            "expected": 1.0,
            "description": "Unsloth with spacing variations"
        },
        {
            "name": "Invalid: Assistant prefix but wrong format",
            "text": "Assistant: Just an answer without think tags",
            "expected": 0.0,
            "description": "Should still fail if format is wrong"
        },
        {
            "name": "Invalid: Assistant prefix, missing think",
            "text": "Assistant: <answer> Only answer tag </answer>",
            "expected": 0.0,
            "description": "Should fail with missing think tag"
        }
    ]
    
    print("Testing various Assistant: prefix scenarios:")
    print()
    
    all_passed = True
    for case in test_cases:
        completion = [[{"content": case["text"]}]]
        actual_reward = format_reward(completion)[0]
        
        if actual_reward == case["expected"]:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
            all_passed = False
        
        print(f"{status} {case['name']}")
        print(f"   📝 {case['description']}")
        print(f"   🎯 Expected: {case['expected']}, Got: {actual_reward}")
        print(f"   📄 Text: {repr(case['text'])}")
        print()
    
    if not all_passed:
        print("❌ Some tests failed! The format_reward function needs to handle Assistant: prefix")
        return False
    else:
        print("🎉 All tests passed! The format_reward function correctly handles Assistant: prefix")
        return True

if __name__ == "__main__":
    success = test_assistant_prefix()
    
    if not success:
        print("\n💡 Recommendation:")
        print("The format_reward function should be updated to:")
        print("1. Strip 'Assistant: ' prefix before checking format")
        print("2. Handle any role prefixes that might appear in completions")
        print("3. Focus on the actual content after the role identifier")