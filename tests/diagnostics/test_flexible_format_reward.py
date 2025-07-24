#!/usr/bin/env python3
"""
Test the flexible format reward function to ensure it handles various format variations.
"""

import sys
sys.path.append('src')
from src.rewards.openr1_rewards import format_reward

def test_flexible_format_reward():
    """Test the flexible format reward function with various input formats."""
    print("🧪 Testing Flexible Format Reward Function")
    print("=" * 60)
    
    test_cases = [
        # Original strict format (should still work)
        ("Strict format", "<think>reasoning here</think>\n<answer>answer here</answer>", 1.0),
        
        # System prompt format (with spaces inside tags)
        ("System prompt format", "<think> reasoning here </think><answer> answer here </answer>", 1.0),
        
        # Mixed spacing variations
        ("No spaces in tags", "<think>reasoning</think><answer>answer</answer>", 1.0),
        ("Spaces in both tags", "<think> reasoning </think> <answer> answer </answer>", 1.0),
        ("Newline between tags", "<think>reasoning</think>\n<answer>answer</answer>", 1.0),
        ("Multiple newlines", "<think>reasoning</think>\n\n<answer>answer</answer>", 1.0),
        
        # Leading/trailing whitespace
        ("Leading whitespace", "  <think>reasoning</think><answer>answer</answer>", 1.0),
        ("Trailing whitespace", "<think>reasoning</think><answer>answer</answer>  ", 1.0),
        ("Both whitespace", "  <think>reasoning</think><answer>answer</answer>  ", 1.0),
        
        # Case variations
        ("Uppercase tags", "<THINK>reasoning</THINK><ANSWER>answer</ANSWER>", 1.0),
        ("Mixed case", "<Think>reasoning</Think><Answer>answer</Answer>", 1.0),
        
        # Complex content
        ("Multiline think", "<think>Step 1: do this\nStep 2: do that</think><answer>Final result</answer>", 1.0),
        ("HTML in content", "<think>Use <b>bold</b> formatting</think><answer>Result with <i>italics</i></answer>", 1.0),
        ("Math symbols", "<think>Check if 2 < 3 and 5 > 4</think><answer>Both are true</answer>", 1.0),
        
        # Should fail cases
        ("Missing think tag", "<answer>answer only</answer>", 0.0),
        ("Missing answer tag", "<think>thinking only</think>", 0.0),
        ("Wrong order", "<answer>answer first</answer><think>thinking second</think>", 0.0),
        ("Empty think content", "<think></think><answer>answer</answer>", 0.0),
        ("Empty answer content", "<think>thinking</think><answer></answer>", 0.0),
        ("Only whitespace in think", "<think>   </think><answer>answer</answer>", 0.0),
        ("Only whitespace in answer", "<think>thinking</think><answer>   </answer>", 0.0),
        ("No tags at all", "Just plain text without any tags", 0.0),
        ("Malformed tags", "<think>reasoning<answer>answer</answer>", 0.0),
        
        # Edge cases
        ("Nested similar tags", "<think>Use <think> carefully</think><answer>Be mindful</answer>", 1.0),
        ("Multiple think tags", "<think>first</think><think>second</think><answer>answer</answer>", 1.0),  # Should use first occurrence
        ("Multiple answer tags", "<think>thinking</think><answer>first</answer><answer>second</answer>", 1.0),  # Should use first occurrence
    ]
    
    passed = 0
    failed = 0
    
    for description, text, expected_reward in test_cases:
        completion = [[{"content": text}]]
        actual_reward = format_reward(completion)[0]
        
        if actual_reward == expected_reward:
            status = "✅"
            passed += 1
        else:
            status = "❌"
            failed += 1
        
        print(f"{status} {description}")
        print(f"   Expected: {expected_reward}, Got: {actual_reward}")
        
        if actual_reward != expected_reward:
            print(f"   Text: {repr(text)}")
        print()
    
    print(f"📊 Summary: {passed} passed, {failed} failed")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0

def test_original_vs_flexible():
    """Compare how the original strict format would perform vs the new flexible format."""
    print("\n🔄 Comparing Original vs Flexible Format Reward")
    print("=" * 60)
    
    # Test cases that would fail with original but pass with flexible
    flexible_cases = [
        "<think> reasoning here </think><answer> answer here </answer>",  # System prompt format
        "  <think>reasoning</think><answer>answer</answer>  ",  # Whitespace
        "<THINK>reasoning</THINK><ANSWER>answer</ANSWER>",  # Case insensitive
        "<think>reasoning</think>  <answer>answer</answer>",  # Space between tags
    ]
    
    print("Cases that would benefit from flexible format:")
    for i, case in enumerate(flexible_cases, 1):
        completion = [[{"content": case}]]
        reward = format_reward(completion)[0]
        print(f"{i}. Reward: {reward} - {repr(case)}")
    
    return True

if __name__ == "__main__":
    success = test_flexible_format_reward()
    test_original_vs_flexible()
    
    if success:
        print("\n🎉 All tests passed! The flexible format reward function is working correctly.")
        print("This should resolve the format reward issue for both standard and Unsloth models.")
    else:
        print("\n❌ Some tests failed. The format reward function needs further adjustment.")