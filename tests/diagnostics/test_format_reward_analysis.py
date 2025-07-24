#!/usr/bin/env python3
"""
Lightweight test to analyze format reward patterns and identify why unsloth models
might be getting zero rewards.
"""

import os
import sys
sys.path.append('src')
import re
from src.rewards.openr1_rewards import format_reward

def analyze_format_reward_regex():
    """Analyze the regex pattern used in format_reward to understand what formats are accepted."""
    print("🔍 Analyzing Format Reward Regex Pattern")
    print("=" * 60)
    
    # The regex from format_reward function
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    print(f"📝 Regex pattern: {regex}")
    print()
    
    # Break down the regex components
    print("🧩 Regex components breakdown:")
    print("   ^                          - Must start at beginning of string")
    print("   <think>                    - Literal '<think>' tag")
    print("   ([^<]*(?:<(?!/?think>)[^<]*)*) - Think content (allows < but not </think> or <think>)")
    print("   </think>                   - Literal '</think>' tag")
    print("   \\n                         - Exactly one newline")
    print("   <answer>                   - Literal '<answer>' tag")  
    print("   ([\\s\\S]*?)                - Answer content (any characters, non-greedy)")
    print("   </answer>                  - Literal '</answer>' tag")
    print("   $                          - Must end at end of string")
    print()
    
    return regex

def test_format_variations():
    """Test various format variations to understand what passes and fails."""
    print("🧪 Testing Format Variations")
    print("=" * 60)
    
    test_cases = [
        # Expected valid formats
        ("Valid basic format", "<think>Simple reasoning</think>\n<answer>42</answer>", True),
        ("Valid with complex reasoning", "<think>First I do X, then Y, finally Z</think>\n<answer>The result is 42</answer>", True),
        ("Valid with multiline answer", "<think>Reasoning</think>\n<answer>Line 1\nLine 2\nLine 3</answer>", True),
        
        # Common variations that might come from unsloth
        ("Extra newline before think", "\n<think>Reasoning</think>\n<answer>42</answer>", False),
        ("Extra newline after answer", "<think>Reasoning</think>\n<answer>42</answer>\n", False),
        ("No newline between tags", "<think>Reasoning</think><answer>42</answer>", False),
        ("Multiple newlines between", "<think>Reasoning</think>\n\n<answer>42</answer>", False),
        ("Extra whitespace", " <think>Reasoning</think>\n<answer>42</answer> ", False),
        ("Different case tags", "<Think>Reasoning</Think>\n<Answer>42</Answer>", False),
        
        # Unsloth might generate these patterns
        ("Assistant prefix", "Assistant: <think>Reasoning</think>\n<answer>42</answer>", False),
        ("Text before tags", "Let me solve this:\n<think>Reasoning</think>\n<answer>42</answer>", False),
        ("Text after tags", "<think>Reasoning</think>\n<answer>42</answer>\nI hope this helps!", False),
        
        # Missing components
        ("Missing think tag", "<answer>42</answer>", False),
        ("Missing answer tag", "<think>Reasoning</think>", False),
        ("Missing both tags", "Just plain text with reasoning and answer", False),
        
        # Malformed tags
        ("Unclosed think tag", "<think>Reasoning\n<answer>42</answer>", False),
        ("Unclosed answer tag", "<think>Reasoning</think>\n<answer>42", False),
        ("Wrong tag order", "<answer>42</answer>\n<think>Reasoning</think>", False),
        
        # Edge cases with special characters
        ("HTML in think content", "<think>Use <b>bold</b> text</think>\n<answer>42</answer>", False),  # This might fail due to regex
        ("Math symbols", "<think>Calculate 2 < 3 and 5 > 4</think>\n<answer>True</answer>", False),
    ]
    
    regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
    
    passed = 0
    failed = 0
    
    for description, text, expected_pass in test_cases:
        # Test with format_reward function
        completion = [[{"content": text}]]
        reward = format_reward([completion])[0]
        
        # Also test with direct regex
        match = re.search(regex, text, re.DOTALL)
        regex_passes = match is not None and len(match.groups()) == 2
        
        # Check if results match expectations
        reward_correct = (reward == 1.0) == expected_pass
        regex_correct = regex_passes == expected_pass
        
        status = "✅" if reward_correct else "❌"
        print(f"{status} {description}")
        print(f"   Expected: {'PASS' if expected_pass else 'FAIL'}")
        print(f"   Reward: {reward} | Regex: {'PASS' if regex_passes else 'FAIL'}")
        
        if not reward_correct:
            print(f"   ⚠️  Unexpected result!")
            print(f"   📄 Text: {repr(text)}")
        
        if expected_pass:
            passed += 1
        else:
            failed += 1
        
        print()
    
    print(f"📊 Summary: {passed} expected passes, {failed} expected failures")
    return test_cases

def suggest_unsloth_debugging_steps():
    """Suggest specific debugging steps for unsloth format reward issue."""
    print("🔧 Debugging Steps for Unsloth Format Reward Issue")
    print("=" * 60)
    
    steps = [
        "1. 🎯 Capture actual unsloth model outputs during training",
        "   - Add logging in GRPO trainer to save generated completions",
        "   - Compare first few completions between standard and unsloth models",
        "",
        "2. 🔍 Check system prompt and chat template effectiveness", 
        "   - Verify system prompt includes format instructions",
        "   - Test if unsloth models properly use chat templates",
        "",
        "3. 🎮 Test different generation parameters",
        "   - Try lower temperature (more deterministic)",
        "   - Adjust top_p, top_k sampling parameters", 
        "   - Test with different max_new_tokens",
        "",
        "4. 📝 Analyze training data format compliance",
        "   - Check if training examples follow <think>...</think><answer>...</answer> format",
        "   - Verify system prompt instructs model to use this format",
        "",
        "5. 🚀 Test unsloth-specific configuration",
        "   - Try disabling fast_inference mode",
        "   - Test with load_in_4bit disabled",
        "   - Compare different unsloth versions",
        "",
        "6. 🔧 Add format debugging to reward function",
        "   - Log failed format matches with full text",
        "   - Add partial scoring for near-matches",
        "   - Create format hint rewards to guide training",
    ]
    
    for step in steps:
        print(step)
    
    return steps

def create_format_debugging_reward():
    """Create a debugging version of format_reward that provides more detailed feedback."""
    print("\n🛠️  Enhanced Format Reward for Debugging")
    print("=" * 60)
    
    def debug_format_reward(completions, **kwargs):
        """Enhanced format reward function with detailed logging."""
        rewards = []
        completion_contents = [completion[0]["content"] for completion in completions]
        
        for i, completion in enumerate(completion_contents):
            print(f"\n--- Completion {i+1} ---")
            print(f"Raw content: {repr(completion)}")
            
            try:
                # Check if the format is correct
                regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
                match = re.search(regex, completion, re.DOTALL)
                
                if match is None:
                    print("❌ No regex match found")
                    
                    # Provide detailed diagnostics
                    if not completion.startswith("<think>"):
                        print("   🔍 Issue: Doesn't start with <think>")
                    elif "</think>" not in completion:
                        print("   🔍 Issue: Missing </think> tag")
                    elif "<answer>" not in completion:
                        print("   🔍 Issue: Missing <answer> tag")
                    elif not completion.endswith("</answer>"):
                        print("   🔍 Issue: Doesn't end with </answer>")
                    else:
                        # Check newline between tags
                        think_end = completion.find("</think>")
                        answer_start = completion.find("<answer>", think_end)
                        if think_end != -1 and answer_start != -1:
                            between_tags = completion[think_end + 8:answer_start]
                            if between_tags != "\n":
                                print(f"   🔍 Issue: Wrong content between tags: {repr(between_tags)} (expected '\\n')")
                    
                    rewards.append(0.0)
                elif len(match.groups()) != 2:
                    print(f"❌ Regex matched but wrong number of groups: {len(match.groups())}")
                    rewards.append(0.0)
                else:
                    print("✅ Format is correct!")
                    think_content = match.group(1)
                    answer_content = match.group(2)
                    print(f"   Think: {repr(think_content[:50])}{'...' if len(think_content) > 50 else ''}")
                    print(f"   Answer: {repr(answer_content[:50])}{'...' if len(answer_content) > 50 else ''}")
                    rewards.append(1.0)
                    
            except Exception as e:
                print(f"❌ Exception: {e}")
                rewards.append(0.0)
        
        return rewards
    
    # Test the debug function
    print("\n🧪 Testing Debug Format Reward:")
    test_completions = [
        [[{"content": "<think>Good reasoning</think>\n<answer>Correct answer</answer>"}]],
        [[{"content": "Bad format without tags"}]],
        [[{"content": "<think>Missing newline</think><answer>Wrong</answer>"}]],
    ]
    
    for completion in test_completions:
        debug_format_reward(completion)
    
    return debug_format_reward

if __name__ == "__main__":
    analyze_format_reward_regex()
    print()
    test_format_variations()
    print()
    suggest_unsloth_debugging_steps()
    create_format_debugging_reward()