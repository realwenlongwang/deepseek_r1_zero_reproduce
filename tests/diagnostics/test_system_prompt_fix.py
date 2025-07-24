#!/usr/bin/env python3
"""
Test to verify the system prompt fix resolves the format reward issue.
"""

import sys
sys.path.append('src')
from src.rewards.openr1_rewards import format_reward

def test_system_prompt_fix():
    """Test that new system prompt example format gets correct reward."""
    print("🧪 Testing System Prompt Fix")
    print("=" * 50)
    
    # Old format (from original system prompt)
    old_format = "<think> reasoning process here </think><answer> answer here </answer>"
    old_completion = [[{"content": old_format}]]
    old_reward = format_reward(old_completion)[0]
    
    # New format (from fixed system prompt)  
    new_format = "<think>reasoning process here</think>\n<answer>answer here</answer>"
    new_completion = [[{"content": new_format}]]
    new_reward = format_reward(new_completion)[0]
    
    print(f"❌ Old format reward: {old_reward} - '{old_format}'")
    print(f"✅ New format reward: {new_reward} - '{repr(new_format)}'")
    
    if new_reward == 1.0:
        print("\n🎉 SUCCESS: System prompt fix should resolve the format reward issue!")
        print("   Models trained with the updated prompt should now get non-zero format rewards.")
    else:
        print("\n❌ ISSUE: New format still doesn't work. Need further investigation.")
    
    return new_reward == 1.0

if __name__ == "__main__":
    test_system_prompt_fix()