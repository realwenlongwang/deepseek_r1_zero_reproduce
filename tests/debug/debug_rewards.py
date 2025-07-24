#!/usr/bin/env python3
"""
Debug script to investigate reward function issues.
"""

import re
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rewards.openr1_rewards import (
    format_reward,
    tag_count_reward,
    get_cosine_scaled_reward,
    accuracy_reward
)

# Test responses from our previous run
responses = [
    # Question 1 - No tags
    """To find the coefficient of \\( x^2 y^6 \\) in the expansion of \\( \\left( \\frac{3}{5}x - \\frac{y}{2} \\right)^8 \\), we use the Binomial Theorem. The Binomial Theorem states:

\\[
(a + b)^n = \\sum_{k=0}^{n} \\binom{n}{k} a^{n-k} b^k
\\]

In our case, \\( a = \\frac{3}{5}x \\), \\( b = -\\frac{y}{2} \\), and \\( n = 8 \\). We need to determine the term in the expansion where the powers of \\( x \\) and \\( y \\) match \\( x^2 y^6 \\).

The general term in the binomial expansion is given by:

\\[
T_k = \\binom{8}{k} \\lef...""",

    # Question 2 - No tags
    """To solve for \\( f(2023) \\), we start by using the periodicity of the function. Given that \\( f(x+4) = f(x) \\), the function \\( f \\) is periodic with period 4. This means that \\( f(x) = f(x + 4k) \\) for any integer \\( k \\). We can reduce \\( f(2023) \\) to a value within one period by finding the remainder when 2023 is divided by 4.

First, we compute:
\\[ 2023 \\div 4 = 505 \\text{ remainder } 3. \\]
This tells us that \\( 2023 = 4 \\cdot 505 + 3 \\), so \\( f(2023) = f(3) \\).

Next, we need to determine ...""",

    # Question 3 - Has tags
    """<think> I know that to find out how much money Linda has, I should calculate the total sales from jeans and tees separately and then add them together. To do this, I will multiply the number of each type of shirt by its price: jeans * price per jean + tees * price per tee = total sales. Now, I will perform the calculations. </think><answer> 11 * 4 + 8 * 7 = 44 + 56 = 100 </answer>"""
]

# Ground truth solutions
solutions = [
    """To determine the coefficient of \\(x^2y^6\\) in the expansion of \\(\\left(\\frac{3}{5}x - \\frac{y}{2}\\right)^8\\)...""",
    """To solve the problem, let's break it down step-by-step:

1. **Understand the properties of the function**...""",
    """To determine how much money Linda made at the end of the day, we need to calculate the total revenue..."""
]

def debug_format_reward():
    """Debug the format reward function."""
    print("="*80)
    print("DEBUGGING FORMAT REWARD")
    print("="*80)
    
    # Check the regex pattern
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    print(f"Regex pattern: {pattern}")
    
    for i, response in enumerate(responses):
        print(f"\n--- Response {i+1} ---")
        print(f"Response: {response[:100]}...")
        
        # Test the regex
        match = re.match(pattern, response, re.DOTALL | re.MULTILINE)
        print(f"Regex match: {match is not None}")
        
        # Check for tag presence
        has_think = '<think>' in response and '</think>' in response
        has_answer = '<answer>' in response and '</answer>' in response
        print(f"Has <think> tags: {has_think}")
        print(f"Has <answer> tags: {has_answer}")
        
        if has_think and has_answer:
            print("ISSUE: Response has tags but regex doesn't match!")
            print("Let's examine the exact format...")
            
            # Find the tags
            think_start = response.find('<think>')
            think_end = response.find('</think>')
            answer_start = response.find('<answer>')
            answer_end = response.find('</answer>')
            
            if think_start != -1 and think_end != -1:
                think_part = response[think_start:think_end+8]
                print(f"Think part: {repr(think_part[:50])}...")
            
            if answer_start != -1 and answer_end != -1:
                answer_part = response[answer_start:answer_end+9]
                print(f"Answer part: {repr(answer_part)}")
            
            # Check what's between tags
            if think_end != -1 and answer_start != -1:
                between = response[think_end+8:answer_start]
                print(f"Between tags: {repr(between)}")

def debug_tag_count_reward():
    """Debug the tag count reward function."""
    print("\n" + "="*80)
    print("DEBUGGING TAG COUNT REWARD")
    print("="*80)
    
    def count_tags(text: str) -> float:
        """Recreate the tag counting logic to debug."""
        count = 0.0
        checks = [
            ("<think>\\n", text.count("<think>\\n")),
            ("\\n</think>\\n", text.count("\\n</think>\\n")),
            ("\\n<answer>\\n", text.count("\\n<answer>\\n")),
            ("\\n</answer>", text.count("\\n</answer>"))
        ]
        
        for pattern, found_count in checks:
            print(f"  Pattern '{pattern}': found {found_count} times")
            if found_count == 1:
                count += 0.25
                print(f"    -> Added 0.25 (total: {count})")
        
        return count
    
    for i, response in enumerate(responses):
        print(f"\n--- Response {i+1} ---")
        print(f"Response: {response[:100]}...")
        
        count = count_tags(response)
        print(f"Final tag count reward: {count}")
        
        if '<think>' in response and '</think>' in response:
            print("ISSUE: Response has tags but gets 0 reward!")
            print("The function expects very specific newline patterns.")

def debug_cosine_scaled_reward():
    """Debug the cosine scaled reward function."""
    print("\n" + "="*80) 
    print("DEBUGGING COSINE SCALED REWARD")
    print("="*80)
    
    # Test with known accuracy values
    accuracy_rewards = [0.0, 1.0, 0.0]  # From our test results
    
    cosine_reward_func = get_cosine_scaled_reward(
        min_value_wrong=-0.5,
        max_value_wrong=-0.1,
        min_value_correct=0.8,
        max_value_correct=1.0,
        max_len=1000
    )
    
    # Format as expected by the function
    completions = [
        [{"content": response}] for response in responses
    ]
    
    print("Testing cosine scaled reward with known accuracy values:")
    for i, (response, acc_reward) in enumerate(zip(responses, accuracy_rewards)):
        print(f"\n--- Response {i+1} ---")
        print(f"Length: {len(response)} characters")
        print(f"Accuracy reward: {acc_reward}")
        print(f"Is correct: {acc_reward > 0.5}")
        
        # Calculate expected cosine reward manually
        import math
        gen_len = len(response)
        progress = gen_len / 1000
        cosine = math.cos(progress * math.pi)
        
        if acc_reward > 0.5:  # Correct
            min_value = 0.8
            max_value = 1.0
        else:  # Incorrect
            min_value = -0.1  # max_value_wrong
            max_value = -0.5  # min_value_wrong
        
        expected_reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        print(f"Progress: {progress:.3f}, Cosine: {cosine:.3f}")
        print(f"Expected reward: {expected_reward:.3f}")

def debug_accuracy_reward():
    """Debug the accuracy reward function."""
    print("\n" + "="*80)
    print("DEBUGGING ACCURACY REWARD")
    print("="*80)
    
    completions = [
        [{"content": response}] for response in responses
    ]
    
    for i, (response, solution) in enumerate(zip(responses, solutions)):
        print(f"\n--- Response {i+1} ---")
        print(f"Response: {response[:100]}...")
        print(f"Solution: {solution[:100]}...")
        
        try:
            reward = accuracy_reward(completions[i:i+1], [solution])
            print(f"Accuracy reward: {reward}")
            
            # For Question 3, let's check the math manually
            if i == 2:  # Question 3
                print("Manual check for Question 3:")
                print("Response says: 11 * 4 + 8 * 7 = 44 + 56 = 100")
                print("This is correct math, but parsing might be failing")
                
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Run all debug functions."""
    debug_format_reward()
    debug_tag_count_reward()
    debug_cosine_scaled_reward()
    debug_accuracy_reward()

if __name__ == "__main__":
    main()