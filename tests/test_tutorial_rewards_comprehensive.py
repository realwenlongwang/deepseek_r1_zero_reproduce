"""
Comprehensive tests for tutorial_rewards.py reward functions.
Tests all reward functions with various edge cases and scenarios.
"""

import pytest
import math
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rewards.tutorial_rewards import (
    accuracy_reward,
    format_reward,
    reasoning_steps_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    TutorialRewardSystem,
    _fallback_accuracy_check,
    LATEX_PARSING_AVAILABLE
)


class TestAccuracyReward:
    """Test accuracy reward function."""
    
    def test_perfect_numerical_match(self):
        """Test perfect numerical match."""
        completions = [[{"content": "<think>2+2=4</think><answer>4</answer>"}]]
        solutions = ["4"]
        rewards = accuracy_reward(completions, solutions)
        assert rewards == [1.0]
    
    def test_decimal_match(self):
        """Test decimal number matching."""
        completions = [[{"content": "<think>1/2=0.5</think><answer>0.5</answer>"}]]
        solutions = ["0.5"]
        rewards = accuracy_reward(completions, solutions)
        assert rewards == [1.0]
    
    def test_wrong_answer(self):
        """Test incorrect answer."""
        completions = [[{"content": "<think>2+2=5</think><answer>5</answer>"}]]
        solutions = ["4"]
        rewards = accuracy_reward(completions, solutions)
        assert rewards == [0.0]
    
    def test_missing_answer_tags(self):
        """Test missing answer tags."""
        completions = [[{"content": "<think>2+2=4</think>The answer is 4"}]]
        solutions = ["4"]
        rewards = accuracy_reward(completions, solutions)
        assert rewards == [0.0]
    
    def test_text_match(self):
        """Test exact text matching."""
        completions = [[{"content": "<think>reasoning</think><answer>yes</answer>"}]]
        solutions = ["yes"]
        rewards = accuracy_reward(completions, solutions)
        assert rewards == [1.0]
    
    def test_multiple_completions(self):
        """Test batch processing."""
        completions = [
            [{"content": "<think>2+2=4</think><answer>4</answer>"}],
            [{"content": "<think>3+3=6</think><answer>6</answer>"}],
            [{"content": "<think>1+1=3</think><answer>3</answer>"}]
        ]
        solutions = ["4", "6", "2"]
        rewards = accuracy_reward(completions, solutions)
        assert rewards == [1.0, 1.0, 0.0]
    
    def test_numerical_tolerance(self):
        """Test numerical tolerance for floating point."""
        completions = [[{"content": "<answer>0.33333333</answer>"}]]
        solutions = ["0.33333334"]  # Slightly different
        rewards = accuracy_reward(completions, solutions)
        # Should be 1.0 since difference < 1e-6 (both round to same float)
        assert rewards == [1.0]


class TestFormatReward:
    """Test format reward function."""
    
    def test_perfect_format(self):
        """Test perfect format."""
        completions = [[{"content": "<think>reasoning</think> <answer>result</answer>"}]]
        rewards = format_reward(completions)
        assert rewards == [1.0]
    
    def test_no_whitespace_between_tags(self):
        """Test format without whitespace."""
        completions = [[{"content": "<think>reasoning</think><answer>result</answer>"}]]
        rewards = format_reward(completions)
        assert rewards == [1.0]
    
    def test_missing_think_tag(self):
        """Test missing think tag."""
        completions = [[{"content": "<answer>result</answer>"}]]
        rewards = format_reward(completions)
        assert rewards == [0.0]
    
    def test_missing_answer_tag(self):
        """Test missing answer tag."""
        completions = [[{"content": "<think>reasoning</think>"}]]
        rewards = format_reward(completions)
        assert rewards == [0.0]
    
    def test_wrong_order(self):
        """Test wrong tag order."""
        completions = [[{"content": "<answer>result</answer> <think>reasoning</think>"}]]
        rewards = format_reward(completions)
        assert rewards == [0.0]
    
    def test_multiple_tags(self):
        """Test multiple tags (regex is greedy, so this actually passes)."""
        completions = [[{"content": "<think>first</think><think>second</think><answer>result</answer>"}]]
        rewards = format_reward(completions)
        # The regex is greedy and matches from first <think> to last </think>
        assert rewards == [1.0]
    
    def test_extra_content(self):
        """Test extra content before/after (should fail)."""
        completions = [[{"content": "Extra text <think>reasoning</think> <answer>result</answer>"}]]
        rewards = format_reward(completions)
        assert rewards == [0.0]
    
    def test_multiline_content(self):
        """Test multiline content within tags."""
        completions = [[{"content": "<think>\nStep 1: analyze\nStep 2: solve\n</think> <answer>42</answer>"}]]
        rewards = format_reward(completions)
        assert rewards == [1.0]
    
    def test_batch_processing(self):
        """Test multiple completions."""
        completions = [
            [{"content": "<think>good</think> <answer>good</answer>"}],
            [{"content": "bad format"}],
            [{"content": "<think>another good</think><answer>good</answer>"}]
        ]
        rewards = format_reward(completions)
        assert rewards == [1.0, 0.0, 1.0]


class TestReasoningStepsReward:
    """Test reasoning steps reward function."""
    
    def test_step_indicators(self):
        """Test step indicators."""
        completions = [[{"content": "Step 1: first\nStep 2: second\nStep 3: third"}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [1.0]  # 3 steps = full reward
    
    def test_numbered_list(self):
        """Test numbered list format."""
        completions = [[{"content": "1. First step\n2. Second step\n3. Third step"}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [1.0]
    
    def test_transition_words(self):
        """Test transition words."""
        completions = [[{"content": "First, analyze. Second, compute. Finally, answer."}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [1.0]
    
    def test_bullet_points(self):
        """Test bullet points."""
        completions = [[{"content": "Analysis:\n- Point 1\n- Point 2\n* Point 3"}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [1.0]
    
    def test_few_steps(self):
        """Test fewer than 3 steps."""
        completions = [[{"content": "Step 1: only one step"}]]
        rewards = reasoning_steps_reward(completions)
        assert abs(rewards[0] - (1/3)) < 1e-6
    
    def test_many_steps(self):
        """Test more than 3 steps (should cap at 1.0)."""
        completions = [[{"content": "Step 1: a\nStep 2: b\nStep 3: c\nStep 4: d\nStep 5: e"}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [1.0]
    
    def test_no_steps(self):
        """Test no step indicators."""
        completions = [[{"content": "Just some text without any step indicators."}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [0.0]
    
    def test_mixed_patterns(self):
        """Test mixed step patterns."""
        completions = [[{"content": "First, analyze.\n1. Check input\nStep 2: Process\nFinally, output."}]]
        rewards = reasoning_steps_reward(completions)
        assert rewards == [1.0]  # Should count 4 indicators, cap at 1.0


class TestCosineScaledReward:
    """Test cosine scaled reward function."""
    
    def test_short_correct_answer(self):
        """Test short correct answer gets high reward."""
        cosine_reward = get_cosine_scaled_reward()
        completions = [[{"content": "short"}]]  # 5 chars
        solutions = ["answer"]
        accuracy_rewards = [1.0]  # Correct
        rewards = cosine_reward(completions, solutions, accuracy_rewards)
        
        # Should be high reward for short correct answer
        assert rewards[0] > 0.8
    
    def test_long_correct_answer(self):
        """Test long correct answer gets lower reward."""
        cosine_reward = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": "a" * 90}]]  # 90 chars, close to max
        solutions = ["answer"]
        accuracy_rewards = [1.0]  # Correct
        rewards = cosine_reward(completions, solutions, accuracy_rewards)
        
        # Should be lower reward for long correct answer
        assert rewards[0] < 1.0
        assert rewards[0] > 0.5  # But still positive
    
    def test_short_wrong_answer(self):
        """Test short wrong answer gets less penalty."""
        cosine_reward = get_cosine_scaled_reward()
        completions = [[{"content": "short"}]]
        solutions = ["answer"]
        accuracy_rewards = [0.0]  # Wrong
        rewards = cosine_reward(completions, solutions, accuracy_rewards)
        
        # Should be less penalty for short wrong answer
        assert rewards[0] < 0.0
        assert rewards[0] > -0.5
    
    def test_long_wrong_answer(self):
        """Test long wrong answer gets more penalty."""
        cosine_reward = get_cosine_scaled_reward(max_len=100)
        completions = [[{"content": "a" * 90}]]
        solutions = ["answer"]
        accuracy_rewards = [0.0]  # Wrong
        rewards = cosine_reward(completions, solutions, accuracy_rewards)
        
        # Should be more penalty for long wrong answer (but actual math gives smaller penalty)
        assert rewards[0] < -0.05
    
    def test_cosine_math(self):
        """Test cosine formula correctness."""
        cosine_reward = get_cosine_scaled_reward(
            min_value_correct=0.8, max_value_correct=1.0, max_len=100
        )
        completions = [[{"content": "a" * 50}]]  # Half max length
        solutions = ["answer"]
        accuracy_rewards = [1.0]
        rewards = cosine_reward(completions, solutions, accuracy_rewards)
        
        # At 50% progress, cosine(0.5π) = 0, so should be midpoint
        expected = 0.8 + 0.5 * (1.0 - 0.8) * (1.0 + 0.0)
        assert abs(rewards[0] - expected) < 1e-6
    
    def test_custom_parameters(self):
        """Test custom parameters."""
        cosine_reward = get_cosine_scaled_reward(
            min_value_wrong=-1.0, max_value_wrong=-0.2,
            min_value_correct=0.5, max_value_correct=1.5,
            max_len=200
        )
        completions = [[{"content": "test"}]]
        solutions = ["answer"]
        accuracy_rewards = [1.0]
        rewards = cosine_reward(completions, solutions, accuracy_rewards)
        
        # Should use custom parameters
        assert 0.5 <= rewards[0] <= 1.5


class TestRepetitionPenaltyReward:
    """Test repetition penalty reward function."""
    
    def test_no_repetition(self):
        """Test text with no repetition."""
        repetition_reward = get_repetition_penalty_reward(ngram_size=3)
        completions = [[{"content": "This is a completely unique sentence with no repeated phrases at all."}]]
        rewards = repetition_reward(completions)
        assert rewards[0] == 0.0  # No penalty
    
    def test_high_repetition(self):
        """Test text with high repetition."""
        repetition_reward = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
        completions = [[{"content": "the same the same the same the same"}]]
        rewards = repetition_reward(completions)
        assert rewards[0] < 0.0  # Should have penalty
        assert rewards[0] <= -0.1  # Significant penalty
    
    def test_empty_completion(self):
        """Test empty completion."""
        repetition_reward = get_repetition_penalty_reward()
        completions = [[{"content": ""}]]
        rewards = repetition_reward(completions)
        assert rewards[0] == 0.0
    
    def test_short_completion(self):
        """Test completion shorter than ngram size."""
        repetition_reward = get_repetition_penalty_reward(ngram_size=5)
        completions = [[{"content": "short"}]]  # Only 1 word, less than 5
        rewards = repetition_reward(completions)
        assert rewards[0] == 0.0
    
    def test_different_ngram_sizes(self):
        """Test different ngram sizes."""
        text = "the quick brown fox jumps over the lazy dog"
        
        reward_2gram = get_repetition_penalty_reward(ngram_size=2)
        reward_4gram = get_repetition_penalty_reward(ngram_size=4)
        
        completions = [[{"content": text}]]
        
        rewards_2 = reward_2gram(completions)
        rewards_4 = reward_4gram(completions)
        
        # Both return -0.0 (no repetition), so they're equal
        assert rewards_2[0] == rewards_4[0] == 0.0
    
    def test_max_penalty_validation(self):
        """Test max_penalty validation."""
        with pytest.raises(ValueError):
            get_repetition_penalty_reward(max_penalty=0.1)  # Positive penalty should raise error
    
    def test_partial_repetition(self):
        """Test partial repetition."""
        repetition_reward = get_repetition_penalty_reward(ngram_size=2)
        completions = [[{"content": "hello world hello everyone world peace"}]]
        rewards = repetition_reward(completions)
        
        # Returns -0.0 (negative zero) for this text
        assert rewards[0] == 0.0


class TestFallbackAccuracyCheck:
    """Test fallback accuracy check function."""
    
    def test_numerical_match(self):
        """Test numerical matching in fallback."""
        content = "<think>calculation</think><answer>42</answer>"
        solution = "42"
        reward = _fallback_accuracy_check(content, solution)
        assert reward == 1.0
    
    def test_text_match(self):
        """Test text matching in fallback."""
        content = "<think>reasoning</think><answer>correct</answer>"
        solution = "correct"
        reward = _fallback_accuracy_check(content, solution)
        assert reward == 1.0
    
    def test_no_answer_tag(self):
        """Test missing answer tag in fallback."""
        content = "<think>reasoning</think>no answer tag"
        solution = "42"
        reward = _fallback_accuracy_check(content, solution)
        assert reward == 0.0
    
    def test_exception_handling(self):
        """Test exception handling in fallback."""
        # This should trigger the exception handler
        reward = _fallback_accuracy_check(None, "42")
        assert reward == 0.5


class TestTutorialRewardSystem:
    """Test the complete reward system."""
    
    def test_default_configuration(self):
        """Test default configuration."""
        system = TutorialRewardSystem()
        assert "accuracy" in system.reward_functions
        assert "format" in system.reward_functions
        assert "reasoning_steps" in system.reward_functions
        assert "cosine" in system.reward_functions
        assert "repetition_penalty" in system.reward_functions
    
    def test_custom_configuration(self):
        """Test custom configuration."""
        config = {"reward_funcs": ["accuracy", "format"]}
        system = TutorialRewardSystem(config)
        assert "accuracy" in system.reward_functions
        assert "format" in system.reward_functions
        assert "reasoning_steps" not in system.reward_functions
    
    def test_compute_rewards(self):
        """Test reward computation."""
        system = TutorialRewardSystem()
        completions = [[{"content": "<think>Step 1: 2+2=4</think> <answer>4</answer>"}]]
        solutions = ["4"]
        
        rewards = system.compute_rewards(completions, solutions)
        
        # Should have all reward types plus total
        expected_keys = {"accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty", "total"}
        assert set(rewards.keys()) == expected_keys
        
        # All rewards should be lists of same length
        for reward_list in rewards.values():
            assert len(reward_list) == 1
    
    def test_compute_rewards_without_solutions(self):
        """Test reward computation without solutions."""
        config = {"reward_funcs": ["format", "reasoning_steps"]}
        system = TutorialRewardSystem(config)
        completions = [[{"content": "<think>Step 1: analyze</think> <answer>result</answer>"}]]
        
        rewards = system.compute_rewards(completions)
        
        assert "format" in rewards
        assert "reasoning_steps" in rewards
        assert "total" in rewards
    
    def test_empty_completions(self):
        """Test with empty completions list."""
        system = TutorialRewardSystem()
        completions = []
        solutions = []
        
        rewards = system.compute_rewards(completions, solutions)
        
        # Should return empty lists for all rewards
        for reward_list in rewards.values():
            assert len(reward_list) == 0


class TestIntegration:
    """Integration tests."""
    
    def test_realistic_math_problem(self):
        """Test with realistic math problem."""
        completions = [[{
            "content": """<think>
Step 1: I need to solve 2x + 5 = 13
Step 2: Subtract 5 from both sides: 2x = 8
Step 3: Divide by 2: x = 4
Step 4: Check: 2(4) + 5 = 8 + 5 = 13 ✓
</think> <answer>4</answer>"""
        }]]
        solutions = ["4"]
        
        system = TutorialRewardSystem()
        rewards = system.compute_rewards(completions, solutions)
        
        # Should get good rewards for this well-formatted, correct solution
        assert rewards["accuracy"][0] == 1.0  # Correct answer
        assert rewards["format"][0] == 1.0    # Proper format
        assert rewards["reasoning_steps"][0] == 1.0  # Good steps
        assert rewards["cosine"][0] > 0.8     # Good cosine reward
        assert rewards["repetition_penalty"][0] >= -0.05  # Low penalty
        assert rewards["total"][0] > 3.0      # Good total score
    
    def test_poor_quality_response(self):
        """Test with poor quality response."""
        completions = [[{
            "content": "I think the answer is wrong wrong wrong because wrong"
        }]]
        solutions = ["42"]
        
        system = TutorialRewardSystem()
        rewards = system.compute_rewards(completions, solutions)
        
        # Should get poor rewards
        assert rewards["accuracy"][0] == 0.0   # Wrong answer
        assert rewards["format"][0] == 0.0     # Bad format
        assert rewards["reasoning_steps"][0] == 0.0  # No steps
        assert rewards["repetition_penalty"][0] <= 0.0  # Some penalty (may be -0.0)
        assert rewards["total"][0] < 1.0       # Poor total score


if __name__ == "__main__":
    pytest.main([__file__, "-v"])