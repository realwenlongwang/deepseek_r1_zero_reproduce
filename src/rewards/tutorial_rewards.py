"""
Reward functions based on the DeepSeek R1 training tutorial.
These implementations follow the exact patterns from the tutorial.
"""

import re
import math
from typing import List, Dict, Any, Callable
from collections import Counter

# Try to import latex2sympy2 and math_verify, provide fallbacks if not available
try:
    from latex2sympy2 import parse, LatexExtractionConfig, NormalizationConfig
    from math_verify import verify
    LATEX_PARSING_AVAILABLE = True
except ImportError:
    print("Warning: latex2sympy2 or math_verify not available. Using fallback implementation.")
    LATEX_PARSING_AVAILABLE = False
    
    # Fallback implementations
    def parse(*args, **kwargs):
        return None
    
    def verify(answer, gold):
        return False
    
    class LatexExtractionConfig:
        def __init__(self, **kwargs):
            pass
    
    class NormalizationConfig:
        def __init__(self, **kwargs):
            pass


def accuracy_reward(completions, solutions, **kwargs):
    """
    Reward function to check if the model's response is mathematically 
    equivalent to the ground truth solution.
    Uses latex2sympy2 for parsing and math_verify for validation.
    """
    
    # Extract responses
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solutions):
        if LATEX_PARSING_AVAILABLE:
            # Parse the ground truth solution
            gold_parsed = parse(sol, extraction_mode="first_match", 
                                extraction_config=[LatexExtractionConfig()])
            
            if gold_parsed:  # Check if parsing was successful
                # Parse the model's answer with relaxed normalization
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )

                # Reward 1.0 if correct, 0.0 if incorrect
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If ground truth cannot be parsed, assign neutral reward (0.5)
                reward = 0.5
                print("Warning: Failed to parse gold solution:", sol)
        else:
            # Fallback: simple text/numerical comparison
            reward = _fallback_accuracy_check(content, sol)

        rewards.append(reward)
    
    return rewards


def _fallback_accuracy_check(content: str, solution: str) -> float:
    """Fallback accuracy check when latex2sympy2 is not available."""
    try:
        # Extract answer from <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
        if not answer_match:
            return 0.0
        
        model_answer = answer_match.group(1).strip()
        
        # Extract numbers for comparison
        model_numbers = re.findall(r'-?\d+\.?\d*', model_answer)
        solution_numbers = re.findall(r'-?\d+\.?\d*', solution)
        
        if model_numbers and solution_numbers:
            try:
                model_num = float(model_numbers[-1])
                solution_num = float(solution_numbers[-1])
                return 1.0 if abs(model_num - solution_num) < 1e-6 else 0.0
            except ValueError:
                pass
        
        # Fallback to text comparison
        return 1.0 if model_answer.lower().strip() == solution.lower().strip() else 0.0
    except Exception:
        return 0.5


# Implement Format Reward Function
def format_reward(completions, **kwargs):
  """
  Reward function to check if the completion has the correct format:
  <think>...</think> <answer>...</answer>.
  """
  # Define the regex pattern for the desired format
  pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"

  # Extract the content from each completion
  completion_contents = [completion[0]["content"] for completion in completions]

  # Check if each completion matches the pattern
  matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE)
             for content in completion_contents]

  # Reward 1.0 for correct format, 0.0 otherwise
  return [1.0 if match else 0.0 for match in matches]


def reasoning_steps_reward(completions, **kwargs):
    r"""
    Reward function to encourage clear step-by-step reasoning.
    It looks for patterns like "Step 1:", numbered lists, bullet points,
    and transition words.
    """
    # Regex pattern to find indicators of reasoning steps
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"

    # Extract completion contents
    completion_contents = [completion[0]["content"] for completion in completions]

    # Count the number of reasoning step indicators in each completion
    matches = [len(re.findall(pattern, content, re.MULTILINE))
               for content in completion_contents]

    # Reward is proportional to the number of reasoning steps, maxing out at 1.0
    # We're using a "magic number" 3 here - encourage at least 3 steps for full reward
    return [min(1.0, count / 3) for count in matches]


# Implement Cosine Scaled Reward Function
def get_cosine_scaled_reward(
    min_value_wrong: float = -0.5,
    max_value_wrong: float = -0.1,
    min_value_correct: float = 0.8,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """
    Returns a cosine scaled reward function. This function scales the accuracy reward
    based on completion length. Shorter correct solutions get higher rewards,
    longer incorrect solutions get less penalty.
    """
    def cosine_scaled_reward(completions, solutions, accuracy_rewards, **kwargs):
        """
        Cosine scaled reward function that adjusts accuracy rewards based on completion length.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol, acc_reward in zip(contents, solutions, accuracy_rewards):
            gen_len = len(content)  # Length of the generated answer
            progress = gen_len / max_len # How far we are to max length
            cosine = math.cos(progress * math.pi) # Cosine value based on progress

            if acc_reward > 0.5: # Assuming accuracy_reward gives ~1.0 for correct answers
                min_value = min_value_correct
                max_value = max_value_correct
            else: # Incorrect answer
                min_value = max_value_wrong  # Note the swap!
                max_value = min_value_wrong

            # Cosine scaling formula!
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))
        return rewards
    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int = 3, max_penalty: float = -0.1):
    """
    Returns a repetition penalty reward function. Penalizes repetitions of n-grams
    in the generated text.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        """Helper function to generate n-grams from text."""
        words = text.lower().split() # Lowercase and split into words
        return zip(*[words[i:] for i in range(ngram_size)]) # Create n-grams

    def repetition_penalty_reward(completions, **kwargs) -> List[float]:
        """
        Repetition penalty reward function.
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "": # No penalty for empty completions
                rewards.append(0.0)
                continue
            if len(completion.split()) < ngram_size: # No penalty for short completions
                rewards.append(0.0)
                continue

            ngrams = set() # Use a set to store unique n-grams
            total = 0
            for ng in zipngram(completion, ngram_size): # Generate n-grams
                ngrams.add(ng) # Add n-gram to the set (duplicates are ignored)
                total += 1 # Count total n-grams

            # Calculate scaling factor: more repetition -> higher scaling
            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty # Apply penalty based on scaling
            rewards.append(reward)
        return rewards
    return repetition_penalty_reward


class TutorialRewardSystem:
    """
    Reward system following the exact tutorial implementation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration matching tutorial format."""
        default_config = {
            "reward_funcs": ["accuracy", "format", "reasoning_steps", "cosine", "repetition_penalty"],
            "cosine_min_value_wrong": -0.5,
            "cosine_max_value_wrong": -0.1,
            "cosine_min_value_correct": 0.8,
            "cosine_max_value_correct": 1.0,
            "cosine_max_len": 1000,
            "repetition_ngram_size": 3,
            "repetition_max_penalty": -0.1
        }
        
        if config:
            # Merge user config with defaults
            self.config = default_config.copy()
            self.config.update(config)
        else:
            self.config = default_config
        
        # Initialize reward functions
        self.reward_functions = {}
        
        if "accuracy" in self.config["reward_funcs"]:
            self.reward_functions["accuracy"] = accuracy_reward
            
        if "format" in self.config["reward_funcs"]:
            self.reward_functions["format"] = format_reward
            
        if "reasoning_steps" in self.config["reward_funcs"]:
            self.reward_functions["reasoning_steps"] = reasoning_steps_reward
            
        if "cosine" in self.config["reward_funcs"]:
            self.reward_functions["cosine"] = get_cosine_scaled_reward(
                min_value_wrong=self.config["cosine_min_value_wrong"],
                max_value_wrong=self.config["cosine_max_value_wrong"],
                min_value_correct=self.config["cosine_min_value_correct"],
                max_value_correct=self.config["cosine_max_value_correct"],
                max_len=self.config["cosine_max_len"]
            )
            
        if "repetition_penalty" in self.config["reward_funcs"]:
            self.reward_functions["repetition_penalty"] = get_repetition_penalty_reward(
                ngram_size=self.config["repetition_ngram_size"],
                max_penalty=self.config["repetition_max_penalty"]
            )
    
    def compute_rewards(
        self, 
        completions: List[List[Dict]], 
        solutions: List[str] = None,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Compute all configured reward functions.
        
        Args:
            completions: List of completions in tutorial format
            solutions: List of ground truth solutions
            
        Returns:
            Dictionary with reward function names as keys and reward lists as values
        """
        rewards = {}
        
        # Compute accuracy first if needed (required for cosine scaling)
        if "accuracy" in self.reward_functions:
            accuracy_rewards = self.reward_functions["accuracy"](
                completions, solutions or [""] * len(completions), **kwargs
            )
            rewards["accuracy"] = accuracy_rewards
        else:
            accuracy_rewards = [0.5] * len(completions)  # Neutral default
        
        # Compute other rewards
        for name, func in self.reward_functions.items():
            if name == "accuracy":
                continue  # Already computed
                
            if name == "cosine":
                # Cosine scaling needs accuracy rewards
                reward_values = func(
                    completions, solutions or [""] * len(completions), 
                    accuracy_rewards, **kwargs
                )
            else:
                reward_values = func(completions, **kwargs)
                
            rewards[name] = reward_values
        
        # Compute total reward as sum of all components
        if rewards:
            total_rewards = []
            for i in range(len(completions)):
                total = sum(rewards[name][i] for name in rewards)
                total_rewards.append(total)
            rewards["total"] = total_rewards
        
        return rewards


def create_tutorial_reward_system(config: Dict[str, Any] = None) -> TutorialRewardSystem:
    """Create a reward system following the tutorial implementation."""
    return TutorialRewardSystem(config)