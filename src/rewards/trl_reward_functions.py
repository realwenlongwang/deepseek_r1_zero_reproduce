"""
TRL-compatible reward functions for GRPO training.
Creates wrapper functions that match TRL's expected interface.
"""

# Import our tutorial-based reward functions
from .tutorial_rewards import (
    accuracy_reward,
    format_reward,
    reasoning_steps_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward
)


def create_trl_wrapper(tutorial_func, need_solutions=False, wrapper_name=None):
    """Create a TRL-compatible wrapper for tutorial reward functions with intuitive naming."""
    def trl_wrapper(*args, **kwargs):
        """TRL-compatible wrapper function."""
        # Handle flexible argument patterns from TRL
        if len(args) >= 3:
            inputs, prompts, completions = args[:3]
        elif len(args) >= 1:
            # Sometimes TRL passes just completions
            completions = args[0]
            prompts = kwargs.get('prompts', [''] * len(completions))
            inputs = kwargs.get('inputs', [''] * len(completions))
        else:
            completions = kwargs.get('completions', [])
            prompts = kwargs.get('prompts', [''] * len(completions))
            inputs = kwargs.get('inputs', [''] * len(completions))
        
        # Convert completions to tutorial format
        formatted_completions = []
        for completion in completions:
            # Handle different types of completion data
            if isinstance(completion, str):
                formatted_completions.append([{"content": completion}])
            elif isinstance(completion, list):
                # If completion is already a list, extract the content
                content = completion[0] if len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
                if isinstance(content, dict):
                    formatted_completions.append([content])
                else:
                    formatted_completions.append([{"content": str(content)}])
            elif isinstance(completion, dict):
                formatted_completions.append([completion])
            else:
                formatted_completions.append([{"content": str(completion)}])
        
        if need_solutions:
            # For accuracy and cosine functions that need solutions
            # Use empty solutions as placeholder since we don't have reference answers in GRPO
            solutions = [""] * len(completions)
            return tutorial_func(formatted_completions, solutions)
        else:
            # For format, reasoning, and repetition functions
            return tutorial_func(formatted_completions)
    
    # Set intuitive function name for WandB metrics
    if wrapper_name:
        trl_wrapper.__name__ = wrapper_name
    
    return trl_wrapper


def create_cosine_wrapper(script_args):
    """Create TRL-compatible cosine reward wrapper with intuitive naming."""
    cosine_func = get_cosine_scaled_reward(
        min_value_wrong=script_args.cosine_min_value_wrong,
        max_value_wrong=script_args.cosine_max_value_wrong,
        min_value_correct=script_args.cosine_min_value_correct,
        max_value_correct=script_args.cosine_max_value_correct,
        max_len=script_args.cosine_max_len,
    )
    
    def cosine_length_reward(*args, **kwargs):
        """TRL-compatible cosine length scaling reward."""
        # Handle flexible argument patterns from TRL
        if len(args) >= 3:
            inputs, prompts, completions = args[:3]
        elif len(args) >= 1:
            completions = args[0]
        else:
            completions = kwargs.get('completions', [])
        
        formatted_completions = []
        for completion in completions:
            # Handle different types of completion data
            if isinstance(completion, str):
                formatted_completions.append([{"content": completion}])
            elif isinstance(completion, list):
                # If completion is already a list, extract the content
                content = completion[0] if len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
                if isinstance(content, dict):
                    formatted_completions.append([content])
                else:
                    formatted_completions.append([{"content": str(content)}])
            elif isinstance(completion, dict):
                formatted_completions.append([completion])
            else:
                formatted_completions.append([{"content": str(completion)}])
        
        # Since GRPO doesn't have reference solutions, use neutral accuracy
        solutions = [""] * len(completions)
        accuracy_rewards = [0.5] * len(completions)  # Neutral accuracy
        
        return cosine_func(formatted_completions, solutions, accuracy_rewards)
    
    # Set intuitive name for WandB metrics
    cosine_length_reward.__name__ = "cosine_length_reward"
    
    return cosine_length_reward


def create_repetition_wrapper(script_args):
    """Create TRL-compatible repetition penalty wrapper with intuitive naming."""
    repetition_func = get_repetition_penalty_reward(
        ngram_size=script_args.repetition_n_grams,
        max_penalty=script_args.repetition_max_penalty,
    )
    
    def repetition_penalty_reward(*args, **kwargs):
        """TRL-compatible repetition penalty reward."""
        # Handle flexible argument patterns from TRL
        if len(args) >= 3:
            inputs, prompts, completions = args[:3]
        elif len(args) >= 1:
            completions = args[0]
        else:
            completions = kwargs.get('completions', [])
        
        formatted_completions = []
        for completion in completions:
            # Handle different types of completion data
            if isinstance(completion, str):
                formatted_completions.append([{"content": completion}])
            elif isinstance(completion, list):
                # If completion is already a list, extract the content
                content = completion[0] if len(completion) > 0 and isinstance(completion[0], dict) else str(completion)
                if isinstance(content, dict):
                    formatted_completions.append([content])
                else:
                    formatted_completions.append([{"content": str(content)}])
            elif isinstance(completion, dict):
                formatted_completions.append([completion])
            else:
                formatted_completions.append([{"content": str(completion)}])
        
        return repetition_func(formatted_completions)
    
    # Set intuitive name for WandB metrics
    repetition_penalty_reward.__name__ = "repetition_penalty_reward"
    
    return repetition_penalty_reward


# Utility function to get reward functions based on script arguments
def get_reward_functions(script_args):
    """
    Returns a list of TRL-compatible reward functions with intuitive names for WandB metrics.
    """
    reward_funcs_list = []
    
    for func_name in script_args.reward_funcs:
        if func_name == "accuracy":
            # Accuracy function needs solutions but we use empty ones
            reward_funcs_list.append(create_trl_wrapper(
                accuracy_reward, 
                need_solutions=True, 
                wrapper_name="accuracy_reward"
            ))
        elif func_name == "format":
            reward_funcs_list.append(create_trl_wrapper(
                format_reward, 
                need_solutions=False, 
                wrapper_name="format_reward"
            ))
        elif func_name == "reasoning_steps":
            reward_funcs_list.append(create_trl_wrapper(
                reasoning_steps_reward, 
                need_solutions=False, 
                wrapper_name="reasoning_steps_reward"
            ))
        elif func_name == "cosine":
            reward_funcs_list.append(create_cosine_wrapper(script_args))
        elif func_name == "repetition_penalty":
            reward_funcs_list.append(create_repetition_wrapper(script_args))
        else:
            raise ValueError(f"Reward function '{func_name}' not found in registry.")
    
    return reward_funcs_list