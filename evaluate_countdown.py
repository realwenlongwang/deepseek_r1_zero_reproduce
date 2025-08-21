#!/usr/bin/env python3
"""
Countdown Dataset Evaluation Function
Generate prompts from countdown dataset and evaluate model correctness.
"""

import os
import sys
import json
import random
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datasets import load_dataset
from src.data.dataset import format_countdown_problem
from tests.countdown_solver import CountdownSolver


class CountdownEvaluator:
    """Evaluates models on the countdown dataset."""
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        """Initialize evaluator with optional model loading."""
        self.solver = CountdownSolver()
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.engine = None
        
        # Load model if checkpoint provided
        if checkpoint_path:
            self._load_model()
    
    def _load_model(self):
        """Load model and create inference engine."""
        from src.inference import AutoCheckpointLoader, InferenceEngine
        from src.inference.checkpoint_loader import find_latest_checkpoint
        
        # Use provided checkpoint or find latest
        if self.checkpoint_path is None:
            self.checkpoint_path = find_latest_checkpoint("saved_models")
            if self.checkpoint_path is None:
                raise ValueError("No checkpoint found. Specify checkpoint_path or run training first.")
        
        print(f"Loading model from: {self.checkpoint_path}")
        
        # Load model
        loader = AutoCheckpointLoader(device_map=self.device)
        self.model, self.tokenizer, metadata = loader.load_checkpoint(self.checkpoint_path)
        
        # Create inference engine
        self.engine = InferenceEngine(self.model, self.tokenizer, self.device)
        
        print(f"Model loaded successfully: {metadata.get('model_name', 'Unknown')}")
    
    def load_countdown_samples(
        self, 
        num_samples: int = 100, 
        seed: int = 42, 
        test_split: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Load countdown dataset samples.
        
        Args:
            num_samples: Number of samples to load
            seed: Random seed for reproducibility
            test_split: If True, use test split logic from inference.py
            
        Returns:
            List of countdown problem samples
        """
        dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
        
        if test_split:
            # Use same test split logic as inference.py
            random.seed(seed)
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            
            # Create 10% test split
            test_size = int(len(dataset) * 0.1)
            train_split_point = len(dataset) - test_size
            test_indices = indices[train_split_point:]
            
            # Get requested samples from test split
            sample_indices = test_indices[:num_samples]
        else:
            # Random sampling from entire dataset
            random.seed(seed)
            sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        
        samples = []
        for i, idx in enumerate(sample_indices):
            sample = dataset[idx]
            problem_text = format_countdown_problem(sample["target"], sample["nums"])
            
            samples.append({
                "index": i,
                "dataset_index": idx,
                "target": sample["target"],
                "nums": sample["nums"],
                "problem_text": problem_text,
                "prompt": problem_text
            })
        
        return samples
    
    def extract_answer_from_response(self, response: str) -> Optional[str]:
        """Extract answer from model response."""
        # Look for <answer> tags first
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fall back to looking for mathematical expressions at the end
        lines = response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            # Look for lines containing mathematical operations and equals sign
            if '=' in line and any(op in line for op in ['+', '-', '*', '/', '×', '÷']):
                return line
        
        return None
    
    def validate_answer(self, answer_text: str, target: int, nums: List[int]) -> Dict[str, Any]:
        """
        Validate if the answer is mathematically correct.
        
        Args:
            answer_text: The extracted answer from model response
            target: Expected target value
            nums: Available numbers
            
        Returns:
            Dictionary with validation results
        """
        if not answer_text:
            return {
                "is_correct": False,
                "error": "No answer found in response",
                "expected_target": target,
                "calculated_result": None
            }
        
        try:
            # Try to evaluate the mathematical expression
            # Replace × and ÷ with * and /
            clean_expr = answer_text.replace('×', '*').replace('÷', '/')
            
            # Extract just the mathematical part (before =)
            if '=' in clean_expr:
                expr_part = clean_expr.split('=')[0].strip()
            else:
                expr_part = clean_expr.strip()
            
            # Evaluate the expression
            try:
                calculated_result = eval(expr_part)
                result_matches = abs(calculated_result - target) < 1e-6
                
                # Also verify using the brute force solver
                solver_solution = self.solver.solve(target, nums)
                solver_validates = solver_solution is not None
                
                return {
                    "is_correct": result_matches and solver_validates,
                    "calculated_result": calculated_result,
                    "expected_target": target,
                    "result_matches_target": result_matches,
                    "solver_found_solution": solver_validates,
                    "solver_solution": solver_solution,
                    "extracted_expression": expr_part
                }
                
            except Exception as eval_error:
                return {
                    "is_correct": False,
                    "error": f"Failed to evaluate expression: {eval_error}",
                    "expected_target": target,
                    "calculated_result": None,
                    "extracted_expression": expr_part
                }
                
        except Exception as e:
            return {
                "is_correct": False,
                "error": f"Validation error: {str(e)}",
                "expected_target": target,
                "calculated_result": None
            }
    
    def evaluate_samples(
        self, 
        samples: List[Dict[str, Any]], 
        generation_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on countdown samples.
        
        Args:
            samples: List of countdown samples to evaluate
            generation_params: Optional generation parameters
            
        Returns:
            Complete evaluation results
        """
        if self.engine is None:
            raise ValueError("Model not loaded. Provide checkpoint_path during initialization.")
        
        # Default generation parameters optimized for reasoning
        if generation_params is None:
            generation_params = {
                "max_new_tokens": 1024,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "use_system_message": True
            }
        
        results = []
        correct_count = 0
        
        print(f"Evaluating {len(samples)} countdown problems...")
        
        for i, sample in enumerate(samples, 1):
            print(f"\nProblem {i}/{len(samples)}: Target={sample['target']}, Numbers={sample['nums']}")
            
            try:
                # Generate response
                response = self.engine.generate(sample["prompt"], **generation_params)
                
                # Extract answer
                extracted_answer = self.extract_answer_from_response(response)
                print(f"  Extracted answer: {extracted_answer}")
                
                # Validate correctness
                validation = self.validate_answer(extracted_answer, sample["target"], sample["nums"])
                is_correct = validation.get("is_correct", False)
                
                if is_correct:
                    correct_count += 1
                    print(f"  ✅ Correct")
                else:
                    print(f"  ❌ Incorrect: {validation.get('error', 'Wrong calculation')}")
                
                # Store complete results
                result = {
                    "sample_index": sample["index"],
                    "dataset_index": sample["dataset_index"],
                    "target": sample["target"],
                    "nums": sample["nums"],
                    "prompt": sample["prompt"],
                    "response": response,
                    "extracted_answer": extracted_answer,
                    "validation": validation,
                    "is_correct": is_correct
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
                results.append({
                    "sample_index": sample["index"],
                    "dataset_index": sample["dataset_index"],
                    "target": sample["target"],
                    "nums": sample["nums"],
                    "prompt": sample["prompt"],
                    "response": f"[Generation failed: {str(e)}]",
                    "extracted_answer": None,
                    "validation": {"is_correct": False, "error": str(e)},
                    "is_correct": False,
                    "error": str(e)
                })
        
        # Calculate summary statistics
        accuracy = correct_count / len(samples) if samples else 0
        
        summary = {
            "total_samples": len(samples),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "checkpoint_path": self.checkpoint_path,
            "generation_params": generation_params,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Total samples: {summary['total_samples']}")
        print(f"Correct answers: {summary['correct_answers']}")
        print(f"Accuracy: {summary['accuracy']:.2%}")
        
        return {
            "summary": summary,
            "individual_results": results
        }
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save evaluation results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_file}")


def evaluate_countdown_dataset(
    checkpoint_path: Optional[str] = None,
    num_samples: int = 100,
    output_file: Optional[str] = None,
    seed: int = 42,
    generation_params: Optional[Dict] = None,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Complete countdown dataset evaluation pipeline.
    
    Args:
        checkpoint_path: Path to model checkpoint (None for latest)
        num_samples: Number of samples to evaluate
        output_file: Output JSON file path (None for auto-generated)
        seed: Random seed for reproducibility
        generation_params: Optional generation parameters
        device: Device for inference
        
    Returns:
        Evaluation results dictionary
    """
    # Create evaluator
    evaluator = CountdownEvaluator(checkpoint_path, device)
    
    # Load samples
    samples = evaluator.load_countdown_samples(num_samples, seed, test_split=True)
    
    # Run evaluation
    results = evaluator.evaluate_samples(samples, generation_params)
    
    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"countdown_evaluation_{timestamp}.json"
    
    # Save results
    evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate model on countdown dataset")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device for inference")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens")
    
    args = parser.parse_args()
    
    generation_params = {
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "use_system_message": True
    }
    
    # Run evaluation
    results = evaluate_countdown_dataset(
        checkpoint_path=args.checkpoint,
        num_samples=args.samples,
        output_file=args.output,
        seed=args.seed,
        generation_params=generation_params,
        device=args.device
    )