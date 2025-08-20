#!/usr/bin/env python3
"""
Brute force solver for countdown number problems.
"""

import json
import itertools
from typing import List, Tuple, Optional, Set
from fractions import Fraction

class CountdownSolver:
    """Brute force solver for countdown number games."""
    
    def __init__(self):
        # Define the four basic operations
        self.operations = [
            ('+', lambda a, b: a + b),
            ('-', lambda a, b: a - b),
            ('*', lambda a, b: a * b),
            ('/', lambda a, b: a / b if b != 0 else None)
        ]
    
    def solve(self, target: int, numbers: List[int], tolerance: float = 1e-9) -> Optional[str]:
        """
        Find a solution to reach the target using the given numbers.
        
        Args:
            target: The target number to reach
            numbers: List of numbers to use
            tolerance: Tolerance for floating point comparison
            
        Returns:
            String representation of the solution if found, None otherwise
        """
        # Try all possible combinations of numbers (subsets)
        for r in range(1, len(numbers) + 1):
            for num_combo in itertools.combinations(numbers, r):
                # Try all permutations of the selected numbers
                for num_perm in itertools.permutations(num_combo):
                    solution = self._solve_with_numbers(target, list(num_perm), tolerance)
                    if solution:
                        return solution
        
        return None
    
    def _solve_with_numbers(self, target: int, numbers: List[int], tolerance: float) -> Optional[str]:
        """Recursively solve using the given numbers."""
        if len(numbers) == 1:
            # Base case: single number
            if abs(numbers[0] - target) < tolerance:
                return str(numbers[0])
            return None
        
        # Try all ways to split the numbers into two groups
        for i in range(1, len(numbers)):
            for left_nums in itertools.combinations(range(len(numbers)), i):
                right_nums = [j for j in range(len(numbers)) if j not in left_nums]
                
                left_numbers = [numbers[j] for j in left_nums]
                right_numbers = [numbers[j] for j in right_nums]
                
                # Try all operations between left and right groups
                for op_name, op_func in self.operations:
                    # Recursively solve left and right sides
                    left_solutions = self._get_all_solutions(left_numbers, tolerance)
                    right_solutions = self._get_all_solutions(right_numbers, tolerance)
                    
                    for left_val, left_expr in left_solutions:
                        for right_val, right_expr in right_solutions:
                            try:
                                result = op_func(left_val, right_val)
                                if result is not None and abs(result - target) < tolerance:
                                    # Format the expression with proper parentheses
                                    if len(left_numbers) > 1 and op_name in ['*', '/']:
                                        left_expr = f"({left_expr})"
                                    if len(right_numbers) > 1 and op_name in ['*', '/', '-']:
                                        right_expr = f"({right_expr})"
                                    
                                    return f"{left_expr} {op_name} {right_expr}"
                            except (ZeroDivisionError, OverflowError):
                                continue
        
        return None
    
    def _get_all_solutions(self, numbers: List[int], tolerance: float) -> List[Tuple[float, str]]:
        """Get all possible values and expressions from the given numbers."""
        if len(numbers) == 1:
            return [(float(numbers[0]), str(numbers[0]))]
        
        solutions = []
        
        # Try all ways to split the numbers into two groups
        for i in range(1, len(numbers)):
            for left_nums in itertools.combinations(range(len(numbers)), i):
                right_nums = [j for j in range(len(numbers)) if j not in left_nums]
                
                left_numbers = [numbers[j] for j in left_nums]
                right_numbers = [numbers[j] for j in right_nums]
                
                # Recursively get solutions for left and right
                left_solutions = self._get_all_solutions(left_numbers, tolerance)
                right_solutions = self._get_all_solutions(right_numbers, tolerance)
                
                # Try all operations between left and right
                for op_name, op_func in self.operations:
                    for left_val, left_expr in left_solutions:
                        for right_val, right_expr in right_solutions:
                            try:
                                result = op_func(left_val, right_val)
                                if result is not None:
                                    # Format the expression with proper parentheses
                                    if len(left_numbers) > 1 and op_name in ['*', '/']:
                                        formatted_left = f"({left_expr})"
                                    else:
                                        formatted_left = left_expr
                                    
                                    if len(right_numbers) > 1 and op_name in ['*', '/', '-']:
                                        formatted_right = f"({right_expr})"
                                    else:
                                        formatted_right = right_expr
                                    
                                    expression = f"{formatted_left} {op_name} {formatted_right}"
                                    solutions.append((result, expression))
                            except (ZeroDivisionError, OverflowError):
                                continue
        
        return solutions

def solve_countdown_problems(problems_file: str = "sampled_countdown_problems.json") -> dict:
    """Solve all problems in the given file and return results."""
    # Load problems
    with open(problems_file, 'r') as f:
        problems = json.load(f)
    
    solver = CountdownSolver()
    results = {
        "total_problems": len(problems),
        "solved": 0,
        "unsolved": 0,
        "solutions": [],
        "unsolved_problems": []
    }
    
    print(f"Solving {len(problems)} countdown problems...")
    
    for i, problem in enumerate(problems):
        target = problem["target"]
        numbers = problem["nums"]
        
        print(f"Problem {i+1}/{len(problems)}: Target={target}, Numbers={numbers}")
        
        solution = solver.solve(target, numbers)
        
        if solution:
            results["solved"] += 1
            results["solutions"].append({
                "problem_index": problem["index"],
                "target": target,
                "numbers": numbers,
                "solution": solution
            })
            print(f"  ✓ Solution: {solution}")
        else:
            results["unsolved"] += 1
            results["unsolved_problems"].append({
                "problem_index": problem["index"],
                "target": target,
                "numbers": numbers
            })
            print(f"  ✗ No solution found")
    
    # Calculate solve rate
    results["solve_rate"] = results["solved"] / results["total_problems"] if results["total_problems"] > 0 else 0
    
    return results

if __name__ == "__main__":
    # Solve the sampled problems
    results = solve_countdown_problems()
    
    # Save results
    with open("countdown_solver_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== RESULTS ===")
    print(f"Total problems: {results['total_problems']}")
    print(f"Solved: {results['solved']}")
    print(f"Unsolved: {results['unsolved']}")
    print(f"Solve rate: {results['solve_rate']:.2%}")
    
    print(f"\nResults saved to countdown_solver_results.json")