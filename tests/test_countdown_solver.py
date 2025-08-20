#!/usr/bin/env python3
"""
Test the countdown solver functionality.
"""

import pytest
from countdown_solver import CountdownSolver

def test_countdown_solver_basic():
    """Test basic countdown solver functionality."""
    solver = CountdownSolver()
    
    # Test simple cases
    assert solver.solve(5, [2, 3]) == "2 + 3"
    assert solver.solve(6, [2, 3]) == "2 * 3"
    
    # Test trivial case
    assert solver.solve(5, [1, 5, 10]) == "5"

def test_countdown_solver_complex():
    """Test complex countdown problems."""
    solver = CountdownSolver()
    
    # Test 3-operation case
    solution = solver.solve(94, [15, 9, 25, 51])
    assert solution is not None
    assert "+" in solution and "*" in solution and "/" in solution
    
    # Test 2-operation case  
    solution = solver.solve(86, [14, 84, 7])
    assert solution is not None
    
def test_countdown_solver_impossible():
    """Test cases that should be impossible (though dataset has 100% solvability)."""
    solver = CountdownSolver()
    
    # This should be solvable with the countdown dataset design
    # but testing the solver's handling of edge cases
    solution = solver.solve(1000000, [1, 2])
    # This might not have a solution with basic operations
    # The solver should return None for truly impossible cases

if __name__ == "__main__":
    pytest.main([__file__])