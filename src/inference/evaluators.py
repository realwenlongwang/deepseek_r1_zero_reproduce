#!/usr/bin/env python3
"""
Response evaluation and validation for DeepSeek R1 Zero models.
Checks format compliance and provides basic scoring metrics.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of response validation."""
    is_valid_format: bool
    has_thinking: bool
    has_answer: bool
    thinking_content: str
    answer_content: str
    reasoning_indicators: int
    format_score: float
    issues: List[str]


class FormatChecker:
    """
    Checks if model responses follow the expected DeepSeek R1 format.
    Expected format: <think>...</think><answer>...</answer>
    """
    
    def __init__(self):
        self.think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
        self.reasoning_pattern = re.compile(
            r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,|Let me|I need to|Therefore)",
            re.MULTILINE
        )
    
    def check_format(self, response: str) -> ValidationResult:
        """
        Check if response follows expected format.
        
        Args:
            response: Generated response text
            
        Returns:
            ValidationResult with detailed analysis
        """
        issues = []
        
        # Check for thinking section
        think_match = self.think_pattern.search(response)
        has_thinking = think_match is not None
        thinking_content = think_match.group(1).strip() if think_match else ""
        
        # Check for answer section
        answer_match = self.answer_pattern.search(response)
        has_answer = answer_match is not None
        answer_content = answer_match.group(1).strip() if answer_match else ""
        
        # Check format validity
        is_valid_format = has_thinking and has_answer
        
        if not has_thinking:
            issues.append("Missing <think>...</think> section")
        if not has_answer:
            issues.append("Missing <answer>...</answer> section")
        
        # Count reasoning indicators
        reasoning_indicators = len(self.reasoning_pattern.findall(response))
        
        # Calculate format score
        format_score = self._calculate_format_score(
            has_thinking, has_answer, thinking_content, answer_content, reasoning_indicators
        )
        
        # Additional quality checks
        if has_thinking and len(thinking_content) < 10:
            issues.append("Thinking section too short")
        if has_answer and len(answer_content) < 5:
            issues.append("Answer section too short")
        if reasoning_indicators == 0 and has_thinking:
            issues.append("No clear reasoning structure in thinking section")
        
        # Check for garbage patterns
        if self._check_garbage_patterns(response):
            issues.append("Contains repetitive/garbage patterns")
        
        return ValidationResult(
            is_valid_format=is_valid_format,
            has_thinking=has_thinking,
            has_answer=has_answer,
            thinking_content=thinking_content,
            answer_content=answer_content,
            reasoning_indicators=reasoning_indicators,
            format_score=format_score,
            issues=issues
        )
    
    def _calculate_format_score(self, 
                               has_thinking: bool,
                               has_answer: bool,
                               thinking_content: str,
                               answer_content: str,
                               reasoning_indicators: int) -> float:
        """Calculate a format compliance score (0-1)."""
        score = 0.0
        
        # Basic structure (60% of score)
        if has_thinking:
            score += 0.3
        if has_answer:
            score += 0.3
        
        # Content quality (30% of score)
        if thinking_content and len(thinking_content) > 20:
            score += 0.1
        if answer_content and len(answer_content) > 5:
            score += 0.1
        if reasoning_indicators > 0:
            score += min(0.1, reasoning_indicators * 0.02)
        
        # Bonus for good structure (10% of score)
        if has_thinking and has_answer and reasoning_indicators >= 2:
            score += 0.1
        
        return min(1.0, score)
    
    def _check_garbage_patterns(self, response: str) -> bool:
        """Check for common garbage patterns in generated text."""
        # Check for excessive repetition
        if "-t-t-t" in response or response.count("-t") > 10:
            return True
        
        # Check for excessive repeated characters
        repetitive_patterns = [
            r'(.)\1{10,}',  # Same character repeated 10+ times
            r'(\w+)\s+\1\s+\1',  # Same word repeated 3+ times
        ]
        
        for pattern in repetitive_patterns:
            if re.search(pattern, response):
                return True
        
        return False


class ResponseValidator:
    """
    Comprehensive response validator with multiple quality metrics.
    """
    
    def __init__(self):
        self.format_checker = FormatChecker()
    
    def validate_response(self, 
                         response: str,
                         expected_answer: str = None,
                         problem_type: str = "general") -> Dict[str, Any]:
        """
        Comprehensive validation of model response.
        
        Args:
            response: Generated response
            expected_answer: Expected answer for comparison (optional)
            problem_type: Type of problem ("math", "countdown", "general")
            
        Returns:
            Validation results dictionary
        """
        # Format checking
        format_result = self.format_checker.check_format(response)
        
        # Basic metrics
        metrics = {
            "format_valid": format_result.is_valid_format,
            "format_score": format_result.format_score,
            "has_thinking": format_result.has_thinking,
            "has_answer": format_result.has_answer,
            "thinking_length": len(format_result.thinking_content),
            "answer_length": len(format_result.answer_content),
            "reasoning_indicators": format_result.reasoning_indicators,
            "issues": format_result.issues
        }
        
        # Content analysis
        if format_result.thinking_content:
            metrics["thinking_analysis"] = self._analyze_thinking_content(
                format_result.thinking_content, problem_type
            )
        
        if format_result.answer_content:
            metrics["answer_analysis"] = self._analyze_answer_content(
                format_result.answer_content, expected_answer, problem_type
            )
        
        # Overall quality score
        metrics["quality_score"] = self._calculate_quality_score(metrics, format_result)
        
        return metrics
    
    def _analyze_thinking_content(self, thinking: str, problem_type: str) -> Dict[str, Any]:
        """Analyze the quality of thinking content."""
        analysis = {
            "word_count": len(thinking.split()),
            "has_step_by_step": bool(re.search(r'step \d+|first|second|next|then|finally', thinking, re.IGNORECASE)),
            "has_mathematical_notation": bool(re.search(r'[+\-*/=<>]|\d+', thinking)),
            "has_logical_connectors": bool(re.search(r'because|therefore|since|thus|however|but|so', thinking, re.IGNORECASE))
        }
        
        # Problem-specific analysis
        if problem_type == "math":
            analysis["has_equations"] = bool(re.search(r'=', thinking))
            analysis["has_calculations"] = bool(re.search(r'\d+\s*[+\-*/]\s*\d+', thinking))
        elif problem_type == "countdown":
            analysis["mentions_target"] = "target" in thinking.lower()
            analysis["mentions_operations"] = bool(re.search(r'add|subtract|multiply|divide|\+|\-|\*|\/', thinking, re.IGNORECASE))
        
        return analysis
    
    def _analyze_answer_content(self, answer: str, expected: str, problem_type: str) -> Dict[str, Any]:
        """Analyze the quality of answer content."""
        analysis = {
            "word_count": len(answer.split()),
            "is_numeric": bool(re.search(r'^\s*-?\d+\.?\d*\s*$', answer.strip())),
            "has_explanation": len(answer.split()) > 3
        }
        
        # Compare with expected answer if provided
        if expected:
            analysis["matches_expected"] = self._compare_answers(answer, expected, problem_type)
            analysis["similarity_score"] = self._calculate_similarity(answer, expected)
        
        return analysis
    
    def _compare_answers(self, answer: str, expected: str, problem_type: str) -> bool:
        """Compare generated answer with expected answer."""
        # Normalize both answers
        answer_norm = re.sub(r'\s+', ' ', answer.lower().strip())
        expected_norm = re.sub(r'\s+', ' ', expected.lower().strip())
        
        # Exact match
        if answer_norm == expected_norm:
            return True
        
        # For numeric answers, extract and compare numbers
        if problem_type in ["math", "countdown"]:
            answer_nums = re.findall(r'-?\d+\.?\d*', answer)
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            
            if answer_nums and expected_nums:
                try:
                    return float(answer_nums[0]) == float(expected_nums[0])
                except ValueError:
                    pass
        
        # Partial match
        return expected_norm in answer_norm or answer_norm in expected_norm
    
    def _calculate_similarity(self, answer: str, expected: str) -> float:
        """Calculate similarity score between answers."""
        # Simple word overlap similarity
        answer_words = set(answer.lower().split())
        expected_words = set(expected.lower().split())
        
        if not answer_words or not expected_words:
            return 0.0
        
        intersection = answer_words.intersection(expected_words)
        union = answer_words.union(expected_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_quality_score(self, metrics: Dict[str, Any], format_result: ValidationResult) -> float:
        """Calculate overall quality score."""
        score = format_result.format_score * 0.6  # Format is 60% of score
        
        # Content quality (40% of score)
        if "thinking_analysis" in metrics:
            thinking = metrics["thinking_analysis"]
            if thinking["has_step_by_step"]:
                score += 0.1
            if thinking["word_count"] > 20:
                score += 0.1
            if thinking["has_logical_connectors"]:
                score += 0.05
        
        if "answer_analysis" in metrics:
            answer = metrics["answer_analysis"]
            if answer["word_count"] > 0:
                score += 0.1
            if "matches_expected" in answer and answer["matches_expected"]:
                score += 0.05
        
        return min(1.0, score)
    
    def validate_batch(self, 
                      responses: List[str],
                      expected_answers: List[str] = None,
                      problem_type: str = "general") -> Dict[str, Any]:
        """
        Validate a batch of responses.
        
        Args:
            responses: List of generated responses
            expected_answers: List of expected answers (optional)
            problem_type: Type of problems
            
        Returns:
            Batch validation results
        """
        if expected_answers is None:
            expected_answers = [None] * len(responses)
        
        individual_results = []
        for i, (response, expected) in enumerate(zip(responses, expected_answers)):
            result = self.validate_response(response, expected, problem_type)
            result["index"] = i
            individual_results.append(result)
        
        # Calculate batch statistics
        batch_stats = {
            "total_responses": len(responses),
            "valid_format_count": sum(1 for r in individual_results if r["format_valid"]),
            "avg_format_score": sum(r["format_score"] for r in individual_results) / len(individual_results),
            "avg_quality_score": sum(r["quality_score"] for r in individual_results) / len(individual_results),
            "thinking_present_count": sum(1 for r in individual_results if r["has_thinking"]),
            "answer_present_count": sum(1 for r in individual_results if r["has_answer"]),
            "individual_results": individual_results
        }
        
        if expected_answers and any(expected_answers):
            correct_count = sum(1 for r in individual_results 
                              if "answer_analysis" in r and 
                              r["answer_analysis"].get("matches_expected", False))
            batch_stats["accuracy"] = correct_count / len(responses)
        
        return batch_stats