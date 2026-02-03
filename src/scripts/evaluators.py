"""
Evaluators for WoRMS Agent
Checks: plan adherence, query classification, tool usage
"""

from typing import Dict, Any, Optional
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example


def plan_adherence_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Check if agent called the planned tools
    
    Scores:
    - 1.0 = Called all planned tools
    - 0.0 = Called none of the planned tools
    - Partial score in between
    """
    try:
        # Get planned tools from metadata
        metadata = run.metadata if hasattr(run, 'metadata') and run.metadata else {}
        planned_tools = set(metadata.get("planned_tools", []))
        
        # Get actually called tools from child runs
        called_tools = set()
        if run.child_runs:
            for child in run.child_runs:
                if child.run_type == "tool":
                    called_tools.add(child.name)
        
        # Calculate score
        if not planned_tools:
            score = 1.0  # No plan, so no violation
        else:
            overlap = len(planned_tools & called_tools)
            score = overlap / len(planned_tools)
        
        return {
            "key": "plan_adherence",
            "score": score,
            "comment": f"Called {len(called_tools & planned_tools)}/{len(planned_tools)} planned tools",
            "metadata": {
                "planned": list(planned_tools),
                "called": list(called_tools),
                "missing": list(planned_tools - called_tools),
                "extra": list(called_tools - planned_tools)
            }
        }
    except Exception as e:
        return {
            "key": "plan_adherence",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def query_classification_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Check if agent correctly classified the query type
    
    Compares expected_query_type from ground truth to actual
    """
    try:
        # Expected from ground truth
        expected_type = example.outputs.get("expected_query_type")
        
        # Actual from agent metadata
        metadata = run.metadata if hasattr(run, 'metadata') and run.metadata else {}
        actual_type = metadata.get("query_type")
        
        # Score
        score = 1.0 if actual_type == expected_type else 0.0
        
        return {
            "key": "query_classification",
            "score": score,
            "comment": f"Expected: {expected_type}, Got: {actual_type}",
            "metadata": {
                "expected": expected_type,
                "actual": actual_type
            }
        }
    except Exception as e:
        return {
            "key": "query_classification",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def species_count_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Check if agent identified correct number of species
    """
    try:
        expected_count = example.outputs.get("expected_species_count")
        
        metadata = run.metadata if hasattr(run, 'metadata') and run.metadata else {}
        actual_count = metadata.get("species_count")
        
        score = 1.0 if actual_count == expected_count else 0.0
        
        return {
            "key": "species_count",
            "score": score,
            "comment": f"Expected: {expected_count}, Got: {actual_count}",
            "metadata": {
                "expected": expected_count,
                "actual": actual_count
            }
        }
    except Exception as e:
        return {
            "key": "species_count",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def tool_selection_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Check if agent selected the right tools for the query
    """
    try:
        expected_tools = set(example.outputs.get("expected_tools", []))
        
        # Get planned tools
        metadata = run.metadata if hasattr(run, 'metadata') and run.metadata else {}
        planned_tools = set(metadata.get("planned_tools", []))
        
        # Score based on overlap
        if not expected_tools:
            score = 1.0
        else:
            overlap = len(expected_tools & planned_tools)
            score = overlap / len(expected_tools)
        
        return {
            "key": "tool_selection",
            "score": score,
            "comment": f"Selected {len(expected_tools & planned_tools)}/{len(expected_tools)} expected tools",
            "metadata": {
                "expected": list(expected_tools),
                "planned": list(planned_tools),
                "correct": list(expected_tools & planned_tools),
                "incorrect": list(planned_tools - expected_tools)
            }
        }
    except Exception as e:
        return {
            "key": "tool_selection",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


def execution_success_evaluator(run: Run, example: Example) -> Dict[str, Any]:
    """
    Check if agent completed without errors
    """
    try:
        # Check for errors in run
        has_error = run.error is not None
        
        # Check if it's expected to fail
        should_fail = example.outputs.get("should_fail", False)
        
        # Score logic
        if should_fail:
            # Should fail - success if it did fail
            score = 1.0 if has_error else 0.0
            comment = "Correctly failed" if has_error else "Should have failed but didn't"
        else:
            # Should succeed - success if no error
            score = 0.0 if has_error else 1.0
            comment = "Completed successfully" if not has_error else f"Failed with error: {run.error}"
        
        return {
            "key": "execution_success",
            "score": score,
            "comment": comment,
            "metadata": {
                "has_error": has_error,
                "should_fail": should_fail
            }
        }
    except Exception as e:
        return {
            "key": "execution_success",
            "score": 0.0,
            "comment": f"Evaluation error: {str(e)}"
        }


# List of all evaluators
ALL_EVALUATORS = [
    plan_adherence_evaluator,
    query_classification_evaluator,
    species_count_evaluator,
    tool_selection_evaluator,
    execution_success_evaluator
]


if __name__ == "__main__":
    print("Available Evaluators:")
    print("=" * 60)
    for evaluator in ALL_EVALUATORS:
        print(f"- {evaluator.__name__}")
        print(f"  {evaluator.__doc__.strip()}")
        print()