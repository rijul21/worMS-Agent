"""
Run evaluations on WoRMS Agent
Uses dataset and evaluators to measure agent performance
"""

from langsmith import Client
from langsmith.evaluation import evaluate
from evaluators import ALL_EVALUATORS
from datetime import datetime


client = Client()
DATASET_NAME = "worms-agent-eval"


def run_evaluation():
    """
    Run evaluation on the test dataset
    """
    print("=" * 60)
    print("Running WoRMS Agent Evaluation")
    print("=" * 60)
    
    # Load dataset
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Loaded dataset: {DATASET_NAME}")
        print(f"Examples: {len(list(client.list_examples(dataset_id=dataset.id)))}")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Run create_dataset.py first!")
        return
    
    print("\n" + "-" * 60)
    print("Evaluators:")
    for evaluator in ALL_EVALUATORS:
        print(f"  - {evaluator.__name__}")
    
    print("\n" + "-" * 60)
    print("Running evaluation...")
    print("This will analyze EXISTING traces in LangSmith")
    print("(Make sure you've run your agent on these test cases!)")
    print("-" * 60 + "\n")
    
    # Run evaluation on existing runs
    # This analyzes traces that already exist in LangSmith
    try:
        results = evaluate(
            lambda x: None,  # Don't re-run, just evaluate existing traces
            data=dataset.name,
            evaluators=ALL_EVALUATORS,
            experiment_prefix=f"worms-eval-{datetime.now().strftime('%Y%m%d-%H%M')}",
            max_concurrency=1,
            client=client
        )
        
        print("\n" + "=" * 60)
        print("Evaluation Complete")
        print("=" * 60)
        print(f"Results: {results}")
        print("\nView detailed results in LangSmith UI:")
        print("https://smith.langchain.com/")
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you've run your agent on the test cases")
        print("2. Check that traces exist in LangSmith")
        print("3. Verify dataset exists: python create_dataset.py")


def evaluate_existing_traces():
    """
    Alternative: Evaluate your most recent traces
    """
    print("=" * 60)
    print("Evaluating Recent Traces")
    print("=" * 60)
    
    # Get recent runs from your project
    recent_runs = list(client.list_runs(
        project_name="worms-agent",
        limit=10,
        is_root=True  # Only top-level runs
    ))
    
    print(f"Found {len(recent_runs)} recent traces")
    
    if not recent_runs:
        print("No traces found! Run your agent first.")
        return
    
    print("\nEvaluating each trace...\n")
    
    # Evaluate each run
    for i, run in enumerate(recent_runs, 1):
        # Get metadata
        metadata = run.metadata if hasattr(run, 'metadata') and run.metadata else {}
        user_query = metadata.get('user_query', 'N/A')
        
        print(f"Trace {i}/{len(recent_runs)}:")
        print(f"  ID: {run.id}")
        print(f"  Query: {user_query}")
        
        # Run evaluators
        for evaluator in ALL_EVALUATORS:
            try:
                # Create mock example (no ground truth for existing traces)
                from langsmith.schemas import Example
                mock_example = Example(
                    id=run.id,
                    inputs=run.inputs,
                    outputs={}  # No ground truth
                )
                
                result = evaluator(run, mock_example)
                print(f"  {result['key']}: {result['score']:.2f} - {result['comment']}")
            except Exception as e:
                print(f"  {evaluator.__name__}: Error - {e}")
        
        print()
    
    print("=" * 60)
    print("Done! Check output above for scores")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    print("\nChoose evaluation mode:")
    print("1. Evaluate against test dataset (requires running agent on test cases)")
    print("2. Evaluate recent traces (quick analysis of what you have)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_evaluation()
    elif choice == "2":
        evaluate_existing_traces()
    else:
        print("Invalid choice!")