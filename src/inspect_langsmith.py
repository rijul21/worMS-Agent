"""
LangSmith Data Inspector
========================
Run this FIRST to see what data you actually have.
This will show us the structure so we can adapt the error detector.
"""

from langsmith import Client
from datetime import datetime, timedelta
import os
import json


def inspect_langsmith_data(hours: int = 24):
    """
    Inspect your LangSmith data to see what's actually there.
    This helps us understand if the error detector will work.
    """
    
    client = Client()
    
    # Get project name
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    print("=" * 70)
    print("LANGSMITH DATA INSPECTOR")
    print("=" * 70)
    print(f"\nProject: {project_name}")
    print(f"Looking back: {hours} hours")
    print()
    
    # Get time window
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours)
    
    try:
        # Fetch recent runs
        print("Fetching runs...")
        runs = list(client.list_runs(
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            limit=5  # Just get a few to inspect
        ))
        
        print(f"Found {len(runs)} runs\n")
        
        if not runs:
            print("❌ NO RUNS FOUND!")
            print("\nPossible reasons:")
            print("1. Wrong project name - check your .env file")
            print("2. No agent runs in the last 24 hours")
            print("3. LANGCHAIN_API_KEY not set correctly")
            print("\nCurrent settings:")
            print(f"  LANGCHAIN_PROJECT = {project_name}")
            print(f"  LANGCHAIN_API_KEY = {os.getenv('LANGCHAIN_API_KEY', 'NOT SET')[:20]}...")
            return
        
        print("✅ RUNS FOUND! Let's inspect them...\n")
        
        # Inspect first run in detail
        for i, run in enumerate(runs[:3], 1):
            print(f"\n{'='*70}")
            print(f"RUN {i}")
            print(f"{'='*70}")
            print(f"Run ID: {run.id}")
            print(f"Name: {run.name}")
            print(f"Run Type: {run.run_type}")
            print(f"Start Time: {run.start_time}")
            print(f"Is Root: {run.is_root if hasattr(run, 'is_root') else 'Unknown'}")
            
            # Inspect inputs
            print(f"\n--- INPUTS ---")
            if run.inputs:
                print(f"Keys: {list(run.inputs.keys())}")
                
                # Try to extract query
                if "messages" in run.inputs:
                    print(f"\nMessages structure:")
                    messages = run.inputs["messages"]
                    if isinstance(messages, list):
                        print(f"  Type: list with {len(messages)} messages")
                        if messages:
                            print(f"  First message type: {type(messages[0])}")
                            if isinstance(messages[0], dict):
                                print(f"  First message keys: {list(messages[0].keys())}")
                                print(f"  First message content preview: {str(messages[0].get('content', ''))[:100]}")
                            else:
                                print(f"  First message: {messages[0]}")
                else:
                    print(f"Full inputs: {run.inputs}")
            else:
                print("No inputs found")
            
            # Inspect outputs
            print(f"\n--- OUTPUTS ---")
            if run.outputs:
                print(f"Keys: {list(run.outputs.keys())}")
                
                if "messages" in run.outputs:
                    messages = run.outputs["messages"]
                    if isinstance(messages, list):
                        print(f"\nMessages: list with {len(messages)} messages")
                        if messages:
                            last_msg = messages[-1]
                            print(f"  Last message type: {type(last_msg)}")
                            if isinstance(last_msg, dict):
                                print(f"  Last message keys: {list(last_msg.keys())}")
                                content = last_msg.get('content', '')
                                print(f"  Last message content preview: {str(content)[:200]}")
                else:
                    print(f"Outputs: {run.outputs}")
            else:
                print("No outputs found")
            
            # Check for child runs
            print(f"\n--- CHILD RUNS ---")
            if hasattr(run, 'child_runs') and run.child_runs:
                print(f"Found {len(run.child_runs)} child runs")
                for j, child in enumerate(run.child_runs[:3], 1):
                    print(f"\n  Child {j}:")
                    print(f"    Name: {child.name}")
                    print(f"    Type: {child.run_type}")
                    if child.outputs:
                        print(f"    Output keys: {list(child.outputs.keys())}")
                        output_preview = str(child.outputs)[:100]
                        print(f"    Output preview: {output_preview}")
            else:
                print("No child runs found")
                print("(This might mean tool calls aren't being logged as child runs)")
            
            print()
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"✅ Can access LangSmith: YES")
        print(f"✅ Found runs: {len(runs)}")
        print(f"✅ Runs have inputs: {'YES' if runs[0].inputs else 'NO'}")
        print(f"✅ Runs have outputs: {'YES' if runs[0].outputs else 'NO'}")
        print(f"✅ Runs have child runs: {'YES' if hasattr(runs[0], 'child_runs') and runs[0].child_runs else 'NO'}")
        
        print("\n📊 NEXT STEPS:")
        if hasattr(runs[0], 'child_runs') and runs[0].child_runs:
            print("✅ Your data looks good! The error detector should work.")
            print("   Run: python agent_error_detector.py")
        else:
            print("⚠️  No child runs found - we need to modify the error detector")
            print("   The tool calls might be stored differently")
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"\nFull error: {type(e).__name__}: {str(e)}")
        print("\nCheck your LangSmith configuration:")
        print("1. LANGCHAIN_API_KEY is set correctly")
        print("2. LANGCHAIN_PROJECT matches your actual project")
        print("3. You have runs in the last 24 hours")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect LangSmith data structure")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back")
    
    args = parser.parse_args()
    
    inspect_langsmith_data(hours=args.hours)