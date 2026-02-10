"""

Detects the following problems in agent responses:
- Claims data retrieval but shows nothing
- Answers unrelated to the question
- Vague responses to specific questions
- System failures (can't access tools)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import os
import re

import langsmith.client
Client = langsmith.client.Client


@dataclass
class ErrorInstance:
    """Represents a detected error with context"""
    error_type: str
    severity: str
    description: str
    query: str
    response: str
    evidence: str
    run_id: str
    timestamp: datetime


class ErrorDetector:
    """Main error detection system"""
    
    def __init__(self):
        self.client = Client()
        self.errors: List[ErrorInstance] = []
    
    def detect_empty_retrieval(self, query: str, response: str) -> Optional[Dict]:
        """
        Catches when agent says it got data but doesn't actually show it.
        Common issue: "Retrieved all habitats" with no list following.
        """
        retrieval_keywords = [
            "retrieved", "found", "obtained", "extracted",
            "here is", "here are", "the following"
        ]
        
        claims_retrieval = any(word in response.lower() for word in retrieval_keywords)
        
        if not claims_retrieval:
            return None
        
        # Short response with "retrieved" but no colon (no actual data list)
        if len(response) < 100:
            if "retrieved" in response.lower() and ":" not in response:
                return {
                    "type": "empty_retrieval",
                    "severity": "high",
                    "description": "Agent claims to have retrieved data but provides none",
                    "evidence": f"Response says 'retrieved' but is only {len(response)} chars with no actual data"
                }
        
        return None
    
    def detect_topic_mismatch(self, query: str, response: str) -> Optional[Dict]:
        """
        Checks if response actually addresses what was asked.
        Example: user asks conservation status, agent talks about taxonomy.
        """
        query_lower = query.lower()
        
        # Map question types to expected keywords
        topic_keywords = {
            "conservation": ["conservation", "iucn", "endangered", "threatened", "status", "extinct"],
            "distribution": ["distribution", "where", "location", "found", "habitat", "range"],
            "taxonomy": ["family", "order", "class", "phylum", "classification", "taxonomy"],
            "size": ["size", "length", "weight", "how big", "dimensions"],
            "diet": ["diet", "eat", "food", "prey", "feeding"]
        }
        
        # Figure out what the user is asking about
        asked_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                asked_topics.append(topic)
        
        if not asked_topics:
            return None
        
        # See if response mentions any relevant keywords
        response_lower = response.lower()
        has_relevant_content = False
        
        for topic in asked_topics:
            if any(kw in response_lower for kw in topic_keywords[topic]):
                has_relevant_content = True
                break
        
        if not has_relevant_content:
            return {
                "type": "off_topic",
                "severity": "high",
                "description": f"User asked about {', '.join(asked_topics)} but response doesn't address it",
                "evidence": f"Query keywords: {asked_topics}, not found in response"
            }
        
        return None
    
    def detect_vague_language(self, query: str, response: str) -> Optional[Dict]:
        """
        Flags vague responses when user asks something specific.
        "What is the size?" -> "It varies" is not helpful.
        """
        specific_question_words = ["what is", "how many", "when", "where exactly", "which"]
        
        is_specific_question = any(word in query.lower() for word in specific_question_words)
        
        if not is_specific_question:
            return None
        
        # Count non-committal phrases
        vague_phrases = [
            "varies", "it depends", "can range", "typically", "generally",
            "often", "sometimes", "may", "might", "could be"
        ]
        
        vague_count = sum(1 for phrase in vague_phrases if phrase in response.lower())
        
        # Check if response has concrete info
        has_numbers = bool(re.search(r'\d+', response))
        has_locations = any(loc in response.lower() for loc in 
                           ["atlantic", "pacific", "mediterranean", "arctic", "antarctic"])
        
        # Flag if lots of vague words, no concrete data, and short
        if vague_count >= 2 and not has_numbers and not has_locations and len(response) < 200:
            return {
                "type": "weasel_words",
                "severity": "medium",
                "description": f"Vague response with {vague_count} non-committal phrases to specific question",
                "evidence": f"User asked specific question but got vague answer"
            }
        
        return None
    
    def detect_system_failure(self, query: str, response: str) -> Optional[Dict]:
        """
        Agent shouldn't admit it can't use its own tools.
        If it says "I don't have access to WoRMS", something is broken.
        """
        failure_phrases = [
            "i don't have access", "i cannot access", "unable to retrieve",
            "i don't have the ability", "i cannot provide", "not available to me",
            "i don't have information", "i'm not able to"
        ]
        
        response_lower = response.lower()
        
        for phrase in failure_phrases:
            if phrase in response_lower:
                return {
                    "type": "agent_confusion",
                    "severity": "critical",
                    "description": f"Agent claims it cannot do what it's designed to do",
                    "evidence": f"Response contains: '{phrase}'"
                }
        
        return None
    
    def check_conversation(self, run_id: str, query: str, response: str, 
                          timestamp: datetime) -> List[ErrorInstance]:
        """Run all checks on a single conversation"""
        found_errors = []
        
        detectors = [
            self.detect_empty_retrieval,
            self.detect_topic_mismatch,
            self.detect_vague_language,
            self.detect_system_failure
        ]
        
        for detector in detectors:
            result = detector(query, response)
            if result:
                found_errors.append(ErrorInstance(
                    error_type=result["type"],
                    severity=result["severity"],
                    description=result["description"],
                    query=query,
                    response=response,
                    evidence=result["evidence"],
                    run_id=run_id,
                    timestamp=timestamp
                ))
        
        return found_errors
    
    def analyze_traces(self, hours: float = 72):
        """Pull traces from LangSmith and run analysis"""
        project_name = os.getenv("LANGCHAIN_PROJECT", "worms-agent")
        
        # LangSmith uses UTC
        from datetime import timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        print(f"Analyzing last {hours} hours...")
        print(f"Project: {project_name}\n")
        
        try:
            print(f"Fetching all runs from {start_time} to {end_time}...")
            
            all_runs = list(self.client.list_runs(
                project_name=project_name,
                start_time=start_time,
                end_time=end_time,
            ))
            
            print(f"Total runs fetched: {len(all_runs)}")
            
            # Only care about top-level agent runs
            agent_runs = [r for r in all_runs if r.name == "worms_agent_run"]
            
            print(f"Found {len(agent_runs)} worms_agent_run conversations")
            
            # Show what types of runs we got (helpful for debugging)
            run_types = {}
            for r in all_runs:
                run_types[r.name] = run_types.get(r.name, 0) + 1
            
            print(f"\nRun types found:")
            for name, count in sorted(run_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {name}: {count}")
            print()
            
            for run in agent_runs:
                # Extract query from the run
                query = ""
                if run.inputs and "request" in run.inputs:
                    query = run.inputs["request"]
                
                if not query:
                    continue
                
                # Get response from child traces
                try:
                    child_traces = list(self.client.list_runs(trace_id=run.id, limit=50))
                    
                    response = ""
                    for child in child_traces:
                        if child.inputs and "messages" in child.inputs:
                            messages = child.inputs["messages"]
                            if isinstance(messages, list):
                                # Look for the finish tool call that has the summary
                                for msg in reversed(messages):
                                    if isinstance(msg, dict) and msg.get("type") == "ai":
                                        tool_calls = msg.get("tool_calls", [])
                                        for call in tool_calls:
                                            if call.get("name") == "finish":
                                                args = call.get("args", {})
                                                if "summary" in args:
                                                    response = args["summary"]
                                                    break
                                    if response:
                                        break
                            if response:
                                break
                    
                    if not response:
                        continue
                    
                    # Run checks on this conversation
                    conversation_errors = self.check_conversation(
                        str(run.id),
                        query,
                        response,
                        run.start_time
                    )
                    
                    self.errors.extend(conversation_errors)
                    
                except Exception as e:
                    continue
            
            print(f"Analysis complete")
            print(f"Found {len(self.errors)} real errors\n")
            
        except Exception as e:
            print(f"Error: {e}")
    
    def print_report(self):
        """Output the results"""
        print("=" * 70)
        print("ERROR DETECTION REPORT")
        print("=" * 70)
        
        if not self.errors:
            print("\nNo errors detected.")
            print("Agent appears to be working correctly.\n")
            return
        
        # Group errors by type
        errors_by_type = {}
        for error in self.errors:
            if error.error_type not in errors_by_type:
                errors_by_type[error.error_type] = []
            errors_by_type[error.error_type].append(error)
        
        print(f"\nTotal Errors: {len(self.errors)}")
        for error_type, errors in errors_by_type.items():
            print(f"  {error_type}: {len(errors)}")
        
        # Print details for each error type
        for error_type, errors in errors_by_type.items():
            print(f"\n{'='*70}")
            print(f"{error_type.upper().replace('_', ' ')} ({len(errors)} instances)")
            print(f"{'='*70}")
            
            for i, error in enumerate(errors, 1):
                print(f"\n--- Instance {i} ---")
                print(f"Severity: {error.severity}")
                print(f"Timestamp: {error.timestamp}")
                print(f"\nQUERY:")
                print(f"  {error.query}")
                print(f"\nRESPONSE:")
                print(f"  {error.response[:300]}...")
                print(f"\nPROBLEM:")
                print(f"  {error.description}")
                print(f"\nEVIDENCE:")
                print(f"  {error.evidence}")
                print(f"\n{'-'*70}")
        
        print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect errors in agent responses")
    parser.add_argument("--hours", type=float, default=72, help="Hours of history to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all available traces")
    
    args = parser.parse_args()
    
    # --all means go back one year
    hours = 8760 if args.all else args.hours
    
    detector = ErrorDetector()
    detector.analyze_traces(hours=hours)
    detector.print_report()