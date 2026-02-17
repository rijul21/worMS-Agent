"""
Validation Framework

Simple, focused validation with 5 validators across 3 buckets:
- SYSTEM: API Schema, Tool Execution  
- SEMANTIC: Query Alignment, Completeness
- DATA: Domain Constraints
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import os
import re

import langsmith.client
Client = langsmith.client.Client


# Base classes

class ValidationCategory(Enum):
    SYSTEM = "system"
    SEMANTIC = "semantic"
    DATA = "data"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValidationError:
    category: ValidationCategory
    error_type: str
    severity: Severity
    description: str
    query: str
    response: str
    evidence: Dict[str, Any]
    suggestion: str
    run_id: str
    timestamp: datetime


@dataclass
class ValidationResult:
    run_id: str
    timestamp: datetime
    query: str
    response: str
    system_errors: List[ValidationError]
    semantic_errors: List[ValidationError]
    data_errors: List[ValidationError]
    
    @property
    def total_errors(self) -> int:
        return len(self.system_errors) + len(self.semantic_errors) + len(self.data_errors)


# 1: SYSTEM VALIDATORS
class SystemValidator:
    """Validates infrastructure and tool execution"""
    
    def __init__(self, client: Client):
        self.client = client
    
    # API Validator
    def validate_api_schema(self, run_id: str, query: str, response: str, 
                           timestamp: datetime) -> List[ValidationError]:
        """
        Checks if WoRMS API responses have errors.
        Your tools return error strings like "Error retrieving...", "Species not found", etc.
        """
        errors = []
        
        try:
            # Get tool traces
            child_runs = list(self.client.list_runs(trace_id=run_id, limit=100))
            
            for run in child_runs:
                if run.run_type != "tool":
                    continue
                
                tool_name = run.name
                
                # Skip control tools
                if tool_name in ["finish", "abort"]:
                    continue
                
                # Check tool output for error strings
                if run.outputs and "output" in run.outputs:
                    output = str(run.outputs["output"])
                    
                    # Pattern 1: "Error retrieving X: ..."
                    if output.startswith("Error retrieving"):
                        errors.append(ValidationError(
                            category=ValidationCategory.SYSTEM,
                            error_type="api_error",
                            severity=Severity.HIGH,
                            description=f"Tool '{tool_name}' encountered API error",
                            query=query,
                            response=response,
                            evidence={
                                "tool": tool_name,
                                "error_message": output[:200]
                            },
                            suggestion="Check WoRMS API connectivity and response format",
                            run_id=run_id,
                            timestamp=timestamp
                        ))
                    
                    # Pattern 2: "Species 'X' not found in WoRMS database."
                    elif "not found in WoRMS database" in output:
                        errors.append(ValidationError(
                            category=ValidationCategory.SYSTEM,
                            error_type="species_not_found",
                            severity=Severity.HIGH,
                            description=f"Tool '{tool_name}' could not resolve species name",
                            query=query,
                            response=response,
                            evidence={
                                "tool": tool_name,
                                "message": output[:150]
                            },
                            suggestion="Verify species name spelling or try common name search",
                            run_id=run_id,
                            timestamp=timestamp
                        ))
                    
                    # Pattern 3: "No {data_type} found for {species}"
                    elif output.startswith("No ") and "found for" in output:
                        errors.append(ValidationError(
                            category=ValidationCategory.SYSTEM,
                            error_type="no_data_found",
                            severity=Severity.MEDIUM,
                            description=f"Tool '{tool_name}' found no data",
                            query=query,
                            response=response,
                            evidence={
                                "tool": tool_name,
                                "message": output[:150]
                            },
                            suggestion="Species may not have this type of data in WoRMS",
                            run_id=run_id,
                            timestamp=timestamp
                        ))
                    
                    # Pattern 4: "Error searching for common name: ..."
                    elif output.startswith("Error searching"):
                        errors.append(ValidationError(
                            category=ValidationCategory.SYSTEM,
                            error_type="search_error",
                            severity=Severity.HIGH,
                            description=f"Tool '{tool_name}' encountered search error",
                            query=query,
                            response=response,
                            evidence={
                                "tool": tool_name,
                                "error_message": output[:200]
                            },
                            suggestion="Check WoRMS API connectivity",
                            run_id=run_id,
                            timestamp=timestamp
                        ))
        
        except Exception:
            pass
        
        return errors
    
    #2: Tool Execution Validator
    def validate_tool_execution(self, run_id: str, query: str, response: str,
                                timestamp: datetime) -> List[ValidationError]:
        """
        Detects tool failures, empty responses, and execution problems.
    Tools have @cache_tool_result which always returns "" for cached calls.
        """
        errors = []
        
        try:
            child_runs = list(self.client.list_runs(trace_id=run_id, limit=100))
            
            tool_calls = {}  # Track tool call counts
            tool_outputs = {}  # Track outputs (to detect "No data found")
            
            for run in child_runs:
                if run.run_type != "tool":
                    continue
                
                tool_name = run.name
                
                # Skip control tools
                if tool_name in ["finish", "abort"]:
                    continue
                
                # Count tool calls
                tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
                
                # Store output
                if run.outputs and "output" in run.outputs:
                    output = str(run.outputs["output"])
                    tool_outputs[tool_name] = output
            
            # Check for repeated tool calls
            # Due to caching, if same tool called 2x with SAME args, 2nd returns ""
            # Only flag if called with DIFFERENT args (indicates agent confusion)
            for tool_name, count in tool_calls.items():
                if count > 2:  # More than 2 is suspicious even with retries
                    errors.append(ValidationError(
                        category=ValidationCategory.SYSTEM,
                        error_type="excessive_tool_calls",
                        severity=Severity.MEDIUM,
                        description=f"Tool '{tool_name}' called {count} times (agent may be confused)",
                        query=query,
                        response=response,
                        evidence={
                            "tool": tool_name,
                            "call_count": count
                        },
                        suggestion="Check if agent is stuck in a loop or retrying unnecessarily",
                        run_id=run_id,
                        timestamp=timestamp
                    ))
        
        except Exception:
            pass
        
        return errors

#2: SEMANTIC VALIDATORS

class SemanticValidator:
    """Validates meaning and relevance of responses"""
    
    #3: Query-Response Alignment
    def validate_query_alignment(self, query: str, response: str, run_id: str,
                                 timestamp: datetime) -> List[ValidationError]:
        """
        Checks if response actually addresses what was asked.
        Uses keyword matching (simple but effective).
        """
        errors = []
        
        query_lower = query.lower()
        response_lower = response.lower()
        
        # Define topic categories and their keywords
        topics = {
            "conservation": ["conservation", "iucn", "endangered", "threatened", "extinct", "cites", "status"],
            "distribution": ["distribution", "where", "location", "found", "habitat", "range", "live"],
            "taxonomy": ["family", "order", "class", "phylum", "taxonomy", "classification"],
            "size": ["size", "length", "weight", "big", "large", "small", "dimensions", "how tall"],
            "diet": ["diet", "eat", "food", "prey", "feeding", "hunt"],
            "behavior": ["behavior", "behaviour", "social", "migration", "breeding"]
        }
        
        # What is the user asking about?
        asked_topics = []
        for topic, keywords in topics.items():
            if any(kw in query_lower for kw in keywords):
                asked_topics.append(topic)
        
        # If we identified topics, check if response addresses them
        if asked_topics:
            has_relevant_content = False
            
            for topic in asked_topics:
                # Check if response mentions relevant keywords
                if any(kw in response_lower for kw in topics[topic]):
                    has_relevant_content = True
                    break
            
            if not has_relevant_content:
                # Additional check: Maybe response is too generic
                generic_phrases = [
                    "i don't have", "unable to", "cannot provide",
                    "not available", "no information"
                ]
                
                is_generic_failure = any(phrase in response_lower for phrase in generic_phrases)
                
                if not is_generic_failure:
                    errors.append(ValidationError(
                        category=ValidationCategory.SEMANTIC,
                        error_type="query_response_mismatch",
                        severity=Severity.HIGH,
                        description=f"User asked about {'/'.join(asked_topics)} but response doesn't address it",
                        query=query,
                        response=response,
                        evidence={
                            "asked_topics": asked_topics,
                            "response_length": len(response),
                            "response_preview": response[:150]
                        },
                        suggestion="Ensure agent uses correct tools for the query type",
                        run_id=run_id,
                        timestamp=timestamp
                    ))
        
        return errors
    
    #4: Completeness Validator
    def validate_completeness(self, query: str, response: str, run_id: str,
                             timestamp: datetime) -> List[ValidationError]:
        """
        Checks if response contains expected information types.
        Detects "I retrieved data" with no actual data shown.
        """
        errors = []
        
        response_lower = response.lower()
        
        # Check for retrieval claims
        retrieval_claims = [
            "retrieved", "found", "obtained", "collected",
            "here is", "here are", "the following", "i have gathered"
        ]
        
        claims_retrieval = any(claim in response_lower for claim in retrieval_claims)
        
        if claims_retrieval:
           
            
            # Is response too short?
            if len(response) < 80:
                errors.append(ValidationError(
                    category=ValidationCategory.SEMANTIC,
                    error_type="empty_retrieval_claim",
                    severity=Severity.HIGH,
                    description="Response claims to have retrieved data but is very short",
                    query=query,
                    response=response,
                    evidence={
                        "response_length": len(response),
                        "has_retrieval_claim": True
                    },
                    suggestion="Ensure finish() summary includes actual data points",
                    run_id=run_id,
                    timestamp=timestamp
                ))
            
            # Does it have concrete information?
            has_numbers = bool(re.search(r'\d+', response))
            has_specific_terms = bool(re.search(r'(meters?|kg|IUCN|Annex|Directive|endangered)', 
                                               response, re.IGNORECASE))
            
            if not has_numbers and not has_specific_terms and len(response) < 150:
                errors.append(ValidationError(
                    category=ValidationCategory.SEMANTIC,
                    error_type="lacks_specific_data",
                    severity=Severity.MEDIUM,
                    description="Response lacks concrete data (no numbers or specific terms)",
                    query=query,
                    response=response,
                    evidence={
                        "has_numbers": has_numbers,
                        "has_specific_terms": has_specific_terms,
                        "response_length": len(response)
                    },
                    suggestion="Include specific facts in finish() summary",
                    run_id=run_id,
                    timestamp=timestamp
                ))
        
        # Check if user asks specific question but gets vague answer
        specific_question_words = ["what is", "how many", "when", "where exactly", "which"]
        asks_specific_question = any(word in query.lower() for word in specific_question_words)
        
        if asks_specific_question:
            vague_phrases = ["varies", "depends", "can range", "typically", "generally", "may", "might"]
            vague_count = sum(1 for phrase in vague_phrases if phrase in response_lower)
            
            has_concrete_data = bool(re.search(r'\d+', response))
            
            if vague_count >= 2 and not has_concrete_data and len(response) < 200:
                errors.append(ValidationError(
                    category=ValidationCategory.SEMANTIC,
                    error_type="vague_response_to_specific_query",
                    severity=Severity.MEDIUM,
                    description=f"Specific question answered with {vague_count} vague phrases and no concrete data",
                    query=query,
                    response=response,
                    evidence={
                        "vague_phrase_count": vague_count,
                        "has_numbers": has_concrete_data
                    },
                    suggestion="Provide specific values when available in retrieved data",
                    run_id=run_id,
                    timestamp=timestamp
                ))
        
        return errors

# BUCKET 3: DATA VALIDATORS  

class DataValidator:
    """Validates WoRMS API data quality"""
    
    def __init__(self, client: Client):
        self.client = client
    
    # 5: Domain Constraint Validator
    def validate_domain_constraints(self, run_id: str, query: str, response: str,
                                   timestamp: datetime) -> List[ValidationError]:
        """
        Checks domain-specific rules for WoRMS data by examining artifact metadata.
        Your tools create artifacts with metadata containing aphia_id, count, species, etc.
        """
        errors = []
        
        try:
            # Get all child runs
            child_runs = list(self.client.list_runs(trace_id=run_id, limit=100))
            
            for run in child_runs:
                if run.run_type != "tool":
                    continue
                
                tool_name = run.name
                
                # Skip control tools
                if tool_name in ["finish", "abort"]:
                    continue
                
                # Check for invalid inputs before tools even run
                if run.inputs:
                    # Check for empty species names
                    if "species_name" in run.inputs:
                        species = run.inputs["species_name"]
                        if not species or (isinstance(species, str) and species.strip() == ""):
                            errors.append(ValidationError(
                                category=ValidationCategory.DATA,
                                error_type="empty_species_name",
                                severity=Severity.HIGH,
                                description=f"Tool '{tool_name}' called with empty species name",
                                query=query,
                                response=response,
                                evidence={
                                    "tool": tool_name,
                                    "species_name": species
                                },
                                suggestion="Validate input parameters before tool calls",
                                run_id=run_id,
                                timestamp=timestamp
                            ))
                    
                    # Check for suspicious category_id or attribute_id values
                    if "category_id" in run.inputs:
                        cat_id = run.inputs["category_id"]
                        if not isinstance(cat_id, int) or cat_id < 0:
                            errors.append(ValidationError(
                                category=ValidationCategory.DATA,
                                error_type="invalid_category_id",
                                severity=Severity.MEDIUM,
                                description=f"Tool '{tool_name}' called with invalid category_id: {cat_id}",
                                query=query,
                                response=response,
                                evidence={
                                    "tool": tool_name,
                                    "category_id": cat_id,
                                    "type": type(cat_id).__name__
                                },
                                suggestion="Validate category_id is non-negative integer",
                                run_id=run_id,
                                timestamp=timestamp
                            ))
                
                # Check outputs for data constraint violations
                if run.outputs and "output" in run.outputs:
                    output = str(run.outputs["output"])
                    
                    # If output contains error about invalid data, flag it
                    # (Though your tools don't currently validate this, good to catch if added later)
                    if "invalid" in output.lower() or "malformed" in output.lower():
                        errors.append(ValidationError(
                            category=ValidationCategory.DATA,
                            error_type="data_validation_error",
                            severity=Severity.MEDIUM,
                            description=f"Tool '{tool_name}' detected invalid data",
                            query=query,
                            response=response,
                            evidence={
                                "tool": tool_name,
                                "message": output[:200]
                            },
                            suggestion="Check WoRMS API data quality",
                            run_id=run_id,
                            timestamp=timestamp
                        ))
        
        except Exception:
            pass
        
        return errors

# MAIN

class ValidationFramework:
    """Orchestrates all validators"""
    
    def __init__(self):
        self.client = Client()
        self.system_validator = SystemValidator(self.client)
        self.semantic_validator = SemanticValidator()
        self.data_validator = DataValidator(self.client)
        self.results: List[ValidationResult] = []
    
    def validate_run(self, run_id: str, query: str, response: str, 
                    timestamp: datetime) -> ValidationResult:
        """Run all validators on a single agent execution"""
        
        # SYSTEM
        system_errors = []
        system_errors.extend(self.system_validator.validate_api_schema(
            run_id, query, response, timestamp
        ))
        system_errors.extend(self.system_validator.validate_tool_execution(
            run_id, query, response, timestamp
        ))
        
        #  SEMANTIC
        semantic_errors = []
        semantic_errors.extend(self.semantic_validator.validate_query_alignment(
            query, response, run_id, timestamp
        ))
        semantic_errors.extend(self.semantic_validator.validate_completeness(
            query, response, run_id, timestamp
        ))
        
        #DATA
        data_errors = []
        data_errors.extend(self.data_validator.validate_domain_constraints(
            run_id, query, response, timestamp
        ))
        
        return ValidationResult(
            run_id=run_id,
            timestamp=timestamp,
            query=query,
            response=response,
            system_errors=system_errors,
            semantic_errors=semantic_errors,
            data_errors=data_errors
        )
    
    def analyze_traces(self, hours: float = 72):
        """Analyze traces from LangSmith"""
        project_name = os.getenv("LANGCHAIN_PROJECT", "worms-agent")
        
        from datetime import timezone
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        print(f"\n{'='*60}")
        print(f"VALIDATION FRAMEWORK v0.2")
        print(f"{'='*60}")
        print(f"Project: {project_name}")
        print(f"Time range: Last {hours} hours")
        print(f"{'='*10}\n")
        
        try:
            all_runs = list(self.client.list_runs(
                project_name=project_name,
                start_time=start_time,
                end_time=end_time,
            ))
            
            agent_runs = [r for r in all_runs if r.name == "worms_agent_run"]
            
            print(f"Found {len(agent_runs)} agent conversations")
            print(f"Running 5 validators across 3 buckets...\n")
            
            for run in agent_runs:
                # Extract query
                query = run.inputs.get("request", "") if run.inputs else ""
                if not query:
                    print(f"  Skipping run {str(run.id)[:8]}: No query found")
                    continue
                
                # Extract response (simplified)
                response = self._extract_response(run.id)
                if not response:
                    print(f"  Skipping run {str(run.id)[:8]}: No response found (query: '{query[:50]}...')")
                    continue
                
                print(f"  ✓ Analyzing run {str(run.id)[:8]}: '{query[:50]}...'")
                
                # Run validation
                result = self.validate_run(str(run.id), query, response, run.start_time)
                self.results.append(result)
            
            print(f"Validation complete: {len(self.results)} conversations analyzed")
            print(f"{'='*10}\n")
            
        except Exception as e:
            print(f"Error during analysis: {e}\n")
    
    def _extract_response(self, run_id: str) -> Optional[str]:
        """Extract agent's final response"""
        try:
            child_runs = list(self.client.list_runs(trace_id=run_id, limit=50))
            
            # Debug: Print what we found
            finish_tools = [r for r in child_runs if r.name == "finish"]
            # print(f"    DEBUG: Found {len(finish_tools)} finish tools in {len(child_runs)} child runs")
            
            # Look for finish tool
            for child in child_runs:
                if child.name == "finish" and child.inputs:
                    summary = child.inputs.get("summary", "")
                    if summary:
                        # print(f"    DEBUG: Found summary: '{summary[:50]}...'")
                        return summary
            
            # print(f"    DEBUG: No finish tool with summary found")
            return None
        except Exception as e:
            # print(f"    DEBUG: Exception in _extract_response: {e}")
            return None
    
    def print_report(self):
        """Generate categorized report"""
        if not self.results:
            print("No validation results to report.\n")
            return
        
        print(f"\n{'='*60}")
        print("VALIDATION REPORT")
        print(f"{'='*10}\n")
        
        # Summary stats
        total_errors = sum(r.total_errors for r in self.results)
        conversations_with_errors = sum(1 for r in self.results if r.total_errors > 0)
        
        print(f"Conversations analyzed: {len(self.results)}")
        print(f"Conversations with errors: {conversations_with_errors}")
        print(f"Total errors found: {total_errors}\n")
        
        # Errors by category
        system_count = sum(len(r.system_errors) for r in self.results)
        semantic_count = sum(len(r.semantic_errors) for r in self.results)
        data_count = sum(len(r.data_errors) for r in self.results)
        
        print("Errors by Category:")
        print(f"  SYSTEM:   {system_count}")
        print(f"  SEMANTIC: {semantic_count}")
        print(f"  DATA:     {data_count}\n")
        
        # Errors by type
        error_types = {}
        for result in self.results:
            all_errors = result.system_errors + result.semantic_errors + result.data_errors
            for error in all_errors:
                error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        if error_types:
            print("Errors by Type:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
            print()
        
        # Detailed errors
        for result in self.results:
            if result.total_errors == 0:
                continue
            
            print(f"\n{'='*60}")
            print(f"Run ID: {result.run_id[:16]}...")
            print(f"Timestamp: {result.timestamp}")
            print(f"{'='*60}")
            
            print(f"\nQUERY: {result.query}")
            print(f"RESPONSE: {result.response[:200]}{'...' if len(result.response) > 200 else ''}\n")
            
            # Print errors by category
            if result.system_errors:
                print(f"SYSTEM ERRORS ({len(result.system_errors)}):")
                for err in result.system_errors:
                    print(f"  [{err.severity.value.upper()}] {err.error_type}")
                    print(f"    {err.description}")
                    print(f"    Suggestion: {err.suggestion}\n")
            
            if result.semantic_errors:
                print(f"SEMANTIC ERRORS ({len(result.semantic_errors)}):")
                for err in result.semantic_errors:
                    print(f"  [{err.severity.value.upper()}] {err.error_type}")
                    print(f"    {err.description}")
                    print(f"    Suggestion: {err.suggestion}\n")
            
            if result.data_errors:
                print(f"DATA ERRORS ({len(result.data_errors)}):")
                for err in result.data_errors:
                    print(f"  [{err.severity.value.upper()}] {err.error_type}")
                    print(f"    {err.description}")
                    print(f"    Suggestion: {err.suggestion}\n")
        
        print(f"\n{'='*60}")
        print("END OF REPORT")
        print(f"{'='*10}\n")

# CLI

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WoRMS Agent Validation Framework v0.2")
    parser.add_argument("--hours", type=float, default=72, 
                       help="Hours of history to analyze (default: 72)")
    parser.add_argument("--all", action="store_true", 
                       help="Analyze all available traces")
    
    args = parser.parse_args()
    
    hours = 8760 if args.all else args.hours
    
    framework = ValidationFramework()
    framework.analyze_traces(hours=hours)
    framework.print_report()