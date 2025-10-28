"""
Simple structured logging for WoRMS Agent
Keeps cache logs separate from tool logs
"""

from enum import Enum
from typing import Optional, Any


class LogCategory(Enum):
    CACHE = "CACHE"
    TOOL = "TOOL"
    AGENT = "AGENT"
    PLANNING = "PLANNING"  # NEW


async def log(
    process,
    message: str,
    category: LogCategory = LogCategory.TOOL,
    data: Optional[dict[str, Any]] = None,
):
    """
    Structured logging wrapper for process.log()
    
    Usage:
        await log(process, "Fetching data from API", LogCategory.TOOL)
        await log(process, "Cache hit", LogCategory.CACHE, data={"species": "Orcinus orca"})
    """
    formatted_message = f"[{category.value}] {message}"
    
    if data:
        await process.log(formatted_message, data=data)
    else:
        await process.log(formatted_message)


# Convenience functions for common logging patterns


async def log_api_call(process, tool_name: str, species_name: str, aphia_id: int, url: str):
    """Log API call"""
    await log(
        process,
        f"Calling WoRMS API for {tool_name}",
        LogCategory.TOOL,
        data={
            "species": species_name,
            "aphia_id": aphia_id,
            "url": url
        }
    )


async def log_data_fetched(process, tool_name: str, species_name: str, count: int):
    """Log data fetched"""
    await log(
        process,
        f"Found {count} records for {species_name}",
        LogCategory.TOOL,
        data={
            "tool": tool_name,
            "species": species_name,
            "record_count": count
        }
    )


async def log_no_data(process, tool_name: str, species_name: str, aphia_id: int):
    """Log no data found"""
    await log(
        process,
        f"No data found for {species_name}",
        LogCategory.TOOL,
        data={
            "tool": tool_name,
            "species": species_name,
            "aphia_id": aphia_id
        }
    )


async def log_species_not_found(process, species_name: str):
    """Log species not found"""
    await log(
        process,
        f"Species '{species_name}' not found in WoRMS database",
        LogCategory.TOOL
    )


async def log_tool_error(process, tool_name: str, species_name: str, error: Exception):
    """Log tool error"""
    await log(
        process,
        f"Error in {tool_name}",
        LogCategory.TOOL,
        data={
            "species": species_name,
            "error_type": type(error).__name__,
            "error": str(error)
        }
    )


async def log_artifact_created(process, tool_name: str, species_name: str):
    """Log artifact creation"""
    await log(
        process,
        f"Artifact created for {species_name}",
        LogCategory.TOOL,
        data={
            "tool": tool_name,
            "species": species_name
        }
    )


async def log_agent_init(process, request: str, tool_count: int):
    """Log agent initialization"""
    await log(
        process,
        f"Initializing agent with {tool_count} tools",
        LogCategory.AGENT,
        data={
            "request": request,
            "tool_count": tool_count
        }
    )


async def log_agent_error(process, error: Exception):
    """Log agent execution error"""
    await log(
        process,
        f"Agent execution failed",
        LogCategory.AGENT,
        data={
            "error_type": type(error).__name__,
            "error": str(error)
        }
    )


# NEW: Planning-specific logging functions


async def log_plan_created(process, query_type: str, species_count: int, tools_count: int):
    """Log plan creation"""
    await log(
        process,
        f"Research plan created",
        LogCategory.PLANNING,
        data={
            "query_type": query_type,
            "species_count": species_count,
            "tools_planned": tools_count
        }
    )


async def log_species_identified(process, species_name: str, is_common_name: bool):
    """Log species identification"""
    name_type = "common name" if is_common_name else "scientific name"
    await log(
        process,
        f"Identified species: {species_name} ({name_type})",
        LogCategory.PLANNING,
        data={
            "species": species_name,
            "name_type": name_type
        }
    )


async def log_tool_planned(process, tool_name: str, priority: str, reason: str):
    """Log planned tool"""
    await log(
        process,
        f"Tool planned: {tool_name} [{priority}]",
        LogCategory.PLANNING,
        data={
            "tool": tool_name,
            "priority": priority,
            "reason": reason
        }
    )


async def log_name_resolution_start(process, common_names: list[str]):
    """Log start of parallel name resolution"""
    await log(
        process,
        f"Resolving {len(common_names)} common name(s) in parallel",
        LogCategory.PLANNING,
        data={
            "common_names": common_names,
            "count": len(common_names)
        }
    )


async def log_name_resolved(process, common_name: str, scientific_name: str):
    """Log successful name resolution"""
    await log(
        process,
        f"Resolved: {common_name} -> {scientific_name}",
        LogCategory.PLANNING,
        data={
            "common_name": common_name,
            "scientific_name": scientific_name
        }
    )


async def log_name_resolution_failed(process, common_name: str):
    """Log failed name resolution"""
    await log(
        process,
        f"Failed to resolve: {common_name}",
        LogCategory.PLANNING,
        data={
            "common_name": common_name
        }
    )