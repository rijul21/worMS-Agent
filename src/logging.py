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
    PLANNING = "PLANNING"


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
    """Log API call - KEEP data (useful for debugging)"""
    await log(
        process,
        f"{tool_name}: {species_name}",
        LogCategory.TOOL,
        data={
            "species": species_name,
            "aphia_id": aphia_id,
            "url": url
        }
    )


async def log_data_fetched(process, tool_name: str, species_name: str, count: int):
    """Log data fetched - SIMPLIFIED (count visible in message)"""
    await log(
        process,
        f"Found {count} record(s) for {species_name}",
        LogCategory.TOOL
    )


async def log_no_data(process, tool_name: str, species_name: str, aphia_id: int):
    """Log no data found - KEEP data (useful for debugging)"""
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
    """Log species not found - SIMPLIFIED"""
    await log(
        process,
        f"Species '{species_name}' not found in WoRMS",
        LogCategory.TOOL
    )


async def log_tool_error(process, tool_name: str, species_name: str, error: Exception):
    """Log tool error - KEEP data (important for debugging)"""
    await log(
        process,
        f"Error in {tool_name} for {species_name}",
        LogCategory.TOOL,
        data={
            "species": species_name,
            "error_type": type(error).__name__,
            "error": str(error)
        }
    )


async def log_artifact_created(process, tool_name: str, species_name: str):
    """Log artifact creation - DISABLED (visible in UI)"""
    # Artifacts are visible in UI, no need to log
    pass


async def log_agent_init(process, request: str, tool_count: int):
    """Log agent initialization - SIMPLIFIED"""
    await log(
        process,
        f"Initializing agent with {tool_count} tools",
        LogCategory.AGENT
    )


async def log_agent_error(process, error: Exception):
    """Log agent execution error - KEEP data (important for debugging)"""
    await log(
        process,
        f"Agent execution failed",
        LogCategory.AGENT,
        data={
            "error_type": type(error).__name__,
            "error": str(error)
        }
    )