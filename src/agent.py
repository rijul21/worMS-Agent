from typing import override, Optional  
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import dotenv
import asyncio
from functools import lru_cache  
import json
from typing import Literal

from worms_api import WoRMS
from tools import create_worms_tools
from src.logging import log_species_not_found

dotenv.load_dotenv()

class ToolPlan(BaseModel):
    tool_name: str
    priority: Literal["must_call", "should_call", "optional"]
    reason: str

class ResearchPlan(BaseModel):
    query_type: Literal["single_species", "comparison", "conservation", "distribution", "taxonomy"]
    species_mentioned: list[str]
    are_common_names: list[bool]
    tools_planned: list[ToolPlan]
    reasoning: str

class MarineResearchParams(BaseModel):
    """Parameters for marine species research requests"""
    species_names: list[str] = Field(
        default=[],
        description="Scientific names of marine species to research",
        examples=[["Orcinus orca"], ["Orcinus orca", "Delphinus delphis"]]
    )


AGENT_DESCRIPTION = "Marine species research assistant using WoRMS database"


class WoRMSReActAgent(IChatBioAgent):
    def __init__(self):
        self.worms_logic = WoRMS()
        # Stores up to 256 species
        self._cached_lookup = lru_cache(maxsize=256)(
            self.worms_logic.get_species_aphia_id
        )
        
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="WoRMS Agent",
            description=AGENT_DESCRIPTION,
            icon="https://www.marinespecies.org/images/WoRMS_logo.png",
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="research_marine_species",
                    description=AGENT_DESCRIPTION,
                    parameters=MarineResearchParams
                )
            ]
        )
    
    async def _create_plan(self, request: str, species_names: list[str]) -> ResearchPlan:
        """Create execution plan using LLM"""
        
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        
        # Use structured output parser
        parser = JsonOutputParser(pydantic_object=ResearchPlan)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a marine biology research planning expert.
    Analyze queries and create structured execution plans.

    Available tools:
    - search_by_common_name: Convert common names to scientific (USE FIRST if common name)
    - get_species_attributes: Conservation status, body size, IUCN, CITES
    - get_taxonomic_record: Basic taxonomy (family, order, class)
    - get_species_distribution: Geographic range
    - get_vernacular_names: Common names in languages
    - get_taxonomic_classification: Full taxonomy tree

    Query types:
    - "single_species": Info about one species
    - "comparison": Compare multiple species
    - "conservation": Specifically about conservation/IUCN status
    - "distribution": Specifically about where species lives
    - "taxonomy": About classification

    Tool priorities:
    - "must_call": Required to answer the query
    - "should_call": Recommended for complete answer
    - "optional": Only if user specifically asks

    {format_instructions}
    """),
            ("human", """Query: "{request}"
    Species mentioned: {species}

    Create the execution plan.""")
        ])
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        chain = prompt | llm | parser
        
        try:
            plan = await chain.ainvoke({
                "format_instructions": parser.get_format_instructions(),
                "request": request,
                "species": species_names if species_names else "unknown"
            })
            
            return ResearchPlan(**plan)
        
        except Exception as e:
            # Fallback plan if LLM fails
            print(f"Warning: Plan creation failed ({e}), using fallback plan")
            
            # Determine if names are likely common or scientific
            are_common = []
            for name in species_names:
                # Simple heuristic: scientific names usually have 2 words and first letter capitalized
                words = name.split()
                is_scientific = len(words) == 2 and words[0][0].isupper() and words[1][0].islower()
                are_common.append(not is_scientific)
            
            # Build fallback plan
            tools_planned = []
            
            # If any common names, need to resolve them
            if any(are_common):
                tools_planned.append(ToolPlan(
                    tool_name="search_by_common_name",
                    priority="must_call",
                    reason="Need to resolve common names to scientific names"
                ))
            
            # Always get basic attributes
            tools_planned.append(ToolPlan(
                tool_name="get_species_attributes",
                priority="must_call",
                reason="Get ecological traits and conservation status"
            ))
            
            tools_planned.append(ToolPlan(
                tool_name="get_taxonomic_record",
                priority="should_call",
                reason="Get basic taxonomy information"
            ))
            
            return ResearchPlan(
                query_type="single_species" if len(species_names) <= 1 else "comparison",
                species_mentioned=species_names,
                are_common_names=are_common,
                tools_planned=tools_planned,
                reasoning="Fallback plan: get core species information"
            )

    async def _resolve_common_names_parallel(
        self, 
        names: list[str], 
        context: ResponseContext
    ) -> dict[str, str]:
        """Resolve multiple common names to scientific names in parallel"""
        
        from tools import search_by_common_name
        
        async with context.begin_process(f"Resolving {len(names)} species names in parallel") as process:
            
            async def resolve_one(common_name: str) -> tuple[str, Optional[str]]:
                """Resolve single common name"""
                try:
                    # Call search tool
                    result = await search_by_common_name(common_name)
                    
                    # Parse result to extract scientific name
                    if "refers to" in result:
                        scientific_name = result.split("refers to ")[1].split(" ")[0:2]
                        scientific_name = " ".join(scientific_name).strip("()")
                        return (common_name, scientific_name)
                    return (common_name, None)
                except Exception as e:
                    await process.log(f"Error resolving {common_name}: {e}")
                    return (common_name, None)
            
            # Execute all resolutions in parallel
            tasks = [resolve_one(name) for name in names]
            results = await asyncio.gather(*tasks)
            
            # Build mapping
            resolved = {}
            for common_name, scientific_name in results:
                if scientific_name:
                    resolved[common_name] = scientific_name
                    await process.log(f"Resolved {common_name} -> {scientific_name}")
                else:
                    await process.log(f"Failed to resolve {common_name}")
            
            return resolved

    async def _get_cached_aphia_id(self, species_name: str, process) -> Optional[int]:
        """Get AphiaID with automatic caching"""
        loop = asyncio.get_event_loop()
        aphia_id = await loop.run_in_executor(
            None,
            self._cached_lookup,
            species_name
        )
        
        if aphia_id:
            await process.log(f"Resolved {species_name} -> AphiaID {aphia_id}")
        else:
            await log_species_not_found(process, species_name)
        
        return aphia_id
    
    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineResearchParams,
    ):
        """Main entry point with planning and parallel resolution"""

        # ============================================================
        # PHASE 1: PLANNING
        # ============================================================
        
        async with context.begin_process("Fetching data from WoRMS and responding with a plan") as process:
            
            plan = await self._create_plan(request, params.species_names)
            
            # Log 1: Query analysis and species identification
            species_str = ", ".join(plan.species_mentioned)
            await process.log(f"{plan.query_type.replace('_', ' ').title()} query: {species_str}")
            
            # Log 2: Execution plan with tool priorities
            must_call_tools = [t.tool_name for t in plan.tools_planned if t.priority == "must_call"]
            should_call_tools = [t.tool_name for t in plan.tools_planned if t.priority == "should_call"]
            
            plan_details = f"📋 Execution Plan: {len(must_call_tools)} required tools"
            if should_call_tools:
                plan_details += f", {len(should_call_tools)} recommended tools"
            
            await process.log(plan_details, data={
                "query_type": plan.query_type,
                "species_count": len(plan.species_mentioned),
                "must_call": [t.tool_name for t in plan.tools_planned if t.priority == "must_call"],
                "should_call": [t.tool_name for t in plan.tools_planned if t.priority == "should_call"],
                "reasoning": plan.reasoning
            })
            
            # User message
            await context.reply(f"Researching {len(plan.species_mentioned)} species using {len(must_call_tools)} tools...")


        # PHASE 2: PARALLEL NAME RESOLUTION
        
        resolved_names = {}
        
        # Get common names that need resolution
        common_names = [
            name for name, is_common in zip(plan.species_mentioned, plan.are_common_names)
            if is_common
        ]
        
        if common_names:
            async with context.begin_process("Resolving species names") as process:
                
                await process.log(f"Resolving {len(common_names)} common name(s) in parallel")
                
                # Resolve in parallel using the method
                resolved = await self._resolve_common_names_parallel(common_names, context)
                
                # Single summary log
                await process.log(f"Resolved {len(resolved)}/{len(common_names)} species")
                
                resolved_names = resolved
        
        # Pre-resolve scientific names (for caching)
        scientific_names = [
            name for name, is_common in zip(plan.species_mentioned, plan.are_common_names)
            if not is_common
        ]
        
        if scientific_names:
            async with context.begin_process("Validating scientific names") as process:
                for species_name in scientific_names:
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Warning: Could not validate {species_name}")

  
        # PHASE 3: GUIDED EXECUTION
        
        # Create tools
        tools = create_worms_tools(
            worms_logic=self.worms_logic,
            context=context,
            get_cached_aphia_id_func=self._get_cached_aphia_id
        )
        
        # Create agent with plan-enhanced prompt
        llm = ChatOpenAI(model="gpt-4o-mini")
        system_prompt = self._make_system_prompt_with_plan(request, plan, resolved_names)
        agent = create_react_agent(llm, tools)
        
        try:
            await agent.ainvoke({
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=request)
                ]
            })
        except Exception as e:
            await context.reply(f"An error occurred: {str(e)}")
    
    def _make_system_prompt_with_plan(
        self, 
        request: str, 
        plan: ResearchPlan,
        resolved_names: dict[str, str]
    ) -> str:
        """Generate system prompt that includes the plan"""
        
        # Build species context
        species_context = "\n\nSPECIES INFORMATION:\n"
        for species, is_common in zip(plan.species_mentioned, plan.are_common_names):
            if is_common and species in resolved_names:
                species_context += f"  • {species} (common name) -> {resolved_names[species]} (scientific name)\n"
            else:
                species_context += f"  • {species} (scientific name)\n"
        
        # Build tool plan
        must_call = [t for t in plan.tools_planned if t.priority == "must_call"]
        should_call = [t for t in plan.tools_planned if t.priority == "should_call"]
        
        tool_context = "\n\nEXECUTION PLAN:\n"
        tool_context += "MUST CALL (required to answer query):\n"
        for tool in must_call:
            tool_context += f"  • {tool.tool_name} - {tool.reason}\n"
        
        if should_call:
            tool_context += "\nSHOULD CALL (for complete answer):\n"
            for tool in should_call:
                tool_context += f"  • {tool.tool_name} - {tool.reason}\n"
        
        return f"""You are a marine biology research assistant with access to the WoRMS database.

USER REQUEST: "{request}"

QUERY TYPE: {plan.query_type}
STRATEGY: {plan.reasoning}
{species_context}{tool_context}

CRITICAL INSTRUCTIONS:

1. FOLLOW THE EXECUTION PLAN ABOVE:
   - Call all "MUST CALL" tools (required for this query)
   - Call "SHOULD CALL" tools if they help provide a complete answer
   - Skip tools not listed in the plan
   - Call each tool AT MOST ONCE per species

2. USE RESOLVED NAMES:
   - Common names have already been resolved to scientific names (see above)
   - Always use the scientific names shown above for tool calls
   - Do NOT call search_by_common_name again - names are already resolved

3. FOR COMPARISON QUERIES:
   - Collect the SAME data points for all species
   - After collecting, provide comparative analysis with specific facts

4. EFFICIENCY:
   - If you have enough data to answer the query, call finish() immediately
   - Don't call unnecessary tools
   - Don't retry failed calls

5. RESPONSE QUALITY:
   - Lead with direct answer to user's question
   - Include specific facts (IUCN status, sizes, locations)
   - Mention that full data is in artifacts

Always create artifacts when retrieving data from WoRMS.
"""


if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS Agent Server")
    print("=" * 60)
    print(f"URL: http://localhost:9999")
    print(f"Status: Ready with planning capabilities")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)