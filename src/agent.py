from typing import override, Optional  
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
import dotenv
import asyncio
from functools import lru_cache  

from worms_api import WoRMS
from tools import create_worms_tools
from src.logging import log_species_not_found

dotenv.load_dotenv()


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
        # Automatic caching with LRU cache (stores up to 256 species)
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

    async def _get_cached_aphia_id(self, species_name: str, process) -> Optional[int]:
        """Get AphiaID with automatic caching"""
        loop = asyncio.get_event_loop()
        aphia_id = await loop.run_in_executor(
            None,
            self._cached_lookup,
            species_name
        )
        
        if aphia_id:
            await process.log(f"Resolved {species_name} → AphiaID {aphia_id}")
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
        """Main entry point - builds and executes the ReAct agent loop"""

        # Pre-resolve species names if provided
        if params.species_names:
            async with context.begin_process("Resolving species identifiers from WoRMS") as process:
                for species_name in params.species_names:
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Warning: Could not resolve {species_name}")

        # Create all tools using factory function
        tools = create_worms_tools(
            worms_logic=self.worms_logic,
            context=context,
            get_cached_aphia_id_func=self._get_cached_aphia_id
        )
        
        # Execute agent
        llm = ChatOpenAI(model="gpt-4o-mini")
        system_prompt = self._make_system_prompt(params.species_names, request)
        agent = create_react_agent(llm, tools)
        
        try:
            await agent.ainvoke({
                "messages": [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=request)
                ]
            })
        except Exception as e:
            await context.reply(f"An error occurred while processing your request: {str(e)}")
    
    def _make_system_prompt(self, species_names: list[str], user_request: str) -> str:
        """Generate system prompt for the WoRMS agent"""
        species_context = f"\n\nSpecies to research: {', '.join(species_names)}" if species_names else ""

        return f"""\
You are a marine biology research assistant with access to the WoRMS (World Register of Marine Species) database.

Request: "{user_request}"{species_context}

CRITICAL INSTRUCTIONS:

1. HANDLING COMMON NAMES:
   - If the user provides a COMMON NAME (e.g., "killer whale", "great white shark"), ALWAYS call search_by_common_name FIRST
   - Once you get the scientific name, use it for all subsequent tool calls
   - Examples of common names: killer whale, great white, bottlenose dolphin, tiger shark, hammerhead
   - Examples of scientific names: Orcinus orca, Carcharodon carcharias, Tursiops truncatus

2. PLANNING YOUR APPROACH:
   - For simple queries (e.g., "What's the conservation status?"), call only relevant tools
   - For comprehensive queries (e.g., "Tell me everything about X"), call multiple tools systematically
   - For comparison queries, gather the same data points for each species
   - Typical order: search (if common name) → taxonomy → attributes → distribution → names → sources

3. TOOL USAGE GUIDELINES:
   
   SEARCH & IDENTIFICATION:
   - search_by_common_name: Convert common names to scientific names (e.g., "killer whale" → "Orcinus orca")
     * USE THIS FIRST if user provides common/vernacular names
     * Returns scientific name and AphiaID
   
   TAXONOMY:
   - get_taxonomic_record: Basic taxonomy (rank, status, kingdom, phylum, class, order, family)
   - get_taxonomic_classification: Full taxonomic hierarchy (use for detailed taxonomy)
   
   ECOLOGY & CONSERVATION:
   - get_species_attributes: Ecological traits, conservation status, body size, IUCN Red List status, CITES
     * Use for: conservation status, body size, ecological traits, habitat preferences
     * Returns nested data including IUCN status, CITES Annex, size measurements
   
   DISTRIBUTION & NAMES:
   - get_species_distribution: Geographic distribution data (where the species lives)
   - get_vernacular_names: Common names in different languages for a known species
   - get_species_synonyms: Alternative scientific names (historical names, misspellings)
   
   REFERENCES & DATABASES:
   - get_literature_sources: Scientific references and publications (only if explicitly needed)
   - get_external_ids: External database identifiers (FishBase, NCBI, ITIS, BOLD)
   
   OTHER:
   - get_child_taxa: Subspecies/varieties (may return empty for terminal species - this is NORMAL)

4. ERROR HANDLING:
   - If search_by_common_name returns no results, ask user for clarification or try scientific name
   - If get_child_taxa returns empty or error, this is NORMAL for terminal species - don't retry
   - Don't repeatedly call the same tool if it returns empty results
   - If a species is not found, clearly inform the user and suggest alternatives

5. EFFICIENCY & AVOIDING LOOPS:
   - Call each tool AT MOST ONCE per species (except search_by_common_name if needed)
   - Don't retry failed calls - move on to other tools
   - If you get a complete answer, call finish() immediately
   - Only call tools that are relevant to the user's specific question
   - If user asks for "everything", call all relevant tools systematically

6. COMPARISON REQUIREMENTS:
   When comparing multiple species, provide comparative analysis:
   - Which has wider distribution?
   - Which has larger body size?
   - Conservation status differences (IUCN Red List categories)
   - Taxonomic relationships (same family/order?)
   - Which is more studied (literature count)?
   
   Don't just list facts - provide meaningful comparisons and insights.

7. RESPONSE QUALITY:
   - Lead with KEY INFORMATION that directly answers the user's question
   - DON'T ask "Would you like me to...?" - just provide the answer
   - For conservation queries, ALWAYS extract and state IUCN status if available
   - For size queries, mention both male and female sizes if available
   - Always mention that full data is available in artifacts
   - Be concise but comprehensive

8. FINISHING:
   - Call finish() with a summary that DIRECTLY ANSWERS the user's question
   - Include specific facts: conservation status, sizes, locations, etc.
   - For comparisons, include comparative insights, not just individual descriptions
   - Highlight key differences and similarities
   - Keep it concise but informative

EXAMPLES:

Example 1 - Common name query:
User: "What's the conservation status of killer whales?"
Process: search_by_common_name("killer whale") → "Orcinus orca" → get_species_attributes("Orcinus orca") → extract IUCN status → finish()

Example 2 - Scientific name query:
User: "Tell me about Carcharodon carcharias"
Process: get_taxonomic_record() → get_species_attributes() → get_species_distribution() → finish()

Example 3 - Comparison:
User: "Compare great white shark and tiger shark"
Process: search_by_common_name for both → get_species_attributes for both → compare results → finish()

Always create artifacts when retrieving data from WoRMS.
"""


if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS Agent Server")
    print("=" * 60)
    print(f"URL: http://localhost:9999")
    print(f"Status: Ready with {10} tools")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)