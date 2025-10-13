from typing import override
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
import dotenv
import asyncio

from worms_api import (
    WoRMS, 
    SynonymsParams, 
    DistributionParams, 
    VernacularParams, 
    SourcesParams, 
    RecordParams, 
    ClassificationParams, 
    ChildrenParams
)

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
    
    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineResearchParams,
    ):
        """Main entry point - builds and executes the ReAct agent loop"""
        
        @tool(return_direct=True)
        async def abort(reason: str):
            """Call if you cannot fulfill the request."""
            await context.reply(f"Unable to complete request: {reason}")

        @tool(return_direct=True)
        async def finish(summary: str):
            """Call when request is successfully completed."""
            await context.reply(summary)

        @tool
        async def get_species_synonyms(species_name: str) -> str:
            """Get synonyms and alternative scientific names for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching synonyms for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get synonyms from WoRMS API
                    syn_params = SynonymsParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_synonyms_url(syn_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    synonyms = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not synonyms:
                        await process.log(f"No synonyms found for {species_name} (AphiaID: {aphia_id})")
                        return f"No synonyms found for {species_name}"
                    
                    await process.log(f"Found {len(synonyms)} synonym records for {species_name} from WoRMS API")
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Synonyms for {species_name} (AphiaID: {aphia_id}) - {len(synonyms)} total",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(synonyms),
                            "species": species_name
                        }
                    )
                    
                    # Build summary
                    samples = [s.get('scientificname', 'Unknown') for s in synonyms[:3] if isinstance(s, dict)]
                    
                    return f"Found {len(synonyms)} synonyms for {species_name}. Examples: {', '.join(samples)}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving synonyms for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving synonyms: {str(e)}"
                
        @tool
        async def get_species_distribution(species_name: str) -> str:
            """Get geographic distribution data for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching distribution for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get distribution from WoRMS API
                    from worms_api import DistributionParams
                    dist_params = DistributionParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_distribution_url(dist_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    distributions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not distributions:
                        await process.log(f"No distribution data found for {species_name} (AphiaID: {aphia_id})")
                        return f"No distribution data found for {species_name}"
                    
                    await process.log(f"Found {len(distributions)} distribution records for {species_name} from WoRMS API")
                    
                    # Extract location info
                    locations = [d.get('locality', d.get('location', 'Unknown')) for d in distributions[:5] if isinstance(d, dict)]
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Distribution for {species_name} (AphiaID: {aphia_id}) - {len(distributions)} locations",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(distributions),
                            "species": species_name
                        }
                    )
                    
                    return f"Found {len(distributions)} distribution records for {species_name}. Sample locations: {', '.join(locations)}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving distribution for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving distribution: {str(e)}"
                
        @tool
        async def get_vernacular_names(species_name: str) -> str:
            """Get common/vernacular names for a marine species in different languages.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching vernacular names for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get vernacular names from WoRMS API
                    from worms_api import VernacularParams
                    vern_params = VernacularParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_vernacular_url(vern_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    vernaculars = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not vernaculars:
                        await process.log(f"No vernacular names found for {species_name} (AphiaID: {aphia_id})")
                        return f"No vernacular names found for {species_name}"
                    
                    await process.log(f"Found {len(vernaculars)} vernacular name records for {species_name} from WoRMS API")
                    
                    # Extract sample names with languages
                    samples = []
                    languages = set()
                    for v in vernaculars[:5]:
                        if isinstance(v, dict):
                            name = v.get('vernacular', 'Unknown')
                            lang = v.get('language', 'Unknown')
                            languages.add(lang)
                            samples.append(f"{name} ({lang})")
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Vernacular names for {species_name} (AphiaID: {aphia_id}) - {len(vernaculars)} names",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(vernaculars),
                            "species": species_name,
                            "languages": list(languages)
                        }
                    )
                    
                    return f"Found {len(vernaculars)} vernacular names for {species_name} in {len(languages)} languages. Examples: {', '.join(samples)}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving vernacular names for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving vernacular names: {str(e)}"
                
        

        @tool
        async def get_literature_sources(species_name: str) -> str:
            """Get scientific literature sources and references for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching literature sources for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get sources from WoRMS API
                    from worms_api import SourcesParams
                    sources_params = SourcesParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_sources_url(sources_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    sources = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not sources:
                        await process.log(f"No literature sources found for {species_name} (AphiaID: {aphia_id})")
                        return f"No literature sources found for {species_name}"
                    
                    await process.log(f"Found {len(sources)} literature source records for {species_name} from WoRMS API")
                    
                    # Extract sample citations
                    samples = []
                    for s in sources[:3]:
                        if isinstance(s, dict):
                            title = s.get('title', s.get('reference', 'Unknown'))[:50]
                            samples.append(title + "..." if len(title) == 50 else title)
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Literature sources for {species_name} (AphiaID: {aphia_id}) - {len(sources)} sources",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(sources),
                            "species": species_name
                        }
                    )
                    
                    return f"Found {len(sources)} literature sources for {species_name}. Sample titles: {', '.join(samples)}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving literature sources for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving literature sources: {str(e)}"
                

        
        @tool
        async def get_taxonomic_record(species_name: str) -> str:
            """Get basic taxonomic record and classification for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching taxonomic record for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get record from WoRMS API
                    from worms_api import RecordParams
                    record_params = RecordParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_record_url(record_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    if not isinstance(raw_response, dict):
                        await process.log(f"Invalid record format for {species_name}")
                        return f"Could not retrieve taxonomic record for {species_name}"
                    
                    await process.log(f"Retrieved taxonomic record for {species_name} from WoRMS API")
                    
                    # Extract key taxonomic info
                    rank = raw_response.get('rank', 'Unknown')
                    status = raw_response.get('status', 'Unknown')
                    kingdom = raw_response.get('kingdom', 'Unknown')
                    phylum = raw_response.get('phylum', 'Unknown')
                    class_name = raw_response.get('class', 'Unknown')
                    order = raw_response.get('order', 'Unknown')
                    family = raw_response.get('family', 'Unknown')
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Taxonomic record for {species_name} (AphiaID: {aphia_id})",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id,
                            "species": species_name,
                            "rank": rank,
                            "status": status
                        }
                    )
                    
                    return f"Taxonomic record for {species_name}: Rank={rank}, Status={status}, Kingdom={kingdom}, Phylum={phylum}, Class={class_name}, Order={order}, Family={family}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving taxonomic record for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving taxonomic record: {str(e)}"
                

        

        @tool
        async def get_taxonomic_classification(species_name: str) -> str:
            """Get complete taxonomic classification hierarchy for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching taxonomic classification for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get classification from WoRMS API
                    from worms_api import ClassificationParams
                    class_params = ClassificationParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_classification_url(class_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    classification = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not classification:
                        await process.log(f"No classification data found for {species_name} (AphiaID: {aphia_id})")
                        return f"No classification data found for {species_name}"
                    
                    await process.log(f"Found {len(classification)} taxonomic levels for {species_name} from WoRMS API")
                    
                    # Extract taxonomic hierarchy
                    hierarchy = []
                    for level in classification[:6]:
                        if isinstance(level, dict):
                            rank = level.get('rank', 'Unknown')
                            name = level.get('scientificname', 'Unknown')
                            hierarchy.append(f"{rank}: {name}")
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Taxonomic classification for {species_name} (AphiaID: {aphia_id}) - {len(classification)} levels",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(classification),
                            "species": species_name
                        }
                    )
                    
                    return f"Found {len(classification)}-level taxonomic classification for {species_name}. Hierarchy: {' > '.join(hierarchy)}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving classification for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving classification: {str(e)}"
                

        @tool
        async def get_child_taxa(species_name: str) -> str:
            """Get child taxa (subspecies, varieties, forms) for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching child taxa for {species_name}") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Get AphiaID
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                    
                    # Get child taxa from WoRMS API
                    from worms_api import ChildrenParams
                    children_params = ChildrenParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_children_url(children_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    children = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not children:
                        await process.log(f"No child taxa found for {species_name} (AphiaID: {aphia_id})")
                        return f"No child taxa found for {species_name}. This may be a terminal taxonomic unit."
                    
                    await process.log(f"Found {len(children)} child taxa for {species_name} from WoRMS API")
                    
                    # Extract sample child names
                    samples = [c.get('scientificname', 'Unknown') for c in children[:3] if isinstance(c, dict)]
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Child taxa for {species_name} (AphiaID: {aphia_id}) - {len(children)} children",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(children),
                            "species": species_name
                        }
                    )
                    
                    return f"Found {len(children)} child taxa for {species_name}. Examples: {', '.join(samples)}. Full data available in artifact."
                        
                except Exception as e:
                    await process.log(f"Error retrieving child taxa for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving child taxa: {str(e)}"
                
        tools = [
            get_species_synonyms,
            get_species_distribution,
            get_vernacular_names,
            get_literature_sources,
            get_taxonomic_record,
            get_taxonomic_classification,
            get_child_taxa,
            abort,
            finish
        ]
        
        # Execute agent
        async with context.begin_process("Processing your request") as process:
            await process.log(f"Initializing agent for query: '{request}' with {len(tools)} available tools")
        
            
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
                await process.log(f"Agent execution failed: {type(e).__name__} - {str(e)}")
                await context.reply(f"An error occurred: {str(e)}")
                
    
    def _make_system_prompt(self, species_names: list[str], user_request: str) -> str:
        species_context = f"\n\nSpecies to research: {', '.join(species_names)}" if species_names else ""
    
        return f"""\
You are a marine biology research assistant with access to the WoRMS database.

Request: "{user_request}"{species_context}

INSTRUCTIONS:
1. PLAN YOUR APPROACH:
   - For comparison queries, decide which data points to compare
   - For each species, call tools in this order: taxonomy → distribution → names → sources
   - Avoid calling tools you don't need for the specific request

2. TOOL USAGE GUIDELINES:
   - get_taxonomic_record: Basic taxonomy (rank, status, kingdom, phylum, class, order, family)
   - get_taxonomic_classification: Full taxonomic hierarchy (use for detailed taxonomy)
   - get_species_distribution: Geographic distribution data
   - get_vernacular_names: Common names in different languages
   - get_species_synonyms: Alternative scientific names
   - get_literature_sources: Scientific references (only if explicitly needed)
   - get_child_taxa: Subspecies/varieties (may return empty for terminal species - this is normal)

3. ERROR HANDLING:
   - If child taxa returns an error or empty result, this is NORMAL for terminal species
   - Don't retry failed calls
   - Continue with other data gathering

4. COMPARISON REQUIREMENTS:
   - When comparing multiple species, provide comparative insights:
     * Which has wider distribution?
     * Which is more studied (more literature)?
     * What are taxonomic relationships?
     * Any conservation status differences?
   - Don't just list facts - provide analysis

5. EFFICIENCY:
   - Only call tools that are relevant to the user's request
   - If user asks for "everything", call all relevant tools
   - If user asks for specific info (e.g., "distribution"), only call those tools

6. FINISHING:
   - Call finish() with a comprehensive summary that ANSWERS the user's question
   - For comparisons, include comparative analysis, not just individual descriptions
   - Highlight key differences and similarities

Always create artifacts when retrieving data from WoRMS.
"""

if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS Agent Server")
    print("=" * 60)
    print(f"URL: http://localhost:9999")
    print(f"Status: Synonyms tool ready")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)