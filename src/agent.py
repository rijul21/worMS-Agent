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
    ChildrenParams,
    #  for taxonomic info
    RecordFullParams,
    TaxonRanksByIDParams,
    TaxonRanksByNameParams,
    RecordsByTaxonRankIDParams,
    IDByNameParams
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
    async def get_taxonomic_info(
        species_name: str,
        include_basic_record: bool = True,
        include_full_record: bool = False,
        include_classification: bool = True,
        include_children: bool = True
    ) -> str:
        """Get comprehensive taxonomic information for a marine species.
        
        This retrieves taxonomic classification, hierarchy, and relationships including:
        - Basic or full detailed taxonomic record
        - Complete classification hierarchy (Kingdom to Species)
        - Child taxa (subspecies, varieties, forms)
        - Taxonomic rank information
        
        Args:
            species_name: Scientific name (e.g., "Orcinus orca")
            include_basic_record: Include basic taxonomic record (default: True)
            include_full_record: Include detailed full record with extra metadata (default: False)
            include_classification: Include full classification hierarchy (default: True)
            include_children: Include child taxa if available (default: True)
        """
        async with context.begin_process(f"Fetching taxonomic information for {species_name}") as process:
            try:
                loop = asyncio.get_event_loop()
                
                # Get AphiaID first
                aphia_id = await loop.run_in_executor(
                    None, 
                    lambda: self.worms_logic.get_species_aphia_id(species_name)
                )
                
                if not aphia_id:
                    await process.log(f"Species '{species_name}' not found in WoRMS database")
                    return f"Species '{species_name}' not found in WoRMS database."
                
                await process.log(f"Retrieved AphiaID {aphia_id} for species {species_name}")
                
                # Collect all data
                all_data = {}
                total_items = 0
                
                # 1. Basic Record
                if include_basic_record:
                    record_params = RecordParams(aphia_id=aphia_id)
                    record_url = self.worms_logic.build_record_url(record_params)
                    record_data = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(record_url)
                    )
                    if record_data:
                        all_data['basic_record'] = record_data
                        await process.log(f"Retrieved basic taxonomic record")
                
                # 2. Full Record
                if include_full_record:
                    full_params = RecordFullParams(aphia_id=aphia_id)
                    full_url = self.worms_logic.build_record_full_url(full_params)
                    full_data = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(full_url)
                    )
                    if full_data:
                        all_data['full_record'] = full_data
                        await process.log(f"Retrieved full detailed record")
                
                # 3. Classification
                if include_classification:
                    class_params = ClassificationParams(aphia_id=aphia_id)
                    class_url = self.worms_logic.build_classification_url(class_params)
                    class_data = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(class_url)
                    )
                    # Normalize to list
                    classification = class_data if isinstance(class_data, list) else [class_data] if class_data else []
                    if classification:
                        all_data['classification'] = classification
                        total_items += len(classification)
                        await process.log(f"Retrieved {len(classification)}-level classification hierarchy")
                
                # 4. Children
                if include_children:
                    children_params = ChildrenParams(aphia_id=aphia_id)
                    children_url = self.worms_logic.build_children_url(children_params)
                    children_data = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(children_url)
                    )
                    # Normalize to list
                    children = children_data if isinstance(children_data, list) else [children_data] if children_data else []
                    if children:
                        all_data['children'] = children
                        total_items += len(children)
                        await process.log(f"Retrieved {len(children)} child taxa")
                
                if not all_data:
                    await process.log(f"No taxonomic data found for {species_name}")
                    return f"No taxonomic data found for {species_name}"
                
                # Create artifact with all data
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxonomic information for {species_name} (AphiaID: {aphia_id})",
                    uris=[],  # Could add all URLs used
                    metadata={
                        "aphia_id": aphia_id,
                        "species": species_name,
                        "data_types": list(all_data.keys()),
                        "total_items": total_items
                    }
                )
                
                # Build summary
                summary_parts = []
                
                if 'basic_record' in all_data:
                    record = all_data['basic_record']
                    rank = record.get('rank', 'Unknown')
                    status = record.get('status', 'Unknown')
                    summary_parts.append(f"Rank: {rank}, Status: {status}")
                
                if 'classification' in all_data:
                    hierarchy = []
                    for level in all_data['classification'][:4]:
                        if isinstance(level, dict):
                            rank = level.get('rank', 'Unknown')
                            name = level.get('scientificname', 'Unknown')
                            hierarchy.append(f"{rank}: {name}")
                    summary_parts.append(f"Classification: {' > '.join(hierarchy)}")
                
                if 'children' in all_data:
                    child_samples = [c.get('scientificname', 'Unknown') for c in all_data['children'][:3] if isinstance(c, dict)]
                    summary_parts.append(f"{len(all_data['children'])} child taxa. Examples: {', '.join(child_samples)}")
                
                summary = f"Taxonomic information for {species_name}: " + "; ".join(summary_parts) + ". Full data available in artifact."
                
                return summary
                    
            except Exception as e:
                await process.log(f"Error retrieving taxonomic info for {species_name}: {type(e).__name__} - {str(e)}")
                return f"Error retrieving taxonomic info: {str(e)}"
                
        tools = [
        get_species_synonyms,
        get_species_distribution,
        get_vernacular_names,
        get_literature_sources,
        get_taxonomic_info,  
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
        """Generate system prompt"""
        species_context = f"\n\nSpecies: {', '.join(species_names)}" if species_names else ""
        
        return f"""\
You are a marine biology assistant with access to the WoRMS database.

Request: "{user_request}"{species_context}

Instructions:
Instructions:
- Use get_species_synonyms for synonym data
- Use get_species_distribution for geographic distribution
- Use get_vernacular_names for common names in different languages
- Use get_literature_sources for scientific references
- Use get_taxonomic_info for all taxonomy data (classification, hierarchy, children, ranks)
- Call finish() with a summary when done
- Call abort() if you cannot complete the request

Always create artifacts when retrieving data.
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