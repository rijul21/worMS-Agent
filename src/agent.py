from typing import override, Optional  
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
    ExternalIDParams,
    AttributesParams,
    VernacularSearchParams  
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
        self.aphia_id_cache = {}
        self.cache_locks = {} 
        
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
        """Get AphiaID with caching to avoid redundant API calls"""
        
        # Check cache first 
        if species_name in self.aphia_id_cache:
            aphia_id = self.aphia_id_cache[species_name]
            await process.log(f"Using cached AphiaID {aphia_id} for {species_name}")
            return aphia_id
        
        # Get or create lock for this species
        if species_name not in self.cache_locks:
            self.cache_locks[species_name] = asyncio.Lock()
        
        # Acquire lock to prevent duplicate fetches
        async with self.cache_locks[species_name]:
            # Double-check cache (another task might have fetched while we waited for lock)
            if species_name in self.aphia_id_cache:
                aphia_id = self.aphia_id_cache[species_name]
                await process.log(f"Using cached AphiaID {aphia_id} for {species_name}")
                return aphia_id
            
            # Not in cache, fetch it
            await process.log(f"Fetching AphiaID for {species_name} (not in cache)")
            loop = asyncio.get_event_loop()
            aphia_id = await loop.run_in_executor(
                None, 
                lambda: self.worms_logic.get_species_aphia_id(species_name)
            )
            
            if aphia_id:
                self.aphia_id_cache[species_name] = aphia_id
                await process.log(f"Cached AphiaID {aphia_id} for {species_name}")
            
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

        if params.species_names:
            async with context.begin_process("Resolving species identifiers from WoRMS") as process:
                for species_name in params.species_names:
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Warning: Could not resolve {species_name}")
    
        
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
            async with context.begin_process(f"Searching WoRMS for synonyms of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)

                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
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
            async with context.begin_process(f"Searching WoRMS for distribution of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)

                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
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
            async with context.begin_process(f"Searching WoRMS for vernacular names of {species_name}") as process:
                try:
                   # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                                        
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
            async with context.begin_process(f"Searching WoRMS for literature sources of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
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
            async with context.begin_process(f"Searching WoRMS for taxonomic record of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
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
            async with context.begin_process(f"Searching WoRMS for taxonomic classification of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
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
            async with context.begin_process(f"Searching WoRMS for child taxa of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
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
                


        @tool
        async def get_external_ids(species_name: str) -> str:
            """Get external database identifiers (FishBase, GBIF, NCBI, ITIS, etc.) for a marine species.
            Useful for linking to other databases.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for external database IDs of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get external IDs from WoRMS API
                    ext_params = ExternalIDParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_external_id_url(ext_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response - it's usually just a list of ID strings
                    if isinstance(raw_response, list):
                        external_ids = raw_response
                    else:
                        external_ids = [raw_response] if raw_response else []

                    if not external_ids:
                        await process.log(f"No external database IDs found for {species_name} (AphiaID: {aphia_id})")
                        return f"No external database IDs found for {species_name}"

                    await process.log(f"Found {len(external_ids)} external database ID(s) for {species_name} from WoRMS API")

                    # Format the IDs for display
                    ids_display = ", ".join([str(id_val) for id_val in external_ids])

                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"External database IDs for {species_name} (AphiaID: {aphia_id}) - {len(external_ids)} ID(s)",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(external_ids),
                            "species": species_name
                        }
                    )

                    return f"External database IDs for {species_name}: {ids_display}. Full data in artifact."
                                
                except Exception as e:
                    await process.log(f"Error retrieving external IDs for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving external IDs: {str(e)}"
                


        @tool
        async def get_species_attributes(species_name: str) -> str:
            """Get ecological attributes and traits for a marine species.
            This includes habitat preferences, depth range, salinity, temperature, substrate, etc.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for attributes of {species_name}") as process:
                try:
                    # Get AphiaID (cached)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    if not aphia_id:
                        await process.log(f"Species '{species_name}' not found in WoRMS database")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()

                    # Get attributes from WoRMS API
                    attr_params = AttributesParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_attributes_url(attr_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    attributes = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not attributes:
                        await process.log(f"No attributes found for {species_name} (AphiaID: {aphia_id})")
                        return f"No ecological attributes found for {species_name}"
                    
                    await process.log(f"Found {len(attributes)} attribute records for {species_name} from WoRMS API")
                    
                    # Extract and flatten attributes with children
                    attr_summary = []
                    important_attrs = []

                    def extract_attribute_info(attr, depth=0):
                        """Recursively extract attribute information"""
                        if not isinstance(attr, dict):
                            return []
                        
                        results = []
                        measure_type = attr.get('measurementType', 'Unknown')
                        measure_value = attr.get('measurementValue', 'Unknown')
                        quality = attr.get('qualitystatus', '')
                        
                        # Build readable string
                        indent = "  " * depth
                        attr_str = f"{indent}{measure_type}: {measure_value}"
                        if quality == 'checked':
                            attr_str += " ✓"
                        
                        results.append(attr_str)
                        
                        # Process children recursively
                        children = attr.get('children', [])
                        if children:
                            for child in children:
                                results.extend(extract_attribute_info(child, depth + 1))
                        
                        return results

                    # Extract all attributes
                    for attr in attributes:
                        attr_lines = extract_attribute_info(attr)
                        attr_summary.extend(attr_lines)
                        
                        # Also build a simplified summary for important ones
                        measure_type = attr.get('measurementType', '')
                        measure_value = attr.get('measurementValue', '')
                        
                        if measure_type in ['IUCN Red List Category', 'Body size', 'CITES Annex']:
                            important_attrs.append(f"{measure_type}: {measure_value}")
                        
                        # Extract nested important values
                        for child in attr.get('children', []):
                            if isinstance(child, dict):
                                child_type = child.get('measurementType', '')
                                child_value = child.get('measurementValue', '')
                                if child_type == 'IUCN Red List Category':
                                    important_attrs.append(f"Conservation status: {child_value}")
                                elif child_type == 'CITES Annex':
                                    important_attrs.append(f"CITES: Annex {child_value}")

                    await process.log(f"Extracted {len(attr_summary)} attribute details for {species_name}")

                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Ecological attributes for {species_name} (AphiaID: {aphia_id}) - {len(attributes)} attributes",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(attributes),
                            "species": species_name,
                            "attribute_types": list(set([a.get('measurementType', '') for a in attributes if isinstance(a, dict)]))
                        }
                    )

                    # Build response
                    if important_attrs:
                        key_info = "; ".join(important_attrs[:5])
                        return f"{species_name} has {len(attributes)} ecological attributes. Key info: {key_info}. Full detailed data in artifact."
                    else:
                        summary_preview = "; ".join([a.split(':')[0] for a in attr_summary[:5] if ':' in a])
                        return f"{species_name} has {len(attributes)} ecological attributes including: {summary_preview}. Full data in artifact."
                            
                except Exception as e:
                    await process.log(f"Error retrieving attributes for {species_name}: {type(e).__name__} - {str(e)}")
                    return f"Error retrieving attributes: {str(e)}"





        @tool
        async def search_by_common_name(common_name: str) -> str:
            """Search for marine species by their common/vernacular name.
            Use this when the user provides a common name like 'killer whale', 'great white shark', etc.
            
            Args:
                common_name: Common name to search for (e.g., "killer whale", "bottlenose dolphin")
            """
            async with context.begin_process(f"Searching WoRMS for species with common name '{common_name}'") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Search by vernacular name
                    search_params = VernacularSearchParams(vernacular_name=common_name, like=True)
                    api_url = self.worms_logic.build_vernacular_search_url(search_params)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    results = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not results:
                        await process.log(f"No species found with common name '{common_name}'")
                        return f"No species found with common name '{common_name}'. Try a different name or use scientific name."
                    
                    await process.log(f"Found {len(results)} species matching '{common_name}'")
                    
                    # Extract species info
                    species_list = []
                    for result in results[:10]:  # Limit to top 10
                        if isinstance(result, dict):
                            scientific_name = result.get('scientificname', 'Unknown')
                            aphia_id = result.get('AphiaID', 'Unknown')
                            status = result.get('status', 'Unknown')
                            authority = result.get('authority', '')
                            
                            species_info = f"{scientific_name} (AphiaID: {aphia_id}, Status: {status})"
                            if authority:
                                species_info += f" - {authority}"
                            species_list.append(species_info)
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Search results for common name '{common_name}' - {len(results)} species found",
                        uris=[api_url],
                        metadata={
                            "search_term": common_name,
                            "count": len(results),
                            "top_result": results[0].get('scientificname', '') if results else ''
                        }
                    )
                    
                    # Build response
                    if len(results) == 1:
                        sci_name = results[0].get('scientificname', 'Unknown')
                        aphia_id = results[0].get('AphiaID', 'Unknown')
                        return f"Common name '{common_name}' refers to {sci_name} (AphiaID: {aphia_id}). Full details in artifact."
                    else:
                        top_5 = "\n".join(species_list[:5])
                        return f"Found {len(results)} species with common name '{common_name}':\n{top_5}\n\nFull list in artifact."
                            
                except Exception as e:
                    await process.log(f"Error searching for common name '{common_name}': {type(e).__name__} - {str(e)}")
                    return f"Error searching for common name: {str(e)}"

        tools = [
        get_species_synonyms,
        get_species_distribution,
        get_vernacular_names,
        get_literature_sources,
        get_taxonomic_record,
        get_taxonomic_classification,
        get_child_taxa,
        get_external_ids,
        get_species_attributes,
        search_by_common_name,  
        abort,
        finish
    ]
            
        # Execute agent
        async with context.begin_process("Processing your request using WoRMS database") as process:
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
    print(f"Status: Synonyms tool ready")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)