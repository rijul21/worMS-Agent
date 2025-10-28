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
import json

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

"""
Example of using the logging utility in WoRMS agent
"""

from logging_utils import (
    log_cache_hit,
    log_cache_miss,
    log_cache_store,
    log_species_not_found,
    log_api_call,
    log_data_fetched,
    log_no_data,
    log_tool_error,
    log_artifact_created,
    log_agent_init,
    log_agent_error,
    log,             
    LogCategory       
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
        """Get AphiaID with caching and auto-redirect to accepted names"""
        
        # Check cache first 
        if species_name in self.aphia_id_cache:
            aphia_id = self.aphia_id_cache[species_name]
            await log_cache_hit(process, species_name, aphia_id)
            return aphia_id
        
        # Get or create lock for this species
        if species_name not in self.cache_locks:
            self.cache_locks[species_name] = asyncio.Lock()
        
        # Acquire lock to prevent duplicate fetches
        async with self.cache_locks[species_name]:
            # Double-check cache (another task might have fetched while we waited for lock)
            if species_name in self.aphia_id_cache:
                aphia_id = self.aphia_id_cache[species_name]
                await log_cache_hit(process, species_name, aphia_id)
                return aphia_id
            
            # Not in cache, fetch it
            await log_cache_miss(process, species_name)
            loop = asyncio.get_event_loop()
            aphia_id = await loop.run_in_executor(
                None, 
                lambda: self.worms_logic.get_species_aphia_id(species_name)
            )
            
            if not aphia_id:
                return None
            
            # Check if this is an accepted name, if not resolve to accepted
            accepted_name, accepted_aphia_id = await self._resolve_to_accepted_name(
                species_name, 
                aphia_id, 
                process
            )
            
            # Cache both the original name and the accepted name
            self.aphia_id_cache[species_name] = accepted_aphia_id
            await log_cache_store(process, species_name, accepted_aphia_id)
            
            # If the accepted name is different, also cache it
            if accepted_name != species_name:
                self.aphia_id_cache[accepted_name] = accepted_aphia_id
                await log_cache_store(process, accepted_name, accepted_aphia_id)
            
            return accepted_aphia_id

    async def _resolve_to_accepted_name(self, species_name: str, aphia_id: int, process) -> tuple[str, int]:
        """
        Check if a species name is accepted, and if not, resolve to the accepted name.
        
        Args:
            species_name: The scientific name to check
            aphia_id: The AphiaID of the species
            process: The process context for logging
        
        Returns:
            tuple: (accepted_species_name, accepted_aphia_id)
        """
        try:
            loop = asyncio.get_event_loop()
            
            # Get the full record to check status
            record_params = RecordParams(aphia_id=aphia_id)
            api_url = self.worms_logic.build_record_url(record_params)
            
            await log(
                process,
                f"Checking taxonomic status of {species_name}",
                LogCategory.TOOL,
                data={"aphia_id": aphia_id}
            )
            
            record = await loop.run_in_executor(
                None,
                lambda: self.worms_logic.execute_request(api_url)
            )
            
            if not isinstance(record, dict):
                # Can't check status, return original
                return species_name, aphia_id
            
            status = record.get('status', '').lower()
            
            # If accepted, return as-is
            if status == 'accepted':
                await log(
                    process,
                    f"{species_name} is an accepted name",
                    LogCategory.TOOL,
                    data={"aphia_id": aphia_id, "status": "accepted"}
                )
                return species_name, aphia_id
            
            # If unaccepted, get the valid/accepted name
            valid_aphia_id = record.get('valid_AphiaID')
            valid_name = record.get('valid_name')
            
            if valid_aphia_id and valid_name:
                await log(
                    process,
                    f"Redirecting from unaccepted name '{species_name}' to accepted name '{valid_name}'",
                    LogCategory.TOOL,
                    data={
                        "original_name": species_name,
                        "original_aphia_id": aphia_id,
                        "original_status": status,
                        "accepted_name": valid_name,
                        "accepted_aphia_id": valid_aphia_id
                    }
                )
                return valid_name, valid_aphia_id
            else:
                # No valid name found, return original
                await log(
                    process,
                    f"{species_name} is {status} but no accepted name found",
                    LogCategory.TOOL,
                    data={"aphia_id": aphia_id, "status": status}
                )
                return species_name, aphia_id
                
        except Exception as e:
            await log(
                process,
                f"Error checking taxonomic status: {str(e)}",
                LogCategory.TOOL,
                data={"error": str(e)}
            )
            # On error, return original
            return species_name, aphia_id
    
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
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)

                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Fetch all synonyms with pagination (50 records per request)
                    all_synonyms = []
                    offset = 1
                    
                    while True:
                        # Build URL with offset for pagination
                        syn_params = SynonymsParams(aphia_id=aphia_id)
                        api_url = self.worms_logic.build_synonyms_url(syn_params)
                        
                        # Add offset parameter if not first request
                        if offset > 1:
                            separator = '&' if '?' in api_url else '?'
                            api_url = f"{api_url}{separator}offset={offset}"
                        
                        # Log API call
                        await log_api_call(process, "get_species_synonyms", species_name, aphia_id, api_url)
                        
                        # Execute request
                        raw_response = await loop.run_in_executor(
                            None,
                            lambda url=api_url: self.worms_logic.execute_request(url)
                        )
                        
                        # Normalize response
                        synonyms_batch = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                        
                        # Break if no more results
                        if not synonyms_batch:
                            break
                        
                        all_synonyms.extend(synonyms_batch)
                        
                        # If we got less than 50, we've reached the end
                        if len(synonyms_batch) < 50:
                            break
                        
                        # Move to next batch
                        offset += 50
                    
                    if not all_synonyms:
                        await log_no_data(process, "get_species_synonyms", species_name, aphia_id)
                        return f"No synonyms found for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_species_synonyms", species_name, len(all_synonyms))
                    
                    # Extract sample synonym names
                    samples = [s.get('scientificname', 'Unknown') for s in all_synonyms[:5] if isinstance(s, dict)]
                    
                    # Create artifact (matching other tools - just uris, no content)
                    base_api_url = self.worms_logic.build_synonyms_url(syn_params)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Synonyms for {species_name} (AphiaID: {aphia_id}) - {len(all_synonyms)} records",
                        uris=[base_api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(all_synonyms),
                            "species": species_name
                        }
                    )
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_species_synonyms", species_name)
                                        
                    more_text = f" and {len(all_synonyms) - 5} more" if len(all_synonyms) > 5 else ""
                    
                    return f"Found {len(all_synonyms)} synonyms for {species_name}. Examples: {', '.join(samples)}{more_text}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_species_synonyms", species_name, e)
                    return f"Error retrieving synonyms: {str(e)}"


        @tool
        async def get_species_distribution(species_name: str) -> str:
            """Get geographic distribution data for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for distribution of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)

                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get distribution from WoRMS API
                    dist_params = DistributionParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_distribution_url(dist_params)
                    
                    # Log API call
                    await log_api_call(process, "get_species_distribution", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    distributions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not distributions:
                        await log_no_data(process, "get_species_distribution", species_name, aphia_id)
                        return f"No distribution data found for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_species_distribution", species_name, len(distributions))
                    
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
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_species_distribution", species_name)
                    
                    return f"Found {len(distributions)} distribution records for {species_name}. Sample locations: {', '.join(locations)}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_species_distribution", species_name, e)
                    return f"Error retrieving distribution: {str(e)}"


        @tool
        async def get_vernacular_names(species_name: str) -> str:
            """Get common/vernacular names for a marine species in different languages.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for vernacular names of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                                        
                    # Get vernacular names from WoRMS API
                    vern_params = VernacularParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_vernacular_url(vern_params)
                    
                    # Log API call
                    await log_api_call(process, "get_vernacular_names", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    vernaculars = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not vernaculars:
                        await log_no_data(process, "get_vernacular_names", species_name, aphia_id)
                        return f"No vernacular names found for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_vernacular_names", species_name, len(vernaculars))
                    
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
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_vernacular_names", species_name)
                    
                    return f"Found {len(vernaculars)} vernacular names for {species_name} in {len(languages)} languages. Examples: {', '.join(samples)}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_vernacular_names", species_name, e)
                    return f"Error retrieving vernacular names: {str(e)}"


        @tool
        async def get_literature_sources(species_name: str) -> str:
            """Get scientific literature sources and references for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for literature sources of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get sources from WoRMS API
                    sources_params = SourcesParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_sources_url(sources_params)
                    
                    # Log API call
                    await log_api_call(process, "get_literature_sources", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    sources = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not sources:
                        await log_no_data(process, "get_literature_sources", species_name, aphia_id)
                        return f"No literature sources found for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_literature_sources", species_name, len(sources))
                    
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
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_literature_sources", species_name)
                    
                    return f"Found {len(sources)} literature sources for {species_name}. Sample titles: {', '.join(samples)}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_literature_sources", species_name, e)
                    return f"Error retrieving literature sources: {str(e)}"
                


        @tool
        async def get_taxonomic_record(species_name: str) -> str:
            """Get basic taxonomic record and classification for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for taxonomic record of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get record from WoRMS API
                    record_params = RecordParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_record_url(record_params)
                    
                    # Log API call
                    await log_api_call(process, "get_taxonomic_record", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    if not isinstance(raw_response, dict):
                        await log_no_data(process, "get_taxonomic_record", species_name, aphia_id)
                        return f"Could not retrieve taxonomic record for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_taxonomic_record", species_name, 1)
                    
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
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_taxonomic_record", species_name)
                    
                    return f"Taxonomic record for {species_name}: Rank={rank}, Status={status}, Kingdom={kingdom}, Phylum={phylum}, Class={class_name}, Order={order}, Family={family}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_taxonomic_record", species_name, e)
                    return f"Error retrieving taxonomic record: {str(e)}"


        @tool
        async def get_taxonomic_classification(species_name: str) -> str:
            """Get complete taxonomic classification hierarchy for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for taxonomic classification of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get classification from WoRMS API
                    class_params = ClassificationParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_classification_url(class_params)
                    
                    # Log API call
                    await log_api_call(process, "get_taxonomic_classification", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    classification = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not classification:
                        await log_no_data(process, "get_taxonomic_classification", species_name, aphia_id)
                        return f"No classification data found for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_taxonomic_classification", species_name, len(classification))
                    
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
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_taxonomic_classification", species_name)
                    
                    return f"Found {len(classification)}-level taxonomic classification for {species_name}. Hierarchy: {' > '.join(hierarchy)}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_taxonomic_classification", species_name, e)
                    return f"Error retrieving classification: {str(e)}"


        @tool
        async def get_child_taxa(species_name: str) -> str:
            """Get child taxa (subspecies, varieties, forms) for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for child taxa of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get child taxa from WoRMS API
                    children_params = ChildrenParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_children_url(children_params)
                    
                    # Log API call
                    await log_api_call(process, "get_child_taxa", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    children = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not children:
                        await log_no_data(process, "get_child_taxa", species_name, aphia_id)
                        return f"No child taxa found for {species_name} (this is normal for species without subspecies)"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_child_taxa", species_name, len(children))
                    
                    # Extract child names and ranks
                    child_info = []
                    for child in children[:5]:
                        if isinstance(child, dict):
                            child_name = child.get('scientificname', 'Unknown')
                            child_rank = child.get('rank', 'Unknown')
                            child_info.append(f"{child_name} ({child_rank})")
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Child taxa for {species_name} (AphiaID: {aphia_id}) - {len(children)} taxa",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(children),
                            "species": species_name
                        }
                    )
                    
                    # Log artifact created
                    await log_artifact_created(process, "get_child_taxa", species_name)
                    
                    return f"Found {len(children)} child taxa for {species_name}. Examples: {', '.join(child_info)}. Full data available in artifact."
                        
                except Exception as e:
                    await log_tool_error(process, "get_child_taxa", species_name, e)
                    return f"Error retrieving child taxa: {str(e)}"


        @tool
        async def get_external_ids(species_name: str) -> str:
            """Get external database identifiers for a marine species (e.g., FishBase, NCBI, ITIS).
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Searching WoRMS for external IDs of {species_name}") as process:
                try:
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()
                    
                    # Get external IDs from WoRMS API
                    ext_params = ExternalIDParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_external_id_url(ext_params)
                    
                    # Log API call
                    await log_api_call(process, "get_external_ids", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    if isinstance(raw_response, list):
                        external_ids = raw_response
                    else:
                        external_ids = [raw_response] if raw_response else []

                    if not external_ids:
                        await log_no_data(process, "get_external_ids", species_name, aphia_id)
                        return f"No external database IDs found for {species_name}"

                    # Log data fetched
                    await log_data_fetched(process, "get_external_ids", species_name, len(external_ids))

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

                    # Log artifact created
                    await log_artifact_created(process, "get_external_ids", species_name)

                    return f"External database IDs for {species_name}: {ids_display}. Full data in artifact."
                                
                except Exception as e:
                    await log_tool_error(process, "get_external_ids", species_name, e)
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
                    # Get AphiaID (cached - cache logs happen inside _get_cached_aphia_id)
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await log_species_not_found(process, species_name)
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    loop = asyncio.get_event_loop()

                    # Get attributes from WoRMS API
                    attr_params = AttributesParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_attributes_url(attr_params)
                    
                    # Log API call
                    await log_api_call(process, "get_species_attributes", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    attributes = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not attributes:
                        await log_no_data(process, "get_species_attributes", species_name, aphia_id)
                        return f"No ecological attributes found for {species_name}"
                    
                    # Log data fetched
                    await log_data_fetched(process, "get_species_attributes", species_name, len(attributes))
                    
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

                    # Log artifact created
                    await log_artifact_created(process, "get_species_attributes", species_name)

                    # Build response
                    if important_attrs:
                        key_info = "; ".join(important_attrs[:5])
                        return f"{species_name} has {len(attributes)} ecological attributes. Key info: {key_info}. Full detailed data in artifact."
                    else:
                        summary_preview = "; ".join([a.split(':')[0] for a in attr_summary[:5] if ':' in a])
                        return f"{species_name} has {len(attributes)} ecological attributes including: {summary_preview}. Full data in artifact."
                            
                except Exception as e:
                    await log_tool_error(process, "get_species_attributes", species_name, e)
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
                

        @tool
        async def search_species_fuzzy(scientific_name: str) -> str:
            """Search for marine species with fuzzy/near matching for typos and spelling variations.
            Use this when the user might have misspelled a scientific name or when exact match fails.
            
            Args:
                scientific_name: Scientific name to search for (can have typos, e.g., "Orcinus orka")
            """
            async with context.begin_process(f"Searching WoRMS with fuzzy matching for '{scientific_name}'") as process:
                try:
                    loop = asyncio.get_event_loop()
                    
                    # Use fuzzy matching (like=True)
                    search_params = SpeciesSearchParams(
                        scientific_name=scientific_name,
                        like=True,  # Enable fuzzy matching
                        marine_only=True
                    )
                    api_url = self.worms_logic.build_species_search_url(search_params)
                    
                    # Log API call
                    await log_api_call(process, "search_species_fuzzy", scientific_name, 0, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    results = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not results:
                        await log_no_data(process, "search_species_fuzzy", scientific_name, 0)
                        return f"No species found matching '{scientific_name}'. Try checking the spelling or use a common name."
                    
                    # Log data fetched
                    await log_data_fetched(process, "search_species_fuzzy", scientific_name, len(results))
                    
                    # Extract species info
                    species_list = []
                    for result in results[:10]:  # Top 10 matches
                        if isinstance(result, dict):
                            sci_name = result.get('scientificname', 'Unknown')
                            aphia_id = result.get('AphiaID', 'Unknown')
                            status = result.get('status', 'Unknown')
                            authority = result.get('authority', '')
                            
                            species_info = f"{sci_name} (AphiaID: {aphia_id}, Status: {status})"
                            if authority:
                                species_info += f" - {authority}"
                            species_list.append(species_info)
                    
                    # Create artifact
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Fuzzy search results for '{scientific_name}' - {len(results)} matches found",
                        uris=[api_url],
                        metadata={
                            "search_term": scientific_name,
                            "count": len(results),
                            "fuzzy_match": True,
                            "top_result": results[0].get('scientificname', '') if results else ''
                        }
                    )
                    
                    # Log artifact created
                    await log_artifact_created(process, "search_species_fuzzy", scientific_name)
                    
                    # Build response
                    if len(results) == 1:
                        sci_name = results[0].get('scientificname', 'Unknown')
                        aphia_id = results[0].get('AphiaID', 'Unknown')
                        return f"Found match: {sci_name} (AphiaID: {aphia_id}). Did you mean this species?"
                    else:
                        top_5 = "\n".join(species_list[:5])
                        return f"Found {len(results)} possible matches for '{scientific_name}':\n{top_5}\n\nFull list in artifact. Which one did you mean?"
                            
                except Exception as e:
                    await log_tool_error(process, "search_species_fuzzy", scientific_name, e)
                    return f"Error during fuzzy search: {str(e)}"

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
            search_species_fuzzy,  
            abort,
            finish
        ]
                

   

    def _make_system_prompt(self, species_names: list[str], user_request: str) -> str:
        species_context = f"\n\nSpecies to research: {', '.join(species_names)}" if species_names else ""

        return f"""\
    You are a marine biology research assistant with access to the WoRMS (World Register of Marine Species) database.

    Request: "{user_request}"{species_context}

    CRITICAL INSTRUCTIONS:

    1. EFFICIENCY - ONLY CALL TOOLS THAT ANSWER THE USER'S SPECIFIC QUESTION:
    - If user asks ONLY about distribution → call ONLY get_species_distribution
    - If user asks ONLY about attributes → call ONLY get_species_attributes
    - If user asks ONLY about common names → call ONLY get_vernacular_names
    - If user asks ONLY about synonyms → call ONLY get_species_synonyms
    - DO NOT call multiple tools unless the user explicitly asks for multiple things
    - Read the user's question carefully and call ONLY the relevant tool(s)

    2. HANDLING NAMES:
    A. COMMON NAMES (e.g., "killer whale", "great white shark"):
        - ALWAYS call search_by_common_name FIRST
        - Once you get the scientific name, use it for subsequent tool calls
        - Examples: killer whale, great white, bottlenose dolphin, tiger shark
    
    B. SCIENTIFIC NAMES WITH TYPOS (e.g., "Orcinus orka", "Delfinus"):
        - If exact search fails, try search_species_fuzzy
        - This handles spelling mistakes and variations
        - Examples: "Orcinus orka" → "Orcinus orca", "Delfinus" → "Delphinus"
    
    C. CORRECT SCIENTIFIC NAMES (e.g., "Orcinus orca"):
        - Proceed directly to the relevant tool
        - Auto-redirect handles synonyms automatically (you don't need to do anything)

    3. TOOL SELECTION GUIDE - BE SPECIFIC:

    SEARCH & IDENTIFICATION:
    - search_by_common_name: Convert common names to scientific names
        * Examples: "killer whale" → "Orcinus orca", "great white" → "Carcharodon carcharias"
    
    - search_species_fuzzy: Fix typos in scientific names
        * Examples: "Orcinus orka" → "Orcinus orca", "Carcharodon carcarias" → "Carcharodon carcharias"
    
    TAXONOMY (use only when user asks about taxonomy/classification):
    - get_taxonomic_record: Basic taxonomy info (rank, status, kingdom, phylum, class, order, family)
        * When to use: "What's the classification?", "What family is X in?", "Is this species accepted?"
    
    - get_taxonomic_classification: Full hierarchical taxonomy tree
        * When to use: "Show me the complete taxonomy", "Full classification hierarchy"
    
    ECOLOGY & TRAITS (use only when user asks about these specific things):
    - get_species_attributes: Ecological traits, conservation status, body size, IUCN, CITES
        * When to use: "conservation status", "body size", "IUCN status", "ecological traits", "CITES"
        * DO NOT use for general "tell me about" queries
    
    GEOGRAPHY (use only when user asks about distribution/location):
    - get_species_distribution: Where the species lives geographically
        * When to use: "Where does X live?", "distribution", "geographic range", "habitat locations"
        * DO NOT use unless user explicitly asks about location/distribution
    
    NAMES (use only when user asks about names):
    - get_vernacular_names: Common names in different languages
        * When to use: "What are the common names?", "names in other languages"
    
    - get_species_synonyms: Historical/alternative scientific names
        * When to use: "What are the synonyms?", "other scientific names", "historical names"
    
    REFERENCES (use only when user explicitly asks):
    - get_literature_sources: Scientific papers and publications
        * When to use: "Show me references", "scientific literature", "publications"
        * DO NOT call unless explicitly requested
    
    OTHER:
    - get_child_taxa: Subspecies/varieties (may be empty - this is normal)
        * When to use: "Does this have subspecies?", "child taxa", "varieties"
    
    - get_external_ids: Database IDs (FishBase, NCBI, ITIS, BOLD)
        * When to use: "What's the FishBase ID?", "external database IDs"

    4. QUERY TYPE EXAMPLES - LEARN FROM THESE:

    A. SPECIFIC QUERIES (call only 1-2 tools):
    
    "What's the conservation status of killer whales?"
    → search_by_common_name("killer whale") → get_species_attributes() → finish()
    
    "Where do orcas live?"
    → search_by_common_name("orca") → get_species_distribution() → finish()
    
    "What family is Orcinus orca in?"
    → get_taxonomic_record("Orcinus orca") → finish()
    
    "Common names for Delphinus delphis?"
    → get_vernacular_names("Delphinus delphis") → finish()
    
    "Synonyms for great white shark?"
    → search_by_common_name("great white shark") → get_species_synonyms() → finish()
    
    B. COMPREHENSIVE QUERIES (call multiple tools systematically):
    
    "Tell me everything about killer whales"
    → search_by_common_name → get_taxonomic_record → get_species_attributes → 
        get_species_distribution → get_vernacular_names → get_literature_sources → finish()
    
    "Complete profile of Orcinus orca"
    → All relevant tools in order
    
    C. COMPARISON QUERIES (gather same data for each species):
    
    "Compare conservation status of killer whale and dolphin"
    → search_by_common_name for both → get_species_attributes for both → compare → finish()

    5. ERROR HANDLING:
    - If search_by_common_name returns no results → ask user for clarification or try scientific name
    - If search_species_fuzzy returns multiple matches → ask user which one they meant
    - If get_child_taxa returns empty → this is NORMAL for terminal species, don't retry
    - Don't repeatedly call the same tool if it returns empty results
    - Auto-redirect handles synonyms automatically - you don't need to do anything special

    6. STOPPING CRITERIA - KNOW WHEN TO FINISH:
    - Call each tool AT MOST ONCE per species (except search tools if needed)
    - If you have a complete answer, call finish() immediately
    - DO NOT call tools "just in case" - only call what's needed
    - If user asks a simple question, give a simple answer

    7. RESPONSE QUALITY:
    - Lead with KEY INFORMATION that directly answers the user's question
    - DON'T ask "Would you like me to...?" - just provide the answer
    - For conservation queries, extract and state IUCN status clearly
    - For size queries, mention both male and female sizes if available
    - Always mention that full data is available in artifacts
    - Be concise but comprehensive

    8. FINISHING:
    - Call finish() with a summary that DIRECTLY ANSWERS the user's question
    - Include specific facts: conservation status, sizes, locations, etc.
    - For comparisons, provide comparative insights, not just lists
    - Highlight key differences and similarities
    - Keep it concise but informative

    9. SPECIAL CASES:

    A. TYPOS IN SCIENTIFIC NAMES:
    If exact search fails and you suspect a typo:
    → Try search_species_fuzzy() → get correct name → proceed
    
    B. UNACCEPTED NAMES/SYNONYMS:
    Auto-redirect handles this automatically. Just proceed normally.
    The system will log: "Redirecting from unaccepted name X to accepted name Y"
    
    C. AMBIGUOUS COMMON NAMES:
    If search_by_common_name returns multiple species:
    → Ask user which one they meant, show the options

    REMEMBER:
    - Read the user's question carefully
    - Call ONLY the tools that answer their specific question
    - Don't call all tools just because they're available
    - Efficiency is key - less is more
    - Always create artifacts when retrieving data from WoRMS
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