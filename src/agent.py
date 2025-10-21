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
            async with context.begin_process("Resolving species identifiers") as process:
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
            async with context.begin_process(f"Fetching synonyms for {species_name}") as process:
                try:
                    await process.log(f"Request received: {species_name}")
                    
                    # Get AphiaID (cached)
                    await process.log("Retrieving AphiaID from cache or WoRMS")
                    aphia_id = await self._get_cached_aphia_id(species_name, process)

                    if not aphia_id:
                        await process.log(
                            f"Species not found in WoRMS database",
                            data={"species_name": species_name}
                        )
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(
                        "AphiaID resolved successfully",
                        data={"aphia_id": aphia_id, "species": species_name}
                    )
                    
                    loop = asyncio.get_event_loop()
                    
                    # Initialize pagination (offset-based, returns 50 per page)
                    all_synonyms = []
                    offset = 1  # WoRMS uses 1-based offset
                    page_num = 1
                    page_size = 50  # Fixed by API
                    
                    # Paginated retrieval loop
                    while True:
                        syn_params = SynonymsParams(
                            aphia_id=aphia_id,
                            offset=offset
                        )
                        api_url = self.worms_logic.build_synonyms_url(syn_params)
                        
                        await process.log(
                            f"Fetching page {page_num}",
                            data={"offset": offset, "url": api_url}
                        )
                        
                        raw_response = await loop.run_in_executor(
                            None,
                            lambda url=api_url: self.worms_logic.execute_request(url)
                        )
                        
                        page_synonyms = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                        
                        if not page_synonyms:
                            await process.log(f"Page {page_num} returned 0 results, pagination complete")
                            break
                        
                        await process.log(
                            f"Page {page_num} retrieved successfully",
                            data={
                                "records_retrieved": len(page_synonyms),
                                "total_so_far": len(all_synonyms) + len(page_synonyms)
                            }
                        )
                        
                        all_synonyms.extend(page_synonyms)
                        
                        if len(page_synonyms) < page_size:
                            break
                        
                        offset += page_size
                        page_num += 1
                    
                    await process.log(
                        "Synonym retrieval complete",
                        data={
                            "total_synonyms": len(all_synonyms),
                            "pages_fetched": page_num
                        }
                    )
                    
                    if not all_synonyms:
                        await process.log("No synonyms found for species")
                        return f"No synonyms found for {species_name}"
                    
                    # Analyze ALL fields from synonym data
                    await process.log("Analyzing synonym data")
                    
                    synonym_statuses = {}
                    taxonomic_ranks = {}
                    valid_names = set()
                    kingdoms = set()
                    phylums = set()
                    classes = set()
                    orders = set()
                    families = set()
                    genera = set()
                    
                    marine_count = 0
                    brackish_count = 0
                    freshwater_count = 0
                    terrestrial_count = 0
                    extinct_count = 0
                    
                    unaccept_reasons = {}
                    match_types = {}
                    
                    for syn in all_synonyms:
                        if isinstance(syn, dict):
                            # Status
                            status = syn.get('status', 'unknown')
                            synonym_statuses[status] = synonym_statuses.get(status, 0) + 1
                            
                            # Rank
                            rank = syn.get('rank', 'unknown')
                            taxonomic_ranks[rank] = taxonomic_ranks.get(rank, 0) + 1
                            
                            # Valid name
                            valid_name = syn.get('valid_name')
                            if valid_name:
                                valid_names.add(valid_name)
                            
                            # Taxonomy
                            kingdom = syn.get('kingdom')
                            if kingdom:
                                kingdoms.add(kingdom)
                            
                            phylum = syn.get('phylum')
                            if phylum:
                                phylums.add(phylum)
                            
                            class_name = syn.get('class')
                            if class_name:
                                classes.add(class_name)
                            
                            order = syn.get('order')
                            if order:
                                orders.add(order)
                            
                            family = syn.get('family')
                            if family:
                                families.add(family)
                            
                            genus = syn.get('genus')
                            if genus:
                                genera.add(genus)
                            
                            # Environment flags
                            if syn.get('isMarine'):
                                marine_count += 1
                            if syn.get('isBrackish'):
                                brackish_count += 1
                            if syn.get('isFreshwater'):
                                freshwater_count += 1
                            if syn.get('isTerrestrial'):
                                terrestrial_count += 1
                            if syn.get('isExtinct'):
                                extinct_count += 1
                            
                            # Unaccept reason
                            unaccept = syn.get('unacceptreason')
                            if unaccept:
                                unaccept_reasons[unaccept] = unaccept_reasons.get(unaccept, 0) + 1
                            
                            # Match type
                            match_type = syn.get('match_type')
                            if match_type:
                                match_types[match_type] = match_types.get(match_type, 0) + 1
                    
                    analysis_data = {
                        "synonym_statuses": synonym_statuses,
                        "taxonomic_ranks": taxonomic_ranks,
                        "unique_valid_names": len(valid_names),
                        "kingdoms": list(kingdoms),
                        "phylums": list(phylums),
                        "classes": list(classes),
                        "orders": list(orders),
                        "families": list(families),
                        "genera": list(genera),
                        "environment": {
                            "marine": marine_count,
                            "brackish": brackish_count,
                            "freshwater": freshwater_count,
                            "terrestrial": terrestrial_count,
                            "extinct": extinct_count
                        },
                        "unaccept_reasons": unaccept_reasons,
                        "match_types": match_types
                    }
                    
                    await process.log(
                        "Synonym analysis complete",
                        data=analysis_data
                    )
                    
                    # Create artifact
                    await process.log("Creating artifact..")
                    base_api_url = self.worms_logic.build_synonyms_url(
                        SynonymsParams(aphia_id=aphia_id)
                    )
                    
                   
                    await process.log("Creating artifact..")

                    import json
                    content_bytes = json.dumps(all_synonyms, indent=2).encode('utf-8')

                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Synonyms for {species_name} (AphiaID: {aphia_id})",
                        content=content_bytes,  # ← THE ACTUAL DATA!
                        uris=[base_api_url],
                        metadata={
                            "aphia_id": aphia_id,
                            "species": species_name,
                            "total_count": len(all_synonyms),
                            "pages_fetched": page_num,
                            **analysis_data
                        }
                    )
                    
                    # Build summary
                    samples = [s.get('scientificname', 'Unknown') for s in all_synonyms[:5] if isinstance(s, dict)]
                    status_summary = ", ".join([f"{count} {status}" for status, count in synonym_statuses.items()])
                    
                    summary_parts = [
                        f"Found {len(all_synonyms)} synonyms for {species_name} across {page_num} pages.",
                        f"Status: {status_summary}."
                    ]
                    
                    if valid_names:
                        summary_parts.append(f"Valid names: {len(valid_names)}.")
                    
                    if extinct_count > 0:
                        summary_parts.append(f"{extinct_count} extinct records.")
                    
                    summary_parts.append(f"Examples: {', '.join(samples)}.")
                    summary_parts.append("Full data available in artifact.")
                    
                    return " ".join(summary_parts)
                            
                except Exception as e:
                    await process.log(
                        "Error during synonym retrieval",
                        data={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "species_name": species_name
                        }
                    )
                    return f"Error retrieving synonyms: {str(e)}"
                        
        @tool
        async def get_species_distribution(species_name: str) -> str:
            """Get geographic distribution data for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching distribution for {species_name}") as process:
                try:
                    await process.log(f"Request received: {species_name}")
                    
                    # Get AphiaID (cached)
                    await process.log("Retrieving AphiaID from cache or WoRMS")
                    aphia_id = await self._get_cached_aphia_id(species_name, process)

                    if not aphia_id:
                        await process.log(
                            f"Species not found in WoRMS database",
                            data={"species_name": species_name}
                        )
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(
                        "AphiaID resolved successfully",
                        data={"aphia_id": aphia_id, "species": species_name}
                    )
                    
                    loop = asyncio.get_event_loop()
                    
                    # Single API call - no pagination
                    dist_params = DistributionParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_distribution_url(dist_params)
                    
                    await process.log(
                        "Fetching distribution data",
                        data={"url": api_url}
                    )
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    all_distributions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    await process.log(
                        "Distribution retrieval complete",
                        data={"total_distributions": len(all_distributions)}
                    )
                    
                    if not all_distributions:
                        await process.log("No distribution data found for species")
                        return f"No distribution data found for {species_name}"
                    
                    # Analyze distribution data - extract ALL fields from API
                    await process.log("Analyzing distribution data")
                    
                    localities = set()
                    higher_geographies = set()
                    record_statuses = {}
                    establishment_means = {}
                    invasiveness_records = {}
                    occurrence_records = {}
                    quality_statuses = {}
                    
                    # Track coordinates
                    coordinates_count = 0
                    lat_range = {"min": None, "max": None}
                    lon_range = {"min": None, "max": None}
                    
                    for dist in all_distributions:
                        if isinstance(dist, dict):
                            # Locality and geography
                            locality = dist.get('locality')
                            if locality:
                                localities.add(locality)
                            
                            higher_geo = dist.get('higherGeography')
                            if higher_geo:
                                higher_geographies.add(higher_geo)
                            
                            # Status fields
                            record_status = dist.get('recordStatus')
                            if record_status:
                                record_statuses[record_status] = record_statuses.get(record_status, 0) + 1
                            
                            est_means = dist.get('establishmentMeans')
                            if est_means:
                                establishment_means[est_means] = establishment_means.get(est_means, 0) + 1
                            
                            invasive = dist.get('invasiveness')
                            if invasive:
                                invasiveness_records[invasive] = invasiveness_records.get(invasive, 0) + 1
                            
                            occurrence = dist.get('occurrence')
                            if occurrence:
                                occurrence_records[occurrence] = occurrence_records.get(occurrence, 0) + 1
                            
                            quality = dist.get('qualityStatus')
                            if quality:
                                quality_statuses[quality] = quality_statuses.get(quality, 0) + 1
                            
                            # Coordinates
                            lat = dist.get('decimalLatitude')
                            lon = dist.get('decimalLongitude')
                            if lat is not None and lon is not None and lat != 0 and lon != 0:
                                coordinates_count += 1
                                if lat_range["min"] is None or lat < lat_range["min"]:
                                    lat_range["min"] = lat
                                if lat_range["max"] is None or lat > lat_range["max"]:
                                    lat_range["max"] = lat
                                if lon_range["min"] is None or lon < lon_range["min"]:
                                    lon_range["min"] = lon
                                if lon_range["max"] is None or lon > lon_range["max"]:
                                    lon_range["max"] = lon
                    
                    analysis_data = {
                        "unique_localities": len(localities),
                        "unique_higher_geographies": len(higher_geographies),
                        "record_statuses": record_statuses,
                        "establishment_means": establishment_means,
                        "invasiveness_records": invasiveness_records,
                        "occurrence_records": occurrence_records,
                        "quality_statuses": quality_statuses,
                        "records_with_coordinates": coordinates_count,
                        "latitude_range": lat_range if coordinates_count > 0 else None,
                        "longitude_range": lon_range if coordinates_count > 0 else None
                    }
                    
                    await process.log(
                        "Distribution analysis complete",
                        data=analysis_data
                    )
                    
                    # Create artifact
                    await process.log("Creating artifact..")

                    # Convert data to JSON bytes
                    import json
                    content_bytes = json.dumps(all_distributions, indent=2).encode('utf-8')

                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Distribution for {species_name} (AphiaID: {aphia_id})",
                        content=content_bytes,  # ← THE ACTUAL DATA!
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id,
                            "species": species_name,
                            "total_count": len(all_distributions),
                            **analysis_data
                        }
                    )
                    
                    # Build summary with key insights
                    summary_parts = [
                        f"Found {len(all_distributions)} distribution records for {species_name}."
                    ]
                    
                    if localities:
                        sample_localities = list(localities)[:3]
                        summary_parts.append(f"Localities: {len(localities)} unique ({', '.join(sample_localities)}).")
                    
                    if establishment_means:
                        est_summary = ", ".join([f"{count} {means}" for means, count in establishment_means.items()])
                        summary_parts.append(f"Establishment: {est_summary}.")
                    
                    if invasiveness_records:
                        inv_summary = ", ".join([f"{count} {inv}" for inv, count in invasiveness_records.items()])
                        summary_parts.append(f"Invasiveness: {inv_summary}.")
                    
                    if coordinates_count > 0:
                        summary_parts.append(f"{coordinates_count} records with coordinates.")
                    
                    summary_parts.append("Full data available in artifact.")
                    
                    return " ".join(summary_parts)
                            
                except Exception as e:
                    await process.log(
                        "Error during distribution retrieval",
                        data={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "species_name": species_name
                        }
                    )
                    return f"Error retrieving distribution: {str(e)}"
        @tool
        async def get_vernacular_names(species_name: str) -> str:
            """Get common/vernacular names for a marine species in different languages.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching vernacular names for {species_name}") as process:
                try:
                    await process.log(f"Request received: {species_name}")
                    
                    # Get AphiaID (cached)
                    await process.log("Retrieving AphiaID from cache or WoRMS")
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await process.log(
                            f"Species not found in WoRMS database",
                            data={"species_name": species_name}
                        )
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(
                        "AphiaID resolved successfully",
                        data={"aphia_id": aphia_id, "species": species_name}
                    )
                    
                    loop = asyncio.get_event_loop()
                    
                    # Single API call
                    vern_params = VernacularParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_vernacular_url(vern_params)
                    
                    await process.log(
                        "Fetching vernacular names data",
                        data={"url": api_url}
                    )
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    vernaculars = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    await process.log(
                        "Vernacular names retrieval complete",
                        data={"total_names": len(vernaculars)}
                    )
                    
                    if not vernaculars:
                        await process.log("No vernacular names found for species")
                        return f"No vernacular names found for {species_name}"
                    
                    # Analyze vernacular names - extract ALL fields
                    await process.log("Analyzing vernacular names data")
                    
                    languages = {}
                    language_codes = set()
                    names_by_language = {}
                    
                    for v in vernaculars:
                        if isinstance(v, dict):
                            vernacular = v.get('vernacular', 'Unknown')
                            language = v.get('language', 'Unknown')
                            language_code = v.get('language_code', 'Unknown')
                            
                            # Count by language
                            languages[language] = languages.get(language, 0) + 1
                            language_codes.add(language_code)
                            
                            # Group names by language
                            if language not in names_by_language:
                                names_by_language[language] = []
                            names_by_language[language].append(vernacular)
                    
                    analysis_data = {
                        "total_names": len(vernaculars),
                        "unique_languages": len(languages),
                        "language_breakdown": languages,
                        "language_codes": list(language_codes)
                    }
                    
                    await process.log(
                        "Vernacular names analysis complete",
                        data=analysis_data
                    )
                    
                    # Create artifact with actual data
                    await process.log("Creating artifact..")
                    
                    import json
                    content_bytes = json.dumps(vernaculars, indent=2).encode('utf-8')
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Vernacular names for {species_name} (AphiaID: {aphia_id})",
                        content=content_bytes,
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id,
                            "species": species_name,
                            **analysis_data
                        }
                    )
                    
                    # Build summary
                    top_languages = sorted(languages.items(), key=lambda x: x[1], reverse=True)[:3]
                    language_summary = ", ".join([f"{lang} ({count})" for lang, count in top_languages])
                    
                    # Get sample names from top languages
                    sample_names = []
                    for lang, _ in top_languages[:2]:
                        if lang in names_by_language and names_by_language[lang]:
                            sample_names.append(f"{names_by_language[lang][0]} ({lang})")
                    
                    summary = (
                        f"Found {len(vernaculars)} vernacular names for {species_name} "
                        f"in {len(languages)} languages. "
                        f"Top languages: {language_summary}. "
                        f"Examples: {', '.join(sample_names)}. "
                        f"Full data available in artifact."
                    )
                    
                    return summary
                            
                except Exception as e:
                    await process.log(
                        "Error during vernacular names retrieval",
                        data={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "species_name": species_name
                        }
                    )
                    return f"Error retrieving vernacular names: {str(e)}"


        @tool
        async def get_literature_sources(species_name: str) -> str:
            """Get scientific literature sources and references for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching literature sources for {species_name}") as process:
                try:
                    await process.log(f"Request received: {species_name}")
                    
                    # Get AphiaID (cached)
                    await process.log("Retrieving AphiaID from cache or WoRMS")
                    aphia_id = await self._get_cached_aphia_id(species_name, process)
                    
                    if not aphia_id:
                        await process.log(
                            f"Species not found in WoRMS database",
                            data={"species_name": species_name}
                        )
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(
                        "AphiaID resolved successfully",
                        data={"aphia_id": aphia_id, "species": species_name}
                    )
                    
                    loop = asyncio.get_event_loop()
                    
                    # Single API call
                    sources_params = SourcesParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_sources_url(sources_params)
                    
                    await process.log(
                        "Fetching literature sources data",
                        data={"url": api_url}
                    )
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    sources = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    await process.log(
                        "Literature sources retrieval complete",
                        data={"total_sources": len(sources)}
                    )
                    
                    if not sources:
                        await process.log("No literature sources found for species")
                        return f"No literature sources found for {species_name}"
                    
                    # Analyze literature sources - extract ALL fields
                    await process.log("Analyzing literature sources data")
                    
                    use_types = {}
                    sources_with_url = 0
                    sources_with_doi = 0
                    sources_with_fulltext = 0
                    sources_with_link = 0
                    
                    for source in sources:
                        if isinstance(source, dict):
                            # Count by use type
                            use = source.get('use', 'Unknown')
                            use_types[use] = use_types.get(use, 0) + 1
                            
                            # Count sources with different link types
                            if source.get('url'):
                                sources_with_url += 1
                            if source.get('doi'):
                                sources_with_doi += 1
                            if source.get('fulltext'):
                                sources_with_fulltext += 1
                            if source.get('link'):
                                sources_with_link += 1
                    
                    analysis_data = {
                        "total_sources": len(sources),
                        "use_types": use_types,
                        "sources_with_url": sources_with_url,
                        "sources_with_doi": sources_with_doi,
                        "sources_with_fulltext": sources_with_fulltext,
                        "sources_with_link": sources_with_link
                    }
                    
                    await process.log(
                        "Literature sources analysis complete",
                        data=analysis_data
                    )
                    
                    # Create artifact with actual data
                    await process.log("Creating artifact..")
                    
                    import json
                    content_bytes = json.dumps(sources, indent=2).encode('utf-8')
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Literature sources for {species_name} (AphiaID: {aphia_id})",
                        content=content_bytes,
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id,
                            "species": species_name,
                            **analysis_data
                        }
                    )
                    
                    # Build summary
                    use_summary = ", ".join([f"{count} {use}" for use, count in use_types.items()])
                    
                    # Extract sample references
                    sample_refs = []
                    for source in sources[:3]:
                        if isinstance(source, dict):
                            ref = source.get('reference', 'Unknown')[:60]
                            if len(ref) == 60:
                                ref += "..."
                            sample_refs.append(ref)
                    
                    summary_parts = [
                        f"Found {len(sources)} literature sources for {species_name}."
                    ]
                    
                    if use_types:
                        summary_parts.append(f"Types: {use_summary}.")
                    
                    if sources_with_doi > 0:
                        summary_parts.append(f"{sources_with_doi} with DOI.")
                    
                    if sources_with_fulltext > 0:
                        summary_parts.append(f"{sources_with_fulltext} with full text.")
                    
                    if sample_refs:
                        summary_parts.append(f"Sample: {sample_refs[0]}.")
                    
                    summary_parts.append("Full data available in artifact.")
                    
                    return " ".join(summary_parts)
                            
                except Exception as e:
                    await process.log(
                        "Error during literature sources retrieval",
                        data={
                            "error": str(e),
                            "error_type": type(e).__name__,
                            "species_name": species_name
                        }
                    )
                    return f"Error retrieving literature sources: {str(e)}"

            @tool
            async def get_taxonomic_record(species_name: str) -> str:
                """Get basic taxonomic record and classification for a marine species.
                
                Args:
                    species_name: Scientific name (e.g., "Orcinus orca")
                """
                async with context.begin_process(f"Fetching taxonomic record for {species_name}") as process:
                    try:
                        await process.log(f"Request received: {species_name}")
                        
                        # Get AphiaID (cached)
                        await process.log("Retrieving AphiaID from cache or WoRMS")
                        aphia_id = await self._get_cached_aphia_id(species_name, process)
                        
                        if not aphia_id:
                            await process.log(
                                f"Species not found in WoRMS database",
                                data={"species_name": species_name}
                            )
                            return f"Species '{species_name}' not found in WoRMS database."
                        
                        await process.log(
                            "AphiaID resolved successfully",
                            data={"aphia_id": aphia_id, "species": species_name}
                        )
                        
                        loop = asyncio.get_event_loop()
                        
                        # Single API call
                        record_params = RecordParams(aphia_id=aphia_id)
                        api_url = self.worms_logic.build_record_url(record_params)
                        
                        await process.log(
                            "Fetching taxonomic record data",
                            data={"url": api_url}
                        )
                        
                        raw_response = await loop.run_in_executor(
                            None,
                            lambda: self.worms_logic.execute_request(api_url)
                        )
                        
                        if not isinstance(raw_response, dict):
                            await process.log("Invalid record format received")
                            return f"Could not retrieve taxonomic record for {species_name}"
                        
                        await process.log(
                            "Taxonomic record retrieval complete",
                            data={"record_retrieved": True}
                        )
                        
                        # Extract ALL fields from the record
                        await process.log("Analyzing taxonomic record data")
                        
                        # Basic info
                        aphia_id_returned = raw_response.get('AphiaID')
                        url = raw_response.get('url', '')
                        scientific_name = raw_response.get('scientificname', 'Unknown')
                        authority = raw_response.get('authority', 'Unknown')
                        rank = raw_response.get('rank', 'Unknown')
                        status = raw_response.get('status', 'Unknown')
                        unaccept_reason = raw_response.get('unacceptreason')
                        
                        # Valid name info
                        valid_aphia_id = raw_response.get('valid_AphiaID')
                        valid_name = raw_response.get('valid_name')
                        valid_authority = raw_response.get('valid_authority')
                        
                        # Taxonomy hierarchy
                        kingdom = raw_response.get('kingdom', 'Unknown')
                        phylum = raw_response.get('phylum', 'Unknown')
                        class_name = raw_response.get('class', 'Unknown')
                        order = raw_response.get('order', 'Unknown')
                        family = raw_response.get('family', 'Unknown')
                        genus = raw_response.get('genus', 'Unknown')
                        
                        # Environment flags
                        is_marine = raw_response.get('isMarine', False)
                        is_brackish = raw_response.get('isBrackish', False)
                        is_freshwater = raw_response.get('isFreshwater', False)
                        is_terrestrial = raw_response.get('isTerrestrial', False)
                        is_extinct = raw_response.get('isExtinct', False)
                        
                        # Additional info
                        citation = raw_response.get('citation', '')
                        lsid = raw_response.get('lsid', '')
                        modified = raw_response.get('modified', '')
                        
                        # Build environment list
                        environments = []
                        if is_marine:
                            environments.append("marine")
                        if is_brackish:
                            environments.append("brackish")
                        if is_freshwater:
                            environments.append("freshwater")
                        if is_terrestrial:
                            environments.append("terrestrial")
                        
                        analysis_data = {
                            "aphia_id": aphia_id_returned,
                            "rank": rank,
                            "status": status,
                            "environments": environments,
                            "is_extinct": is_extinct,
                            "has_valid_name": valid_name is not None,
                            "taxonomy": {
                                "kingdom": kingdom,
                                "phylum": phylum,
                                "class": class_name,
                                "order": order,
                                "family": family,
                                "genus": genus
                            }
                        }
                        
                        await process.log(
                            "Taxonomic record analysis complete",
                            data=analysis_data
                        )
                        
                        # Create artifact with actual data
                        await process.log("Creating artifact..")
                        
                        import json
                        content_bytes = json.dumps(raw_response, indent=2).encode('utf-8')
                        
                        await process.create_artifact(
                            mimetype="application/json",
                            description=f"Taxonomic record for {species_name} (AphiaID: {aphia_id})",
                            content=content_bytes,
                            uris=[api_url],
                            metadata={
                                "aphia_id": aphia_id,
                                "species": species_name,
                                **analysis_data
                            }
                        )
                        
                        # Build summary
                        taxonomy_hierarchy = f"{kingdom} > {phylum} > {class_name} > {order} > {family} > {genus}"
                        env_summary = ", ".join(environments) if environments else "Unknown"
                        
                        summary_parts = [
                            f"{scientific_name} ({authority}):",
                            f"Rank: {rank}.",
                            f"Status: {status}."
                        ]
                        
                        if unaccept_reason:
                            summary_parts.append(f"Unaccepted reason: {unaccept_reason}.")
                        
                        summary_parts.append(f"Environment: {env_summary}.")
                        
                        if is_extinct:
                            summary_parts.append("Status: Extinct.")
                        
                        summary_parts.append(f"Taxonomy: {taxonomy_hierarchy}.")
                        summary_parts.append("Full data available in artifact.")
                        
                        return " ".join(summary_parts)
                                
                    except Exception as e:
                        await process.log(
                            "Error during taxonomic record retrieval",
                            data={
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "species_name": species_name
                            }
                        )
                        return f"Error retrieving taxonomic record: {str(e)}"


            @tool
            async def get_taxonomic_classification(species_name: str) -> str:
                """Get complete taxonomic classification hierarchy for a marine species.
                
                Args:
                    species_name: Scientific name (e.g., "Orcinus orca")
                """
                async with context.begin_process(f"Fetching taxonomic classification for {species_name}") as process:
                    try:
                        await process.log(f"Request received: {species_name}")
                        
                        # Get AphiaID (cached)
                        await process.log("Retrieving AphiaID from cache or WoRMS")
                        aphia_id = await self._get_cached_aphia_id(species_name, process)
                        
                        if not aphia_id:
                            await process.log(
                                f"Species not found in WoRMS database",
                                data={"species_name": species_name}
                            )
                            return f"Species '{species_name}' not found in WoRMS database."
                        
                        await process.log(
                            "AphiaID resolved successfully",
                            data={"aphia_id": aphia_id, "species": species_name}
                        )
                        
                        loop = asyncio.get_event_loop()
                        
                        # Single API call - returns nested structure
                        class_params = ClassificationParams(aphia_id=aphia_id)
                        api_url = self.worms_logic.build_classification_url(class_params)
                        
                        await process.log(
                            "Fetching taxonomic classification data",
                            data={"url": api_url}
                        )
                        
                        raw_response = await loop.run_in_executor(
                            None,
                            lambda: self.worms_logic.execute_request(api_url)
                        )
                        
                        if not isinstance(raw_response, dict):
                            await process.log("No classification data found")
                            return f"No classification data found for {species_name}"
                        
                        await process.log(
                            "Taxonomic classification retrieval complete",
                            data={"classification_retrieved": True}
                        )
                        
                        # Parse nested classification structure
                        await process.log("Analyzing taxonomic classification")
                        
                        def extract_hierarchy(node, hierarchy=[]):
                            """Recursively extract classification hierarchy"""
                            if not isinstance(node, dict):
                                return hierarchy
                            
                            aphia_id = node.get('AphiaID')
                            rank = node.get('rank', 'Unknown')
                            scientific_name = node.get('scientificname', 'Unknown')
                            
                            if aphia_id and rank and scientific_name:
                                hierarchy.append({
                                    "AphiaID": aphia_id,
                                    "rank": rank,
                                    "scientificname": scientific_name
                                })
                            
                            # Process child (next level down)
                            child = node.get('child')
                            if child:
                                return extract_hierarchy(child, hierarchy)
                            
                            return hierarchy
                        
                        classification_hierarchy = extract_hierarchy(raw_response)
                        
                        # Extract ranks present
                        ranks_present = [item['rank'] for item in classification_hierarchy]
                        
                        analysis_data = {
                            "total_levels": len(classification_hierarchy),
                            "ranks": ranks_present,
                            "root_rank": classification_hierarchy[0]['rank'] if classification_hierarchy else None,
                            "leaf_rank": classification_hierarchy[-1]['rank'] if classification_hierarchy else None
                        }
                        
                        await process.log(
                            "Taxonomic classification analysis complete",
                            data=analysis_data
                        )
                        
                        # Create artifact with actual data
                        await process.log("Creating artifact..")
                        
                        import json
                        content_bytes = json.dumps(raw_response, indent=2).encode('utf-8')
                        
                        await process.create_artifact(
                            mimetype="application/json",
                            description=f"Taxonomic classification for {species_name} (AphiaID: {aphia_id})",
                            content=content_bytes,
                            uris=[api_url],
                            metadata={
                                "aphia_id": aphia_id,
                                "species": species_name,
                                **analysis_data
                            }
                        )
                        
                        # Build summary with hierarchy
                        hierarchy_display = []
                        for item in classification_hierarchy:
                            hierarchy_display.append(f"{item['rank']}: {item['scientificname']}")
                        
                        summary = (
                            f"Found {len(classification_hierarchy)}-level taxonomic classification for {species_name}. "
                            f"Hierarchy: {' > '.join(hierarchy_display)}. "
                            f"Full data available in artifact."
                        )
                        
                        return summary
                                
                    except Exception as e:
                        await process.log(
                            "Error during taxonomic classification retrieval",
                            data={
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "species_name": species_name
                            }
                        )
                        return f"Error retrieving taxonomic classification: {str(e)}"


            @tool
            async def get_child_taxa(species_name: str) -> str:
                """Get child taxa (subspecies, varieties, forms) for a marine species.
                
                Args:
                    species_name: Scientific name (e.g., "Orcinus orca")
                """
                async with context.begin_process(f"Fetching child taxa for {species_name}") as process:
                    try:
                        await process.log(f"Request received: {species_name}")
                        
                        # Get AphiaID (cached)
                        await process.log("Retrieving AphiaID from cache or WoRMS")
                        aphia_id = await self._get_cached_aphia_id(species_name, process)
                        
                        if not aphia_id:
                            await process.log(
                                f"Species not found in WoRMS database",
                                data={"species_name": species_name}
                            )
                            return f"Species '{species_name}' not found in WoRMS database."
                        
                        await process.log(
                            "AphiaID resolved successfully",
                            data={"aphia_id": aphia_id, "species": species_name}
                        )
                        
                        loop = asyncio.get_event_loop()
                        
                        # Initialize pagination (API supports offset, returns max 50 per page)
                        all_children = []
                        offset = 1  # 1-based offset
                        page_num = 1
                        page_size = 50
                        
                        # Paginated retrieval loop
                        while True:
                            children_params = ChildrenParams(
                                aphia_id=aphia_id,
                                offset=offset
                            )
                            api_url = self.worms_logic.build_children_url(children_params)
                            
                            await process.log(
                                f"Fetching page {page_num}",
                                data={"offset": offset, "url": api_url}
                            )
                            
                            raw_response = await loop.run_in_executor(
                                None,
                                lambda url=api_url: self.worms_logic.execute_request(url)
                            )
                            
                            page_children = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                            
                            if not page_children:
                                await process.log(f"Page {page_num} returned 0 results, pagination complete")
                                break
                            
                            await process.log(
                                f"Page {page_num} retrieved successfully",
                                data={
                                    "records_retrieved": len(page_children),
                                    "total_so_far": len(all_children) + len(page_children)
                                }
                            )
                            
                            all_children.extend(page_children)
                            
                            if len(page_children) < page_size:
                                break
                            
                            offset += page_size
                            page_num += 1
                        
                        await process.log(
                            "Child taxa retrieval complete",
                            data={
                                "total_children": len(all_children),
                                "pages_fetched": page_num
                            }
                        )
                        
                        if not all_children:
                            await process.log("No child taxa found for species")
                            return f"No child taxa found for {species_name}. This may be a terminal taxonomic unit."
                        
                        # Analyze child taxa - extract ALL fields
                        await process.log("Analyzing child taxa data")
                        
                        child_ranks = {}
                        child_statuses = {}
                        marine_count = 0
                        brackish_count = 0
                        freshwater_count = 0
                        terrestrial_count = 0
                        extinct_count = 0
                        
                        for child in all_children:
                            if isinstance(child, dict):
                                # Rank
                                rank = child.get('rank', 'Unknown')
                                child_ranks[rank] = child_ranks.get(rank, 0) + 1
                                
                                # Status
                                status = child.get('status', 'Unknown')
                                child_statuses[status] = child_statuses.get(status, 0) + 1
                                
                                # Environment
                                if child.get('isMarine'):
                                    marine_count += 1
                                if child.get('isBrackish'):
                                    brackish_count += 1
                                if child.get('isFreshwater'):
                                    freshwater_count += 1
                                if child.get('isTerrestrial'):
                                    terrestrial_count += 1
                                if child.get('isExtinct'):
                                    extinct_count += 1
                        
                        analysis_data = {
                            "total_children": len(all_children),
                            "pages_fetched": page_num,
                            "rank_breakdown": child_ranks,
                            "status_breakdown": child_statuses,
                            "environment": {
                                "marine": marine_count,
                                "brackish": brackish_count,
                                "freshwater": freshwater_count,
                                "terrestrial": terrestrial_count,
                                "extinct": extinct_count
                            }
                        }
                        
                        await process.log(
                            "Child taxa analysis complete",
                            data=analysis_data
                        )
                        
                        # Create artifact with actual data
                        await process.log("Creating artifact..")
                        
                        import json
                        content_bytes = json.dumps(all_children, indent=2).encode('utf-8')
                        
                        base_api_url = self.worms_logic.build_children_url(
                            ChildrenParams(aphia_id=aphia_id)
                        )
                        
                        await process.create_artifact(
                            mimetype="application/json",
                            description=f"Child taxa for {species_name} (AphiaID: {aphia_id})",
                            content=content_bytes,
                            uris=[base_api_url],
                            metadata={
                                "aphia_id": aphia_id,
                                "species": species_name,
                                **analysis_data
                            }
                        )
                        
                        # Build summary
                        sample_names = [c.get('scientificname', 'Unknown') for c in all_children[:5] if isinstance(c, dict)]
                        rank_summary = ", ".join([f"{count} {rank}" for rank, count in child_ranks.items()])
                        
                        summary_parts = [
                            f"Found {len(all_children)} child taxa for {species_name} across {page_num} pages."
                        ]
                        
                        if child_ranks:
                            summary_parts.append(f"Ranks: {rank_summary}.")
                        
                        if extinct_count > 0:
                            summary_parts.append(f"{extinct_count} extinct.")
                        
                        summary_parts.append(f"Examples: {', '.join(sample_names)}.")
                        summary_parts.append("Full data available in artifact.")
                        
                        return " ".join(summary_parts)
                                
                    except Exception as e:
                        await process.log(
                            "Error during child taxa retrieval",
                            data={
                                "error": str(e),
                                "error_type": type(e).__name__,
                                "species_name": species_name
                            }
                        )
                        return f"Error retrieving child taxa: {str(e)}" 


        @tool
        async def get_external_ids(species_name: str) -> str:
            """Get external database identifiers (FishBase, GBIF, NCBI, ITIS, etc.) for a marine species.
            Useful for linking to other databases.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching external database IDs for {species_name}") as process:
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
            async with context.begin_process(f"Fetching attributes for {species_name}") as process:
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
            async with context.begin_process(f"Searching for species with common name '{common_name}'") as process:
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
You are a marine biology research assistant with access to the WoRMS (World Register of Marine Species) database.

Request: "{user_request}"{species_context}

CRITICAL INSTRUCTIONS:

1. HANDLING COMMON NAMES:
   - If the user provides a COMMON NAME (e.g., "killer whale", "great white shark"), ALWAYS call search_by_common_name FIRST
   - Once you get the scientific name, use it for all subsequent tool calls
   - Examples of common names: killer whale, great white, bottlenose dolphin, tiger shark, hammerhead
   - Examples of scientific names: Orcinus orca, Carcharodon carcharias, Tursiops truncatus

2. UNDERSTANDING THE USER'S REQUEST - READ CAREFULLY:
   - **SPECIFIC single-topic queries**: Call ONLY the ONE relevant tool
     * "What is the distribution?" → ONLY get_species_distribution
     * "What are the synonyms?" → ONLY get_species_synonyms
     * "What are common names?" → ONLY get_vernacular_names
     * "What's the conservation status?" → ONLY get_species_attributes
     * "What's the taxonomy?" → ONLY get_taxonomic_record OR get_taxonomic_classification
   
   - **Comprehensive queries**: Call multiple relevant tools
     * "Tell me everything about X" → call multiple tools
     * "Give me a full report on X" → call multiple tools
     * "What can you tell me about X?" → call multiple tools
   
   - **Comparison queries**: Call the same tool(s) for each species
     * "Compare distribution of X and Y" → get_species_distribution for both
     * "Compare X and Y" → call relevant tools for both species

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
     * Use for: distribution, range, geographic locations, invasiveness
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

5. EFFICIENCY & AVOIDING UNNECESSARY CALLS:
   - **READ THE USER'S QUESTION CAREFULLY** - only call tools that answer their specific question
   - If user asks ONLY about distribution, do NOT call synonyms, vernacular names, attributes, etc.
   - If user asks ONLY about synonyms, do NOT call distribution, attributes, etc.
   - Call each tool AT MOST ONCE per species (except search_by_common_name if needed)
   - Don't retry failed calls - move on or inform the user
   - If you get a complete answer, call finish() immediately
   - **DO NOT CALL ALL TOOLS BY DEFAULT** - only call what's needed to answer the question

6. EXAMPLES OF CORRECT BEHAVIOR:

   Example 1 - Distribution query:
   User: "What is the distribution of Carcinus maenas?"
   Correct: Call ONLY get_species_distribution → finish()
   Wrong: Call distribution + synonyms + vernacular + attributes + sources...
   
   Example 2 - Synonym query:
   User: "What are the synonyms for Orcinus orca?"
   Correct: Call ONLY get_species_synonyms → finish()
   Wrong: Call synonyms + distribution + attributes...
   
   Example 3 - Conservation query:
   User: "What's the conservation status of great white sharks?"
   Correct: search_by_common_name → get_species_attributes → finish()
   Wrong: Call all tools
   
   Example 4 - Comprehensive query:
   User: "Tell me everything about Tursiops truncatus"
   Correct: Call multiple relevant tools (taxonomy, attributes, distribution, names)
   
   Example 5 - Comparison query:
   User: "Compare the distribution of Orcinus orca and Tursiops truncatus"
   Correct: Call get_species_distribution for both species → finish()
   Wrong: Call all tools for both species

7. COMPARISON REQUIREMENTS:
   When comparing multiple species, provide comparative analysis:
   - Which has wider distribution?
   - Which has larger body size?
   - Conservation status differences (IUCN Red List categories)
   - Taxonomic relationships (same family/order?)
   - Which is more studied (literature count)?
   
   Don't just list facts - provide meaningful comparisons and insights.

8. RESPONSE QUALITY:
   - Lead with KEY INFORMATION that directly answers the user's question
   - DON'T ask "Would you like me to...?" - just provide the answer
   - For conservation queries, ALWAYS extract and state IUCN status if available
   - For size queries, mention both male and female sizes if available
   - Always mention that full data is available in artifacts
   - Be concise but comprehensive

9. FINISHING:
   - Call finish() with a summary that DIRECTLY ANSWERS the user's question
   - Include specific facts: conservation status, sizes, locations, etc.
   - For comparisons, include comparative insights, not just individual descriptions
   - Highlight key differences and similarities
   - Keep it concise but informative

**CRITICAL REMINDER**: The user asked a SPECIFIC question. Answer ONLY what they asked. Do not call unnecessary tools. If they ask about distribution, give them distribution. If they ask about synonyms, give them synonyms. Only call multiple tools if they explicitly ask for comprehensive information.

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