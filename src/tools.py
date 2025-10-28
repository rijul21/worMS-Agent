"""
WoRMS Agent Tools
Defines all tools for interacting with the WoRMS database
"""

import asyncio
from typing import Callable, Dict
from langchain.tools import tool
from worms_api import (
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
from src.logging import (
    log_species_not_found,
    log_api_call,
    log_data_fetched,
    log_no_data,
    log_tool_error,
    log_artifact_created
)


def create_worms_tools(
    worms_logic,
    context,
    get_cached_aphia_id_func: Callable
):
    """
    Factory function to create all WoRMS tools with necessary dependencies injected.
    
    Args:
        worms_logic: WoRMS API client instance
        context: ResponseContext for creating artifacts and replies
        get_cached_aphia_id_func: Function to get cached AphiaID
    
    Returns:
        List of tool functions
    """
    
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)

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
                    api_url = worms_logic.build_synonyms_url(syn_params)
                    
                    # Add offset parameter if not first request
                    if offset > 1:
                        separator = '&' if '?' in api_url else '?'
                        api_url = f"{api_url}{separator}offset={offset}"
                    
                    # Log API call
                    await log_api_call(process, "get_species_synonyms", species_name, aphia_id, api_url)
                    
                    # Execute request
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda url=api_url: worms_logic.execute_request(url)
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
                base_api_url = worms_logic.build_synonyms_url(syn_params)
                
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)

                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                
                # Get distribution from WoRMS API
                dist_params = DistributionParams(aphia_id=aphia_id)
                api_url = worms_logic.build_distribution_url(dist_params)
                
                # Log API call
                await log_api_call(process, "get_species_distribution", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                                    
                # Get vernacular names from WoRMS API
                vern_params = VernacularParams(aphia_id=aphia_id)
                api_url = worms_logic.build_vernacular_url(vern_params)
                
                # Log API call
                await log_api_call(process, "get_vernacular_names", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                
                # Get sources from WoRMS API
                sources_params = SourcesParams(aphia_id=aphia_id)
                api_url = worms_logic.build_sources_url(sources_params)
                
                # Log API call
                await log_api_call(process, "get_literature_sources", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                
                # Get record from WoRMS API
                record_params = RecordParams(aphia_id=aphia_id)
                api_url = worms_logic.build_record_url(record_params)
                
                # Log API call
                await log_api_call(process, "get_taxonomic_record", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                
                # Get classification from WoRMS API
                class_params = ClassificationParams(aphia_id=aphia_id)
                api_url = worms_logic.build_classification_url(class_params)
                
                # Log API call
                await log_api_call(process, "get_taxonomic_classification", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                
                # Get child taxa from WoRMS API
                children_params = ChildrenParams(aphia_id=aphia_id)
                api_url = worms_logic.build_children_url(children_params)
                
                # Log API call
                await log_api_call(process, "get_child_taxa", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()
                
                # Get external IDs from WoRMS API
                ext_params = ExternalIDParams(aphia_id=aphia_id)
                api_url = worms_logic.build_external_id_url(ext_params)
                
                # Log API call
                await log_api_call(process, "get_external_ids", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    return f"Species '{species_name}' not found in WoRMS database."
                
                loop = asyncio.get_event_loop()

                # Get attributes from WoRMS API
                attr_params = AttributesParams(aphia_id=aphia_id)
                api_url = worms_logic.build_attributes_url(attr_params)
                
                # Log API call
                await log_api_call(process, "get_species_attributes", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
                api_url = worms_logic.build_vernacular_search_url(search_params)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
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
            


    return [
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
