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
                
                # Create artifact
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
                
                return f"Found {len(all_synonyms)} synonyms for {species_name}."
                    
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
                    
                    # Create artifact with full data
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
                    
                    return f"Found {len(distributions)} distribution records for {species_name}."
                        
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
                
                # Create artifact
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Vernacular names for {species_name} (AphiaID: {aphia_id}) - {len(vernaculars)} names",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id, 
                        "count": len(vernaculars),
                        "species": species_name
                    }
                )
                
                # Log artifact created
                await log_artifact_created(process, "get_vernacular_names", species_name)
                
                return f"Found {len(vernaculars)} vernacular names for {species_name}."
                    
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
                
                return f"Found {len(sources)} literature sources for {species_name}."
                    
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
                
                # Create artifact
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Taxonomic record for {species_name} (AphiaID: {aphia_id})",
                    uris=[api_url],
                    metadata={
                        "aphia_id": aphia_id,
                        "species": species_name,
                        "rank": raw_response.get('rank', 'Unknown'),
                        "status": raw_response.get('status', 'Unknown')
                    }
                )
                
                # Log artifact created
                await log_artifact_created(process, "get_taxonomic_record", species_name)
                
                return f"Retrieved taxonomic record for {species_name}."
                    
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
                
                return f"Found {len(classification)}-level taxonomic classification for {species_name}."
                    
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
                
                return f"Found {len(children)} child taxa for {species_name}."
                    
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

                return f"Found {len(external_ids)} external database IDs for {species_name}."
                            
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

                return f"Found {len(attributes)} ecological attributes for {species_name}."
                        
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
            
    @tool
    async def get_attribute_definitions(attribute_id: int = 0) -> str:
        """Get the tree of attribute definitions available in WoRMS.
        This shows what types of ecological attributes can be recorded (e.g., habitat, depth, IUCN status).
        Use attribute_id=0 to get root definitions, or specify an ID to get a subtree.
        
        Args:
            attribute_id: The attribute definition ID (default: 0 for root items)
        """
        async with context.begin_process(f"Searching WoRMS for attribute definitions") as process:
            try:
                loop = asyncio.get_event_loop()
                
                # Get attribute definition tree
                from worms_api import AttributeKeysParams
                keys_params = AttributeKeysParams(attribute_id=attribute_id, include_children=True)
                api_url = worms_logic.build_attribute_keys_url(keys_params)
                
                # Log API call
                await process.log(f"Calling WoRMS API: {api_url}")
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                # Normalize response
                definitions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not definitions:
                    await process.log(f"No attribute definitions found for ID {attribute_id}")
                    return f"No attribute definitions found for ID {attribute_id}"
                
                # Log data fetched
                await process.log(f"Found {len(definitions)} attribute definition(s)")
                
                # Extract definition info
                def_list = []
                for defn in definitions[:10]:  # Show top 10
                    if isinstance(defn, dict):
                        mtype = defn.get('measurementType', 'Unknown')
                        mtype_id = defn.get('measurementTypeID', 'N/A')
                        category_id = defn.get('CategoryID', 'N/A')
                        def_list.append(f"{mtype} (TypeID: {mtype_id}, CategoryID: {category_id})")
                
                # Create artifact
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Attribute definitions from WoRMS - {len(definitions)} definition(s)",
                    uris=[api_url],
                    metadata={
                        "attribute_id": attribute_id,
                        "count": len(definitions)
                    }
                )
                
                # Log artifact created
                await process.log(f"Created artifact with {len(definitions)} definitions")
                
                summary = f"Found {len(definitions)} attribute definitions. Top items:\n"
                summary += "\n".join(def_list[:10])
                if len(definitions) > 10:
                    summary += f"\n\n...and {len(definitions) - 10} more in artifact."
                
                return summary
                        
            except Exception as e:
                await process.log(f"Error retrieving attribute definitions: {type(e).__name__} - {str(e)}")
                return f"Error retrieving attribute definitions: {str(e)}"
            
    @tool
    async def get_attribute_value_options(category_id: int) -> str:
        """Get the list of possible values for a specific attribute category.
        For example, CategoryID 7 might return values like 'benthos', 'zooplankton', 'phytoplankton'.
        Use get_attribute_definitions first to find category IDs.
        
        Args:
            category_id: The CategoryID to get value options for (e.g., 1, 7, 9)
        """
        async with context.begin_process(f"Searching WoRMS for attribute values in category {category_id}") as process:
            try:
                loop = asyncio.get_event_loop()
                
                # Get attribute values by category
                from worms_api import AttributeValuesByCategoryParams
                values_params = AttributeValuesByCategoryParams(category_id=category_id)
                api_url = worms_logic.build_attribute_values_by_category_url(values_params)
                
                # Log API call
                await process.log(f"Calling WoRMS API: {api_url}")
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                # Normalize response
                values = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not values:
                    await process.log(f"No attribute values found for category {category_id}")
                    return f"No attribute values found for category {category_id}"
                
                # Log data fetched
                await process.log(f"Found {len(values)} attribute value(s)")
                
                # Extract value info
                value_list = []
                for val in values[:20]:  # Show top 20
                    if isinstance(val, dict):
                        mvalue = val.get('measurementValue', 'Unknown')
                        mvalue_id = val.get('measurementValueID', 'N/A')
                        value_list.append(f"{mvalue} (ValueID: {mvalue_id})")
                
                # Create artifact
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Attribute values for category {category_id} - {len(values)} value(s)",
                    uris=[api_url],
                    metadata={
                        "category_id": category_id,
                        "count": len(values)
                    }
                )
                
                # Log artifact created
                await process.log(f"Created artifact with {len(values)} values")
                
                summary = f"Found {len(values)} possible values for category {category_id}:\n"
                summary += "\n".join(value_list[:20])
                if len(values) > 20:
                    summary += f"\n\n...and {len(values) - 20} more in artifact."
                
                return summary
                        
            except Exception as e:
                await process.log(f"Error retrieving attribute values: {type(e).__name__} - {str(e)}")
                return f"Error retrieving attribute values: {str(e)}"
            

    @tool
    async def get_recent_species_changes(start_date: str, end_date: str = None, max_results: int = 50) -> str:
        """Get species that were added or modified in WoRMS during a time period.
        Useful for tracking new discoveries and taxonomic updates.
        
        Args:
            start_date: Start date in ISO 8601 format (e.g., "2024-01-01T00:00:00+00:00")
            end_date: Optional end date in ISO 8601 format (defaults to today)
            max_results: Maximum number of results to return (default: 50)
        """
        async with context.begin_process(f"Searching WoRMS for species changes since {start_date}") as process:
            try:
                loop = asyncio.get_event_loop()
                
                # Get records by date
                from worms_api import RecordsByDateParams
                date_params = RecordsByDateParams(
                    startdate=start_date,
                    enddate=end_date,
                    marine_only=True,
                    extant_only=True,
                    offset=1
                )
                api_url = worms_logic.build_records_by_date_url(date_params)
                
                # Log API call
                await process.log(f"Calling WoRMS API: {api_url}")
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                # Normalize response
                records = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not records:
                    await process.log(f"No species changes found since {start_date}")
                    return f"No species changes found in WoRMS since {start_date}"
                
                # Limit results
                if len(records) > max_results:
                    await process.log(f"Limiting results to {max_results} (found {len(records)})")
                    records = records[:max_results]
                
                # Log data fetched
                await process.log(f"Found {len(records)} species modified since {start_date}")
                
                # Extract species info
                species_list = []
                for record in records[:10]:  # Show top 10 in summary
                    if isinstance(record, dict):
                        sci_name = record.get('scientificname', 'Unknown')
                        aphia_id = record.get('AphiaID', 'N/A')
                        status = record.get('status', 'Unknown')
                        modified = record.get('modified', 'N/A')
                        species_list.append(f"{sci_name} (AphiaID: {aphia_id}, Status: {status}, Modified: {modified})")
                
                # Create artifact
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Species changes since {start_date} - {len(records)} records",
                    uris=[api_url],
                    metadata={
                        "start_date": start_date,
                        "end_date": end_date or "today",
                        "count": len(records),
                        "limited_to": max_results
                    }
                )
                
                # Log artifact created
                await process.log(f"Created artifact with {len(records)} species records")
                
                summary = f"Found {len(records)} species modified since {start_date}. Top 10:\n"
                summary += "\n".join(species_list[:10])
                if len(records) > 10:
                    summary += f"\n\n...and {len(records) - 10} more in artifact."
                
                return summary
                        
            except Exception as e:
                await process.log(f"Error retrieving recent changes: {type(e).__name__} - {str(e)}")
                return f"Error retrieving recent changes: {str(e)}"
                


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
    get_attribute_definitions,      
    get_attribute_value_options,   
    get_recent_species_changes,    
    search_by_common_name,
    abort,
    finish
]
