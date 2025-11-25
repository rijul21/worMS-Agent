import asyncio
from typing import Callable
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
    VernacularSearchParams,
    AttributeKeysParams,
    AttributeValuesByCategoryParams,
    RecordsByDateParams
)
from src.logging import (
    log_species_not_found,
    log_api_call,
    log_data_fetched,
    log_no_data,
    log_tool_error,
    log_artifact_created
)


def create_worms_tools(worms_logic, context, get_cached_aphia_id_func: Callable):
    tool_call_tracker = {}
    
    def create_tracked_key(tool_name: str, **kwargs) -> str:
        """Create a unique key for tool + arguments"""
        sorted_args = sorted(kwargs.items())
        args_str = "_".join(f"{k}={v}" for k, v in sorted_args)
        return f"{tool_name}:{args_str}"
    
    @tool(return_direct=True)
    async def abort(reason: str):
        """Call if you cannot fulfill the request. Provide a clear reason why."""
        await context.reply(f"Unable to complete request: {reason}")

    @tool(return_direct=True)
    async def finish(summary: str):
        """Call when request is successfully completed. Provide a summary of findings including specific facts and mention artifacts."""
        await context.reply(summary)

    @tool
    async def get_species_synonyms(species_name: str) -> str:
        """Get synonyms and alternative scientific names for a marine species."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_species_synonyms", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for synonyms of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                all_synonyms = []
                offset = 1
                
                while True:
                    syn_params = SynonymsParams(aphia_id=aphia_id)
                    api_url = worms_logic.build_synonyms_url(syn_params)
                    
                    if offset > 1:
                        separator = '&' if '?' in api_url else '?'
                        api_url = f"{api_url}{separator}offset={offset}"
                    
                    await log_api_call(process, "get_species_synonyms", species_name, aphia_id, api_url)
                    
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda url=api_url: worms_logic.execute_request(url)
                    )
                    
                    synonyms_batch = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    
                    if not synonyms_batch:
                        break
                    
                    all_synonyms.extend(synonyms_batch)
                    
                    if len(synonyms_batch) < 50:
                        break
                    
                    offset += 50
                
                if not all_synonyms:
                    await log_no_data(process, "get_species_synonyms", species_name, aphia_id)
                    result = f"No synonyms found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_species_synonyms", species_name, len(all_synonyms))
                
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
                
                await log_artifact_created(process, "get_species_synonyms", species_name)
                result = f"Found {len(all_synonyms)} synonyms for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_species_synonyms", species_name, e)
                result = f"Error retrieving synonyms: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_species_distribution(species_name: str) -> str:
        """Get geographic distribution and range data for a marine species. Shows where the species is found globally."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_species_distribution", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for distribution of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                dist_params = DistributionParams(aphia_id=aphia_id)
                api_url = worms_logic.build_distribution_url(dist_params)
                
                await log_api_call(process, "get_species_distribution", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                distributions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not distributions:
                    await log_no_data(process, "get_species_distribution", species_name, aphia_id)
                    result = f"No distribution data found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_species_distribution", species_name, len(distributions))
                
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
                
                await log_artifact_created(process, "get_species_distribution", species_name)
                result = f"Found {len(distributions)} distribution records for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_species_distribution", species_name, e)
                result = f"Error retrieving distribution: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_vernacular_names(species_name: str) -> str:
        """Get common names for a marine species in different languages. Useful for finding local or colloquial names."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_vernacular_names", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for vernacular names of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                vern_params = VernacularParams(aphia_id=aphia_id)
                api_url = worms_logic.build_vernacular_url(vern_params)
                
                await log_api_call(process, "get_vernacular_names", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                vernaculars = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not vernaculars:
                    await log_no_data(process, "get_vernacular_names", species_name, aphia_id)
                    result = f"No vernacular names found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_vernacular_names", species_name, len(vernaculars))
                
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
                
                await log_artifact_created(process, "get_vernacular_names", species_name)
                result = f"Found {len(vernaculars)} vernacular names for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_vernacular_names", species_name, e)
                result = f"Error retrieving vernacular names: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_literature_sources(species_name: str) -> str:
        """Get scientific literature sources, references, and citations for a marine species. Provides academic sources."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_literature_sources", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for literature sources of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                sources_params = SourcesParams(aphia_id=aphia_id)
                api_url = worms_logic.build_sources_url(sources_params)
                
                await log_api_call(process, "get_literature_sources", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                sources = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not sources:
                    await log_no_data(process, "get_literature_sources", species_name, aphia_id)
                    result = f"No literature sources found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_literature_sources", species_name, len(sources))
                
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
                
                await log_artifact_created(process, "get_literature_sources", species_name)
                result = f"Found {len(sources)} literature sources for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_literature_sources", species_name, e)
                result = f"Error retrieving literature sources: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_taxonomic_record(species_name: str) -> str:
        """Get basic taxonomic record including family, order, class, status, and authority. Good for quick taxonomy overview."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_taxonomic_record", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for taxonomic record of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                record_params = RecordParams(aphia_id=aphia_id)
                api_url = worms_logic.build_record_url(record_params)
                
                await log_api_call(process, "get_taxonomic_record", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                if not isinstance(raw_response, dict):
                    await log_no_data(process, "get_taxonomic_record", species_name, aphia_id)
                    result = f"Could not retrieve taxonomic record for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_taxonomic_record", species_name, 1)
                
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
                
                await log_artifact_created(process, "get_taxonomic_record", species_name)
                result = f"Retrieved taxonomic record for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_taxonomic_record", species_name, e)
                result = f"Error retrieving taxonomic record: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_taxonomic_classification(species_name: str) -> str:
        """Get complete taxonomic hierarchy from kingdom to species. Use for full classification lineage."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_taxonomic_classification", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for taxonomic classification of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                class_params = ClassificationParams(aphia_id=aphia_id)
                api_url = worms_logic.build_classification_url(class_params)
                
                await log_api_call(process, "get_taxonomic_classification", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                classification = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not classification:
                    await log_no_data(process, "get_taxonomic_classification", species_name, aphia_id)
                    result = f"No classification data found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_taxonomic_classification", species_name, len(classification))
                
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
                
                await log_artifact_created(process, "get_taxonomic_classification", species_name)
                result = f"Found {len(classification)}-level taxonomic classification for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_taxonomic_classification", species_name, e)
                result = f"Error retrieving classification: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_child_taxa(species_name: str) -> str:
        """Get child taxa like subspecies, varieties, or forms. Returns empty for species without subspecies (normal)."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_child_taxa", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for child taxa of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                children_params = ChildrenParams(aphia_id=aphia_id)
                api_url = worms_logic.build_children_url(children_params)
                
                await log_api_call(process, "get_child_taxa", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                children = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not children:
                    await log_no_data(process, "get_child_taxa", species_name, aphia_id)
                    result = f"No child taxa found for {species_name} (this is normal for species without subspecies)"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_child_taxa", species_name, len(children))
                
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
                
                await log_artifact_created(process, "get_child_taxa", species_name)
                result = f"Found {len(children)} child taxa for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                    
            except Exception as e:
                await log_tool_error(process, "get_child_taxa", species_name, e)
                result = f"Error retrieving child taxa: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_external_ids(species_name: str) -> str:
        """Get external database identifiers (FishBase, NCBI, ITIS, etc.). Useful for cross-referencing with other databases."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_external_ids", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for external IDs of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                ext_params = ExternalIDParams(aphia_id=aphia_id)
                api_url = worms_logic.build_external_id_url(ext_params)
                
                await log_api_call(process, "get_external_ids", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                external_ids = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []

                if not external_ids:
                    await log_no_data(process, "get_external_ids", species_name, aphia_id)
                    result = f"No external database IDs found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result

                await log_data_fetched(process, "get_external_ids", species_name, len(external_ids))

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

                await log_artifact_created(process, "get_external_ids", species_name)
                result = f"Found {len(external_ids)} external database IDs for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                            
            except Exception as e:
                await log_tool_error(process, "get_external_ids", species_name, e)
                result = f"Error retrieving external IDs: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_species_attributes(species_name: str) -> str:
        """Get ecological attributes including IUCN status, CITES, body size, habitat, depth range, and environmental traits. Essential for conservation and ecology queries."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_species_attributes", species_name=species_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for attributes of {species_name}") as process:
            try:
                aphia_id = await get_cached_aphia_id_func(species_name, process)
                if not aphia_id:
                    await log_species_not_found(process, species_name)
                    result = f"Species '{species_name}' not found in WoRMS database."
                    tool_call_tracker[call_key] = result
                    return result
                
                loop = asyncio.get_event_loop()
                attr_params = AttributesParams(aphia_id=aphia_id)
                api_url = worms_logic.build_attributes_url(attr_params)
                
                await log_api_call(process, "get_species_attributes", species_name, aphia_id, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                attributes = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not attributes:
                    await log_no_data(process, "get_species_attributes", species_name, aphia_id)
                    result = f"No ecological attributes found for {species_name}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_species_attributes", species_name, len(attributes))
                
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

                await log_artifact_created(process, "get_species_attributes", species_name)
                result = f"Found {len(attributes)} ecological attributes for {species_name}."
                tool_call_tracker[call_key] = result
                return result
                        
            except Exception as e:
                await log_tool_error(process, "get_species_attributes", species_name, e)
                result = f"Error retrieving attributes: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def search_by_common_name(common_name: str) -> str:
        """Search for species using common names like 'killer whale' or 'great white shark'. Returns matching species with scientific names."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("search_by_common_name", common_name=common_name)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for species with common name '{common_name}'") as process:
            try:
                loop = asyncio.get_event_loop()
                search_params = VernacularSearchParams(vernacular_name=common_name, like=True)
                api_url = worms_logic.build_vernacular_search_url(search_params)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                results = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not results:
                    await process.log(f"No species found with common name '{common_name}'")
                    result = f"No species found with common name '{common_name}'. Try a different name or use scientific name."
                    tool_call_tracker[call_key] = result
                    return result
                
                await process.log(f"Found {len(results)} species matching '{common_name}'")
                
                species_list = []
                for result in results[:10]:
                    if isinstance(result, dict):
                        scientific_name = result.get('scientificname', 'Unknown')
                        aphia_id = result.get('AphiaID', 'Unknown')
                        status = result.get('status', 'Unknown')
                        authority = result.get('authority', '')
                        
                        species_info = f"{scientific_name} (AphiaID: {aphia_id}, Status: {status})"
                        if authority:
                            species_info += f" - {authority}"
                        species_list.append(species_info)
                
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
                
                if len(results) == 1:
                    sci_name = results[0].get('scientificname', 'Unknown')
                    aphia_id = results[0].get('AphiaID', 'Unknown')
                    result = f"Common name '{common_name}' refers to {sci_name} (AphiaID: {aphia_id}). Full details in artifact."
                    tool_call_tracker[call_key] = result
                    return result
                else:
                    top_5 = "\n".join(species_list[:5])
                    result = f"Found {len(results)} species with common name '{common_name}':\n{top_5}\n\nFull list in artifact."
                    tool_call_tracker[call_key] = result
                    return result
                        
            except Exception as e:
                await process.log(f"Error searching for common name '{common_name}': {type(e).__name__} - {str(e)}")
                result = f"Error searching for common name: {str(e)}"
                tool_call_tracker[call_key] = result
                return result
            
    @tool
    async def get_attribute_definitions(attribute_id: int = 0, include_children: bool = True) -> str:
        """Get the tree of available attribute types in WoRMS. Shows what ecological data categories exist (use attribute_id=0 for root)."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_attribute_definitions", attribute_id=attribute_id, include_children=include_children)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for attribute definitions (ID: {attribute_id})") as process:
            try:
                loop = asyncio.get_event_loop()
                keys_params = AttributeKeysParams(attribute_id=attribute_id, include_children=include_children)
                api_url = worms_logic.build_attribute_keys_url(keys_params)
                
                await log_api_call(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", None, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                definitions = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not definitions:
                    await log_no_data(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", None)
                    result = f"No attribute definitions found for ID {attribute_id}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", len(definitions))
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Attribute definitions from WoRMS (ID: {attribute_id}) - {len(definitions)} definition(s)",
                    uris=[api_url],
                    metadata={
                        "attribute_id": attribute_id,
                        "include_children": include_children,
                        "count": len(definitions)
                    }
                )
                
                await log_artifact_created(process, "get_attribute_definitions", f"Attribute ID {attribute_id}")
                result = f"Found {len(definitions)} attribute definitions for ID {attribute_id}."
                tool_call_tracker[call_key] = result
                return result
                        
            except Exception as e:
                await log_tool_error(process, "get_attribute_definitions", f"Attribute ID {attribute_id}", e)
                result = f"Error retrieving attribute definitions: {str(e)}"
                tool_call_tracker[call_key] = result
                return result
            
    @tool
    async def get_attribute_value_options(category_id: int) -> str:
        """Get possible values for a specific attribute category. Use after get_attribute_definitions to find valid options."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_attribute_value_options", category_id=category_id)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for attribute values in category {category_id}") as process:
            try:
                loop = asyncio.get_event_loop()
                values_params = AttributeValuesByCategoryParams(category_id=category_id)
                api_url = worms_logic.build_attribute_values_by_category_url(values_params)
                
                await log_api_call(process, "get_attribute_value_options", f"Category {category_id}", None, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                values = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not values:
                    await log_no_data(process, "get_attribute_value_options", f"Category {category_id}", None)
                    result = f"No attribute values found for category {category_id}"
                    tool_call_tracker[call_key] = result
                    return result
                
                await log_data_fetched(process, "get_attribute_value_options", f"Category {category_id}", len(values))
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Attribute values for category {category_id} - {len(values)} value(s)",
                    uris=[api_url],
                    metadata={
                        "category_id": category_id,
                        "count": len(values)
                    }
                )
                
                await log_artifact_created(process, "get_attribute_value_options", f"Category {category_id}")
                result = f"Found {len(values)} attribute values for category {category_id}."
                tool_call_tracker[call_key] = result
                return result
                        
            except Exception as e:
                await log_tool_error(process, "get_attribute_value_options", f"Category {category_id}", e)
                result = f"Error retrieving attribute values: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

    @tool
    async def get_recent_species_changes(start_date: str, end_date: str = None, marine_only: bool = True, extant_only: bool = True, offset: int = 1, max_results: int = 50) -> str:
        """Get species added or modified in WoRMS during a date range. Useful for tracking new discoveries and taxonomic updates. Use ISO 8601 format (e.g., '2024-01-01T00:00:00+00:00')."""
        #checks if the tool is already called or not
        call_key = create_tracked_key("get_recent_species_changes", start_date=start_date, end_date=end_date, marine_only=marine_only, extant_only=extant_only, offset=offset, max_results=max_results)
        if call_key in tool_call_tracker:
            return tool_call_tracker[call_key]
        
        async with context.begin_process(f"Searching WoRMS for species changes since {start_date}") as process:
            try:
                loop = asyncio.get_event_loop()
                date_params = RecordsByDateParams(
                    startdate=start_date,
                    enddate=end_date,
                    marine_only=marine_only,
                    extant_only=extant_only,
                    offset=offset
                )
                api_url = worms_logic.build_records_by_date_url(date_params)
                
                await log_api_call(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", None, api_url)
                
                raw_response = await loop.run_in_executor(
                    None,
                    lambda: worms_logic.execute_request(api_url)
                )
                
                records = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                
                if not records:
                    await log_no_data(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", None)
                    result = f"No species changes found in WoRMS since {start_date}"
                    tool_call_tracker[call_key] = result
                    return result
                
                total_found = len(records)
                if len(records) > max_results:
                    await process.log(f"Limiting results to {max_results} out of {total_found} found")
                    records = records[:max_results]
                
                await log_data_fetched(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", len(records))
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Species changes since {start_date} - {len(records)} records" + (f" (limited from {total_found})" if total_found > max_results else ""),
                    uris=[api_url],
                    metadata={
                        "start_date": start_date,
                        "end_date": end_date or "today",
                        "marine_only": marine_only,
                        "extant_only": extant_only,
                        "offset": offset,
                        "count": len(records),
                        "total_found": total_found,
                        "limited_to": max_results
                    }
                )
                
                await log_artifact_created(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}")
                
                result_msg = f"Found {len(records)} species modified since {start_date}"
                if total_found > max_results:
                    result_msg += f" (limited from {total_found} total results)"
                result_msg += "."
                
                result = result_msg
                tool_call_tracker[call_key] = result
                return result
                        
            except Exception as e:
                await log_tool_error(process, "get_recent_species_changes", f"Date range {start_date} to {end_date or 'today'}", e)
                result = f"Error retrieving recent changes: {str(e)}"
                tool_call_tracker[call_key] = result
                return result

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