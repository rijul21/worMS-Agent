from typing_extensions import override
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext  
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
import asyncio
import json
from typing import Dict, Any, Optional

# Import from worms_api instead of worms_client and worms_agent
from worms_api import (
    WoRMS,
    SynonymsParams, 
    DistributionParams,
    VernacularParams,
    SourcesParams,
    RecordParams,
    ClassificationParams,
    ChildrenParams,
    NoParams
)
# Simple Agent Parameter Models - 5 endpoints (removed attributes, added 3 new)
class MarineSynonymsParams(BaseModel):
    """Parameters for getting marine species synonyms"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )

class MarineDistributionParams(BaseModel):
    """Parameters for getting marine species distribution"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )

class MarineVernacularParams(BaseModel):
    """Parameters for getting marine species vernacular/common names"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_vernaculars: Optional[bool] = Field(True,
        description="Include vernacular names in results"
    )

class MarineSourcesParams(BaseModel):
    """Parameters for getting marine species literature sources/references"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_sources: Optional[bool] = Field(True,
        description="Include literature sources in results"
    )

class MarineRecordParams(BaseModel):
    """Parameters for getting marine species basic taxonomic record"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_record: Optional[bool] = Field(True,
        description="Include taxonomic record in results"
    )

class MarineClassificationParams(BaseModel):
    """Parameters for getting marine species taxonomic classification"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_classification: Optional[bool] = Field(True,
        description="Include taxonomic classification in results"
    )

class MarineChildrenParams(BaseModel):
    """Parameters for getting marine species child taxa"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_children: Optional[bool] = Field(True,
        description="Include child taxa in results"
    )

class WoRMSiChatBioAgent:
    """The iChatBio agent implementation for WoRMS - 5 endpoint version"""

    def __init__(self):
        self.worms_logic = WoRMS()

    async def run_get_synonyms(self, context, params: MarineSynonymsParams):
        """Workflow for getting marine species synonyms"""
        async with context.begin_process(f"Getting synonyms for '{params.species_name}'") as process:
            await process.log("Synonyms search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get synonyms
                syn_params = SynonymsParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_synonyms_url(syn_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    synonyms = raw_response
                elif isinstance(raw_response, dict):
                    synonyms = [raw_response]
                else:
                    synonyms = []

                synonym_count = len(synonyms)
                
                if synonym_count > 0:
                    await process.log(f"Found {synonym_count} synonyms")
                    
                    # Extract sample synonyms for display
                    sample_synonyms = []
                    for syn in synonyms[:8]:  # Show first 8
                        if isinstance(syn, dict):
                            syn_name = syn.get('scientificname', 'Unknown')
                            syn_status = syn.get('status', '')
                            if syn_status and syn_status != 'accepted':
                                sample_synonyms.append(f"{syn_name} ({syn_status})")
                            else:
                                sample_synonyms.append(syn_name)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species synonyms for {params.species_name} (AphiaID: {aphia_id}) - {synonym_count} synonyms",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Synonyms",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "synonym_count": synonym_count
                        }
                    )
                    
                    # Create detailed reply - WORKING VERSION
                    reply = f"Found {synonym_count} synonyms for {params.species_name} (AphiaID: {aphia_id})"
                    if sample_synonyms:
                        reply += f". Examples: {', '.join(sample_synonyms[:5])}"
                        if synonym_count > 5:
                            reply += f" and {synonym_count - 5} more"
                    reply += ". I've created an artifact with all the synonyms."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No synonyms found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving synonyms: {e}")

    async def run_get_distribution(self, context, params: MarineDistributionParams):
        """Workflow for getting marine species distribution"""
        async with context.begin_process(f"Getting distribution for '{params.species_name}'") as process:
            await process.log("Distribution search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get distribution
                dist_params = DistributionParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_distribution_url(dist_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    distributions = raw_response
                elif isinstance(raw_response, dict):
                    distributions = [raw_response]
                else:
                    distributions = []

                distribution_count = len(distributions)
                
                if distribution_count > 0:
                    await process.log(f"Found {distribution_count} distribution records")
                    
                    # Extract location details for user-friendly response
                    countries = set()
                    localities = []
                    
                    for dist in distributions:
                        if isinstance(dist, dict):
                            if dist.get('country'):
                                countries.add(dist['country'])
                            if dist.get('locality'):
                                localities.append(dist['locality'])
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species distribution for {params.species_name} (AphiaID: {aphia_id}) - {distribution_count} locations",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Distribution",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "distribution_count": distribution_count
                        }
                    )
                    
                    # Create detailed user-friendly response
                    reply_parts = [f"Found distribution data for {params.species_name} (AphiaID: {aphia_id}) across {distribution_count} locations"]
                    
                    if countries:
                        country_list = sorted(list(countries))
                        if len(country_list) <= 10:
                            reply_parts.append(f"Countries/regions: {', '.join(country_list)}")
                        else:
                            reply_parts.append(f"Countries/regions: {', '.join(country_list[:10])} and {len(country_list)-10} more")
                    
                    if localities:
                        if len(localities) <= 5:
                            reply_parts.append(f"Specific locations include: {', '.join(localities)}")
                        else:
                            reply_parts.append(f"Specific locations include: {', '.join(localities[:5])} and {len(localities)-5} more")
                    
                    reply_parts.append("I've created an artifact with the complete distribution data.")
                    
                    await context.reply(". ".join(reply_parts))
                else:
                    await context.reply(f"No distribution data found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving distribution data: {e}")

    async def run_get_vernacular_names(self, context, params: MarineVernacularParams):
        """Workflow for getting marine species vernacular/common names"""
        async with context.begin_process(f"Getting vernacular names for '{params.species_name}'") as process:
            await process.log("Vernacular names search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get vernacular names
                vern_params = VernacularParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_vernacular_url(vern_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    vernaculars = raw_response
                elif isinstance(raw_response, dict):
                    vernaculars = [raw_response]
                else:
                    vernaculars = []

                vernacular_count = len(vernaculars)
                
                if vernacular_count > 0:
                    await process.log(f"Found {vernacular_count} vernacular names")
                    
                    # Extract sample vernacular names by language for display
                    sample_names = []
                    languages = set()
                    
                    for vern in vernaculars[:10]:  # Show first 10
                        if isinstance(vern, dict):
                            name = vern.get('vernacular', 'Unknown')
                            lang = vern.get('language', 'Unknown')
                            if lang != 'Unknown':
                                languages.add(lang)
                                sample_names.append(f"{name} ({lang})")
                            else:
                                sample_names.append(name)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species vernacular names for {params.species_name} (AphiaID: {aphia_id}) - {vernacular_count} names",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Vernacular Names",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "vernacular_count": vernacular_count,
                            "languages": list(languages)
                        }
                    )
                    
                    # Create detailed reply
                    reply = f"Found {vernacular_count} vernacular names for {params.species_name} (AphiaID: {aphia_id})"
                    if languages:
                        reply += f" in {len(languages)} languages ({', '.join(sorted(list(languages))[:5])}{'...' if len(languages) > 5 else ''})"
                    if sample_names:
                        reply += f". Examples: {', '.join(sample_names[:4])}"
                        if vernacular_count > 4:
                            reply += f" and {vernacular_count - 4} more"
                    reply += ". I've created an artifact with all the vernacular names."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No vernacular names found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving vernacular names: {e}")

    async def run_get_sources(self, context, params: MarineSourcesParams):
        """Workflow for getting marine species literature sources/references"""
        async with context.begin_process(f"Getting sources for '{params.species_name}'") as process:
            await process.log("Sources search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get sources
                sources_params = SourcesParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_sources_url(sources_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    sources = raw_response
                elif isinstance(raw_response, dict):
                    sources = [raw_response]
                else:
                    sources = []

                source_count = len(sources)
                
                if source_count > 0:
                    await process.log(f"Found {source_count} sources")
                    
                    # Extract sample sources for display
                    sample_sources = []
                    authors = set()
                    years = set()
                    
                    for source in sources[:8]:  # Show first 8
                        if isinstance(source, dict):
                            title = source.get('title', 'Unknown')
                            author = source.get('authors', source.get('author', ''))
                            year = source.get('year', '')
                            
                            if author:
                                authors.add(author.split(',')[0])  # First author
                            if year:
                                years.add(str(year))
                            
                            if author and year:
                                sample_sources.append(f"{author} ({year})")
                            elif title:
                                sample_sources.append(title[:50] + "..." if len(title) > 50 else title)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species literature sources for {params.species_name} (AphiaID: {aphia_id}) - {source_count} sources",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Sources",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "source_count": source_count,
                            "authors": list(authors),
                            "years": sorted(list(years))
                        }
                    )
                    
                    # Create detailed reply
                    reply = f"Found {source_count} literature sources for {params.species_name} (AphiaID: {aphia_id})"
                    if years:
                        year_range = f"{min(years)}-{max(years)}" if len(years) > 1 else list(years)[0]
                        reply += f" spanning {year_range}"
                    if sample_sources:
                        reply += f". Examples: {'; '.join(sample_sources[:3])}"
                        if source_count > 3:
                            reply += f" and {source_count - 3} more"
                    reply += ". I've created an artifact with all the literature sources."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No literature sources found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving sources: {e}")

    async def run_get_record(self, context, params: MarineRecordParams):
        """Workflow for getting marine species basic taxonomic record"""
        async with context.begin_process(f"Getting basic record for '{params.species_name}'") as process:
            await process.log("Record search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get record
                record_params = RecordParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_record_url(record_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format - this endpoint returns a single object
                if isinstance(raw_response, dict):
                    record = raw_response
                else:
                    await context.reply(f"Unexpected response format for {params.species_name} record.")
                    return

                await process.log("Found taxonomic record")
                
                # Extract key information for display
                scientific_name = record.get('scientificname', 'Unknown')
                authority = record.get('authority', '')
                status = record.get('status', 'Unknown')
                rank = record.get('rank', 'Unknown')
                kingdom = record.get('kingdom', '')
                phylum = record.get('phylum', '')
                class_name = record.get('class', '')
                order = record.get('order', '')
                family = record.get('family', '')
                genus = record.get('genus', '')
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Marine species taxonomic record for {params.species_name} (AphiaID: {aphia_id})",
                    uris=[api_url],
                    metadata={
                        "data_source": "WoRMS Taxonomic Record",
                        "aphia_id": aphia_id,
                        "scientific_name": scientific_name,
                        "status": status,
                        "rank": rank,
                        "taxonomy": {
                            "kingdom": kingdom,
                            "phylum": phylum,
                            "class": class_name,
                            "order": order,
                            "family": family,
                            "genus": genus
                        }
                    }
                )
                
                # Create detailed user-friendly response
                reply_parts = [f"Retrieved taxonomic record for {scientific_name}"]
                
                if authority:
                    reply_parts.append(f"Authority: {authority}")
                
                reply_parts.append(f"Status: {status}, Rank: {rank}")
                
                # Build taxonomy string
                taxonomy_parts = []
                for tax_rank, tax_name in [("Kingdom", kingdom), ("Phylum", phylum), ("Class", class_name), 
                                         ("Order", order), ("Family", family), ("Genus", genus)]:
                    if tax_name:
                        taxonomy_parts.append(f"{tax_rank}: {tax_name}")
                
                if taxonomy_parts:
                    reply_parts.append(f"Taxonomy - {'; '.join(taxonomy_parts)}")
                
                reply_parts.append("I've created an artifact with the complete taxonomic record.")
                
                await context.reply(". ".join(reply_parts))

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving the taxonomic record: {e}")

    async def run_get_classification(self, context, params: MarineClassificationParams):
        """Workflow for getting marine species taxonomic classification"""
        async with context.begin_process(f"Getting classification for '{params.species_name}'") as process:
            await process.log("Classification search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get classification
                class_params = ClassificationParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_classification_url(class_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    classification = raw_response
                elif isinstance(raw_response, dict):
                    classification = [raw_response]
                else:
                    classification = []

                classification_count = len(classification)
                
                if classification_count > 0:
                    await process.log(f"Found {classification_count} taxonomic levels")
                    
                    # Extract taxonomic hierarchy for display
                    taxonomic_levels = []
                    for level in classification:
                        if isinstance(level, dict):
                            rank = level.get('rank', 'Unknown')
                            name = level.get('scientificname', 'Unknown')
                            if rank != 'Unknown' and name != 'Unknown':
                                taxonomic_levels.append(f"{rank}: {name}")
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species taxonomic classification for {params.species_name} (AphiaID: {aphia_id}) - {classification_count} levels",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Classification",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "classification_levels": classification_count
                        }
                    )
                    
                    # Create detailed reply
                    reply = f"Found complete taxonomic classification for {params.species_name} (AphiaID: {aphia_id}) with {classification_count} hierarchical levels"
                    if taxonomic_levels:
                        reply += f". Taxonomic hierarchy: {' → '.join(taxonomic_levels[:6])}"
                        if classification_count > 6:
                            reply += f" and {classification_count - 6} more levels"
                    reply += ". I've created an artifact with the complete classification."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No taxonomic classification found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving classification: {e}")

    async def run_get_children(self, context, params: MarineChildrenParams):
        """Workflow for getting marine species child taxa"""
        async with context.begin_process(f"Getting child taxa for '{params.species_name}'") as process:
            await process.log("Children search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get children
                children_params = ChildrenParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_children_url(children_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    children = raw_response
                elif isinstance(raw_response, dict):
                    children = [raw_response]
                else:
                    children = []

                children_count = len(children)
                
                if children_count > 0:
                    await process.log(f"Found {children_count} child taxa")
                    
                    # Extract sample children for display
                    sample_children = []
                    ranks = set()
                    
                    for child in children[:8]:  # Show first 8
                        if isinstance(child, dict):
                            child_name = child.get('scientificname', 'Unknown')
                            child_rank = child.get('rank', 'Unknown')
                            child_status = child.get('status', '')
                            
                            if child_rank != 'Unknown':
                                ranks.add(child_rank)
                            
                            if child_status and child_status == 'accepted':
                                sample_children.append(f"{child_name} ({child_rank})")
                            else:
                                sample_children.append(child_name)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species child taxa for {params.species_name} (AphiaID: {aphia_id}) - {children_count} children",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Child Taxa",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "children_count": children_count,
                            "ranks_found": list(ranks)
                        }
                    )
                    
                    # Create detailed reply
                    reply = f"Found {children_count} child taxa for {params.species_name} (AphiaID: {aphia_id})"
                    if ranks:
                        reply += f" at taxonomic levels: {', '.join(sorted(list(ranks)))}"
                    if sample_children:
                        reply += f". Examples: {', '.join(sample_children[:5])}"
                        if children_count > 5:
                            reply += f" and {children_count - 5} more"
                    reply += ". I've created an artifact with all the child taxa."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No child taxa found for {params.species_name} in WoRMS. This may indicate it's a terminal taxonomic unit (species level) with no subspecies or varieties.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving child taxa: {e}")


                
# --- AgentCard definition with 7 endpoints ---
card = AgentCard(
    name="WoRMS Marine Species Agent",
    description="Retrieves detailed marine species information from WoRMS (World Register of Marine Species) database including synonyms, distribution, common names, literature sources, taxonomic records, classification, and child taxa.",
    icon="https://www.marinespecies.org/images/WoRMS_logo.png",
    url="http://localhost:9999",  
    entrypoints=[
        AgentEntrypoint(
            id="get_synonyms",
            description="Get synonyms and alternative names for a marine species from WoRMS.",
            parameters=MarineSynonymsParams
        ),
        AgentEntrypoint(
            id="get_distribution",
            description="Get distribution data and geographic locations for a marine species from WoRMS.",
            parameters=MarineDistributionParams
        ),
        AgentEntrypoint(
            id="get_vernacular_names",
            description="Get vernacular/common names for a marine species in different languages from WoRMS.",
            parameters=MarineVernacularParams
        ),
        AgentEntrypoint(
            id="get_sources",
            description="Get literature sources and references for a marine species from WoRMS.",
            parameters=MarineSourcesParams
        ),
        AgentEntrypoint(
            id="get_record",
            description="Get basic taxonomic record and classification for a marine species from WoRMS.",
            parameters=MarineRecordParams
        ),
        AgentEntrypoint(
            id="get_taxonomy",
            description="Get complete taxonomic classification hierarchy for a marine species from WoRMS.",
            parameters=MarineClassificationParams
        ),
        AgentEntrypoint(
            id="get_marine_info",
            description="Get child taxa (subspecies, varieties, forms) for a marine species from WoRMS.",
            parameters=MarineChildrenParams
        )
    ]
)

# --- Implement the iChatBio agent class ---
class WoRMSAgent(IChatBioAgent):
    def __init__(self):
        self.workflow_agent = WoRMSiChatBioAgent()

    @override
    def get_agent_card(self) -> AgentCard:
        """Returns the agent's metadata card."""
        return card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: BaseModel):
        """Executes the requested agent entrypoint using the provided context."""
        
        # Debug logging 
        print(f"=== WoRMS DEBUG INFO ===")
        print(f"request: {request}")
        print(f"entrypoint: {entrypoint}")
        print(f"params type: {type(params)}")
        print(f"params: {params}")
        print(f"========================")
        
        if entrypoint == "get_synonyms":
            await self.workflow_agent.run_get_synonyms(context, params)
        elif entrypoint == "get_distribution":
            await self.workflow_agent.run_get_distribution(context, params)
        elif entrypoint == "get_vernacular_names":
            await self.workflow_agent.run_get_vernacular_names(context, params)
        elif entrypoint == "get_sources":
            await self.workflow_agent.run_get_sources(context, params)
        elif entrypoint == "get_record":
            await self.workflow_agent.run_get_record(context, params)
        elif entrypoint == "get_taxonomy":
            await self.workflow_agent.run_get_classification(context, params)
        elif entrypoint == "get_marine_info":
            await self.workflow_agent.run_get_children(context, params)
        else:
            # Handle unexpected entrypoints 
            await context.reply(f"Unknown entrypoint '{entrypoint}' received. Request was: '{request}'")
            raise ValueError(f"Unsupported entrypoint: {entrypoint}")

if __name__ == "__main__":
    agent = WoRMSAgent()
    print(f"Starting iChatBio agent server for '{card.name}' at http://localhost:9999")
    print("Available endpoints:")
    for ep in card.entrypoints:
        print(f"  - {ep.id}: {ep.description}")
    run_agent_server(agent, host="0.0.0.0", port=9999)