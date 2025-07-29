import os
import json
import dotenv
import instructor
from typing_extensions import override
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import ValidationError, BaseModel
from typing import Optional, List
import httpx
import asyncio
from datetime import datetime, timezone
from urllib.parse import quote

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext  
from ichatbio.types import AgentCard, AgentEntrypoint

from worms_models import (
    WoRMSRecord,
    WoRMSSynonym,
    WoRMSDistribution,
    WoRMSVernacular,
    WoRMSSource,
    MarineQueryModel,
    MarineParameters,
    CompleteMarineSpeciesData
)

dotenv.load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    WORMS_BASE_URL = "https://www.marinespecies.org/rest"
    MODEL_NAME = "llama3-70b-8192"

class WoRMSClient:
    def __init__(self, base_url: str = Config.WORMS_BASE_URL):
        self.base_url = base_url

    async def fetch_json(self, client: httpx.AsyncClient, endpoint: str):
        try:
            url = f"{self.base_url}{endpoint}"
            resp = await client.get(url)
            if resp.status_code != 200:
                print(f"ERROR: WoRMS API {resp.status_code} for {endpoint}")
                return None
            return resp.json()
        except Exception as e:
            print(f"ERROR: WoRMS fetch failed: {str(e)[:100]}")
        return None

    async def get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str) -> Optional[List[WoRMSRecord]]:
        encoded_name = quote(scientific_name)
        endpoint = f"/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            if isinstance(data, list):
                for record in data:
                    # Fix boolean field handling
                    for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                        if bool_field in record:
                            record[bool_field] = bool(record[bool_field]) if record[bool_field] is not None else None
                records = [WoRMSRecord(**record) for record in data]
                print(f"SUCCESS: Found {len(records)} records for {scientific_name}")
                return records
            # Handle single record
            for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                if bool_field in data:
                    data[bool_field] = bool(data[bool_field]) if data[bool_field] is not None else None
            record = WoRMSRecord(**data)
            print(f"SUCCESS: Found 1 record for {scientific_name}")
            return [record]
        except ValidationError as e:
            print(f"ERROR: Validation failed for {scientific_name}: {str(e)[:100]}")
            return None

    async def get_vernaculars_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSVernacular]]:
        endpoint = f"/AphiaVernacularsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSVernacular(**record) for record in data] if isinstance(data, list) else [WoRMSVernacular(**data)]
            print(f"SUCCESS: Found {len(result)} vernacular names")
            return result
        except ValidationError:
            return None

    async def get_synonyms_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSynonym]]:
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSSynonym(**record) for record in data] if isinstance(data, list) else [WoRMSSynonym(**data)]
            print(f"SUCCESS: Found {len(result)} synonyms")
            return result
        except ValidationError:
            return None

    async def get_distributions_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSDistribution]]:
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSDistribution(**record) for record in data] if isinstance(data, list) else [WoRMSDistribution(**data)]
            print(f"SUCCESS: Found {len(result)} distributions")
            return result
        except ValidationError:
            return None

    async def get_attributes_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[dict]]:
        endpoint = f"/AphiaAttributesByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        return data if isinstance(data, list) else [data]

    async def get_sources_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSource]]:
        endpoint = f"/AphiaSourcesByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSSource(**record) for record in data] if isinstance(data, list) else [WoRMSSource(**data)]
            print(f"SUCCESS: Found {len(result)} sources")
            return result
        except ValidationError:
            return None

class MarineAgent(IChatBioAgent):
    def __init__(self):
        print("INIT: MarineAgent initialized")
        self.worms_client = WoRMSClient()

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Marine Species Data Agent",
            description="Retrieves marine species information from WoRMS, including taxonomic data, common names, synonyms, distributions, attributes, and sources.",
            url="http://18.222.189.40:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_info",
                    description="Get complete marine species data including taxonomy, synonyms, distributions, and sources",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_taxonomy",
                    description="Get basic taxonomic classification for a marine species",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Get synonyms and alternative names for a marine species",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_distribution",
                    description="Get geographic distribution data for a marine species",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_vernacular_names",
                    description="Get common names in different languages for a marine species",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_sources",
                    description="Get scientific references and sources for a marine species",
                    parameters=MarineParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: BaseModel):
        """Executes the requested agent entrypoint using the provided context."""
        
        # Debug logging matching your friend's pattern
        print(f"=== DEBUG INFO ===")
        print(f"request: {request} (type: {type(request)})")
        print(f"entrypoint: {entrypoint}")
        print(f"params type: {type(params)}")
        print(f"params: {params}")
        print(f"==================")
        
        # Convert params to the expected type
        if isinstance(params, dict):
            marine_params = MarineParameters(**params)
        elif hasattr(params, 'model_dump'):
            marine_params = MarineParameters(**params.model_dump())
        else:
            marine_params = params
        
        # Route to specific entrypoint methods
        if entrypoint == "get_marine_info":
            await self.run_get_marine_info(context, marine_params)
        elif entrypoint == "get_taxonomy":
            await self.run_get_taxonomy(context, marine_params)
        elif entrypoint == "get_synonyms":
            await self.run_get_synonyms(context, marine_params)
        elif entrypoint == "get_distribution":
            await self.run_get_distribution(context, marine_params)
        elif entrypoint == "get_vernacular_names":
            await self.run_get_vernacular_names(context, marine_params)
        elif entrypoint == "get_sources":
            await self.run_get_sources(context, marine_params)
        else:
            # Handle unexpected entrypoints 
            await context.reply(f"Unknown entrypoint '{entrypoint}' received.")
            raise ValueError(f"Unsupported entrypoint: {entrypoint}")

    async def run_get_marine_info(self, context: ResponseContext, params: MarineParameters):
        """Workflow for retrieving marine species information from WoRMS"""
        
        # Determine what we're searching for - prioritize params.species_name
        search_term = None
        
        if params and params.species_name and params.species_name.strip():
            search_term = params.species_name.strip()
            print(f"SEARCH: Using species '{search_term}' from params")
        
        if not search_term:
            print("ERROR: No species name provided")
            await context.reply("Please provide a marine species name to search for.")
            return

        async with context.begin_process(f"Retrieving marine species data for: {search_term}") as process:
            try:
                await process.log("Extracted search parameters", data=params.model_dump(exclude_defaults=True))

                # Extract query information
                marine_query = await self.extract_query_info(search_term)
                
                if not marine_query.scientificname and not marine_query.common_name:
                    await context.reply(f"Could not identify a marine species name in: '{search_term}'")
                    return
                    
                search_name = marine_query.scientificname or marine_query.common_name
                print(f"QUERY: Searching WoRMS for '{search_name}'")
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Search for species records
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    
                    if not records or len(records) == 0:
                        await context.reply(f"No marine species found matching '{search_name}' in WoRMS database.")
                        return
                        
                    primary_species = records[0]
                    aphia_id = primary_species.AphiaID
                    print(f"FOUND: {primary_species.scientificname} (AphiaID: {aphia_id})")
                    
                    # Fetch additional data based on parameters
                    vernaculars = await self.worms_client.get_vernaculars_by_aphia_id(client, aphia_id) if params.include_vernaculars else None
                    synonyms = await self.worms_client.get_synonyms_by_aphia_id(client, aphia_id) if params.include_synonyms else None
                    distributions = await self.worms_client.get_distributions_by_aphia_id(client, aphia_id) if params.include_distribution else None
                    attributes = await self.worms_client.get_attributes_by_aphia_id(client, aphia_id)
                    sources = await self.worms_client.get_sources_by_aphia_id(client, aphia_id) if params.include_sources else None
                    
                    # Compile complete species data
                    species_data = {
                        "species": primary_species,
                        "synonyms": synonyms,
                        "distribution": distributions,
                        "vernaculars": vernaculars,
                        "sources": sources,
                        "attributes": attributes,
                        "aphia_id": aphia_id,
                        "scientific_name": primary_species.scientificname,
                        "search_term": search_name
                    }
                    
                    try:
                        complete_data = CompleteMarineSpeciesData(**species_data)
                        content = complete_data.model_dump_json()
                    except ValidationError as e:
                        await context.reply(f"Error processing marine species data: {str(e)}")
                        return

                    # Count collected data for summary
                    total_vernaculars = len(vernaculars) if vernaculars else 0
                    total_synonyms = len(synonyms) if synonyms else 0
                    total_distributions = len(distributions) if distributions else 0
                    total_sources = len(sources) if sources else 0

                    print(f"ARTIFACT: Creating artifact with {total_vernaculars}V/{total_synonyms}S/{total_distributions}D/{total_sources}R")

                    # Create the artifact
                    content_bytes = content.encode('utf-8')
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Complete marine species data for {primary_species.scientificname}",
                        content=content_bytes,
                        uris=self._build_worms_uris(aphia_id),
                        metadata={
                            "data_source": "WoRMS",
                            "aphia_id": aphia_id,
                            "scientific_name": primary_species.scientificname,
                            "search_term": search_name,
                            "vernacular_count": total_vernaculars,
                            "synonym_count": total_synonyms,
                            "distribution_count": total_distributions,
                            "source_count": total_sources,
                            "retrieved_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    # Create user-friendly response like your friend's ALA agent
                    vernacular_names = [v.vernacular for v in vernaculars] if vernaculars else []
                    distribution_locations = [d.locality for d in distributions if d.locality] if distributions else []
                    
                    # Build a concise, informative response
                    if vernacular_names:
                        sample_names = ', '.join(vernacular_names[:3])
                        if len(vernacular_names) > 3:
                            sample_names += f" and {len(vernacular_names) - 3} more"
                        common_names_text = f"Common names: {sample_names}. "
                    else:
                        common_names_text = ""
                    
                    if distribution_locations:
                        sample_locations = ', '.join(distribution_locations[:3])
                        if len(distribution_locations) > 3:
                            sample_locations += f" and {len(distribution_locations) - 3} more"
                        distribution_text = f"Found in: {sample_locations}. "
                    else:
                        distribution_text = ""
                    
                    # Simple, clear response following your friend's pattern
                    response_text = (
                        f"Retrieved marine species data for {primary_species.scientificname} (AphiaID: {aphia_id}) from WoRMS. "
                        f"Taxonomic classification: {primary_species.kingdom} > {primary_species.phylum} > "
                        f"{getattr(primary_species, 'class_', 'N/A')} > {primary_species.family}. "
                        f"{common_names_text}"
                        f"{distribution_text}"
                        f"Data includes {total_synonyms} synonyms, {total_distributions} distribution records, "
                        f"and {total_sources} reference sources. Complete dataset compiled in the attached artifact."
                    )
                    if distribution_locations:
                        sample_locations = ', '.join(distribution_locations[:4])
                        if len(distribution_locations) > 4:
                            sample_locations += f" + {len(distribution_locations) - 4} more"
                        response_text += f"🌍 **FOUND IN**: {sample_locations}\n\n"
                    
                    response_text += "📁 **Complete taxonomic data, synonyms, distributions, and references have been compiled.**"
                    
                    if total_sources > 0:
                        response_text += f"**Sources:** {total_sources} references available\n\n"
                    
                    response_text += "The complete dataset has been compiled in the attached artifact."
                    
                    # Try multiple response approaches to ensure visibility
                    await context.reply(response_text)
                    
                    # Also send a follow-up with key data
                    summary = f"✅ **COMPLETED**: Found {primary_species.scientificname} in WoRMS!\n"
                    summary += f"🔍 **AphiaID**: {aphia_id}\n"
                    summary += f"📊 **Data Retrieved**: {total_vernaculars} names, {total_synonyms} synonyms, {total_distributions} locations, {total_sources} sources\n"
                    summary += f"🌊 **Classification**: {primary_species.kingdom} → {primary_species.phylum} → {getattr(primary_species, 'class_', 'N/A')} → {primary_species.family}\n"
                    if vernacular_names:
                        summary += f"🏷️ **Common Names**: {', '.join(vernacular_names[:3])}\n"
                    summary += f"📋 **Complete data available in JSON artifact above**"
                    
                    await context.reply(summary)
                    print(f"SUCCESS: Response sent for {primary_species.scientificname}")
                    
            except Exception as e:
                print(f"ERROR: Workflow failed: {str(e)[:100]}")
                await context.reply(f"An error occurred while processing the marine species request: {str(e)}")

    async def run_get_taxonomy(self, context: ResponseContext, params: MarineParameters):
        """Get basic taxonomic classification for a marine species"""
        search_term = params.species_name.strip() if params.species_name else None
        if not search_term:
            await context.reply("Please provide a marine species name.")
            return

        async with context.begin_process(f"Getting taxonomy for: {search_term}") as process:
            try:
                marine_query = await self.extract_query_info(search_term)
                search_name = marine_query.scientificname or marine_query.common_name
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    if not records:
                        await context.reply(f"No marine species found for '{search_name}' in WoRMS.")
                        return
                    
                    species = records[0]
                    await context.reply(
                        f"Taxonomic classification for {species.scientificname} (AphiaID: {species.AphiaID}): "
                        f"Kingdom: {species.kingdom}, Phylum: {species.phylum}, "
                        f"Class: {getattr(species, 'class_', 'N/A')}, Order: {species.order}, "
                        f"Family: {species.family}, Genus: {species.genus}. "
                        f"Authority: {species.authority or 'N/A'}."
                    )
            except Exception as e:
                await context.reply(f"Error retrieving taxonomy: {str(e)}")

    async def run_get_synonyms(self, context: ResponseContext, params: MarineParameters):
        """Get synonyms for a marine species"""
        search_term = params.species_name.strip() if params.species_name else None
        if not search_term:
            await context.reply("Please provide a marine species name.")
            return

        async with context.begin_process(f"Getting synonyms for: {search_term}") as process:
            try:
                marine_query = await self.extract_query_info(search_term)
                search_name = marine_query.scientificname or marine_query.common_name
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    if not records:
                        await context.reply(f"No marine species found for '{search_name}' in WoRMS.")
                        return
                    
                    species = records[0]
                    synonyms = await self.worms_client.get_synonyms_by_aphia_id(client, species.AphiaID)
                    
                    if synonyms:
                        synonym_list = [s.scientificname for s in synonyms]
                        await context.reply(
                            f"Found {len(synonyms)} synonyms for {species.scientificname}: "
                            f"{', '.join(synonym_list[:10])}"
                            f"{' and more...' if len(synonym_list) > 10 else ''}."
                        )
                    else:
                        await context.reply(f"No synonyms found for {species.scientificname} in WoRMS.")
            except Exception as e:
                await context.reply(f"Error retrieving synonyms: {str(e)}")

    async def run_get_distribution(self, context: ResponseContext, params: MarineParameters):
        """Get geographic distribution for a marine species"""
        search_term = params.species_name.strip() if params.species_name else None
        if not search_term:
            await context.reply("Please provide a marine species name.")
            return

        async with context.begin_process(f"Getting distribution for: {search_term}") as process:
            try:
                marine_query = await self.extract_query_info(search_term)
                search_name = marine_query.scientificname or marine_query.common_name
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    if not records:
                        await context.reply(f"No marine species found for '{search_name}' in WoRMS.")
                        return
                    
                    species = records[0]
                    distributions = await self.worms_client.get_distributions_by_aphia_id(client, species.AphiaID)
                    
                    if distributions:
                        locations = [d.locality for d in distributions if d.locality]
                        await context.reply(
                            f"{species.scientificname} is found in {len(distributions)} recorded locations: "
                            f"{', '.join(locations[:8])}"
                            f"{' and more...' if len(locations) > 8 else ''}."
                        )
                    else:
                        await context.reply(f"No distribution data found for {species.scientificname} in WoRMS.")
            except Exception as e:
                await context.reply(f"Error retrieving distribution: {str(e)}")

    async def run_get_vernacular_names(self, context: ResponseContext, params: MarineParameters):
        """Get common names for a marine species"""
        search_term = params.species_name.strip() if params.species_name else None
        if not search_term:
            await context.reply("Please provide a marine species name.")
            return

        async with context.begin_process(f"Getting common names for: {search_term}") as process:
            try:
                marine_query = await self.extract_query_info(search_term)
                search_name = marine_query.scientificname or marine_query.common_name
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    if not records:
                        await context.reply(f"No marine species found for '{search_name}' in WoRMS.")
                        return
                    
                    species = records[0]
                    vernaculars = await self.worms_client.get_vernaculars_by_aphia_id(client, species.AphiaID)
                    
                    if vernaculars:
                        names_by_lang = {}
                        for v in vernaculars:
                            lang = v.language or 'Unknown'
                            if lang not in names_by_lang:
                                names_by_lang[lang] = []
                            names_by_lang[lang].append(v.vernacular)
                        
                        response = f"Found {len(vernaculars)} common names for {species.scientificname}. "
                        for lang, names in list(names_by_lang.items())[:5]:
                            response += f"{lang}: {', '.join(names[:3])}. "
                        
                        await context.reply(response)
                    else:
                        await context.reply(f"No common names found for {species.scientificname} in WoRMS.")
            except Exception as e:
                await context.reply(f"Error retrieving common names: {str(e)}")

    async def run_get_sources(self, context: ResponseContext, params: MarineParameters):
        """Get scientific sources for a marine species"""
        search_term = params.species_name.strip() if params.species_name else None
        if not search_term:
            await context.reply("Please provide a marine species name.")
            return

        async with context.begin_process(f"Getting sources for: {search_term}") as process:
            try:
                marine_query = await self.extract_query_info(search_term)
                search_name = marine_query.scientificname or marine_query.common_name
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    if not records:
                        await context.reply(f"No marine species found for '{search_name}' in WoRMS.")
                        return
                    
                    species = records[0]
                    sources = await self.worms_client.get_sources_by_aphia_id(client, species.AphiaID)
                    
                    if sources:
                        source_refs = []
                        for s in sources[:5]:  # First 5 sources
                            ref = s.reference or f"{s.author} ({s.year})" if s.author and s.year else "Reference available"
                            source_refs.append(ref)
                        
                        await context.reply(
                            f"Found {len(sources)} scientific references for {species.scientificname}. "
                            f"Sample sources: {'; '.join(source_refs)}."
                        )
                    else:
                        await context.reply(f"No sources found for {species.scientificname} in WoRMS.")
            except Exception as e:
                await context.reply(f"Error retrieving sources: {str(e)}")

    async def extract_query_info(self, request: Optional[str]) -> MarineQueryModel:
        # Handle None or empty request
        if not request or not request.strip():
            return MarineQueryModel(scientificname=None, common_name=None)
        
        # If the request looks like a scientific name (two words), use it directly
        request_clean = request.strip()
        words = request_clean.split()
        if len(words) == 2 and words[0][0].isupper() and words[1][0].islower():
            return MarineQueryModel(scientificname=request_clean, common_name=None)
        
        # Use OpenAI to extract species information
        if not Config.GROQ_API_KEY:
            return MarineQueryModel(scientificname=request_clean, common_name=None)
            
        try:
            openai_client = AsyncOpenAI(
                api_key=Config.GROQ_API_KEY,
                base_url=Config.GROQ_BASE_URL,
            )
            instructor_client = instructor.from_openai(openai_client)
            
            marine_query = await instructor_client.chat.completions.create(
                model=Config.MODEL_NAME,
                response_model=MarineQueryModel,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract the scientific or common name of a marine species from the user question and return a JSON object with fields: "
                            "'scientificname' (e.g., 'Acropora cervicornis') and 'common_name' (e.g., 'staghorn coral'). "
                            "If no name is identified, return {'scientificname': null, 'common_name': null}."
                        )
                    },
                    {"role": "user", "content": request_clean}
                ],
                max_retries=3
            )
            return marine_query
        except Exception as e:
            print(f"WARN: LLM extraction failed, using direct: {str(e)[:50]}")
            # Fallback to using the request directly
            return MarineQueryModel(scientificname=request_clean, common_name=None)

    def _build_worms_uris(self, aphia_id: int) -> List[str]:
        return [
            f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaRecordsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaAttributesByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaSourcesByAphiaID/{aphia_id}"
        ]

# Debug: Print when module is imported
print("INIT: marine_agent.py module imported successfully")