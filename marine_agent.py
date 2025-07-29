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
            print(f"DEBUG: Fetching URL: {url}")
            resp = await client.get(url)
            print(f"DEBUG: Status Code: {resp.status_code}")
            if resp.status_code != 200:
                print(f"DEBUG: Non-200 status code: {resp.status_code}")
                return None
            data = resp.json()
            print(f"DEBUG: Successfully fetched data")
            return data
        except Exception as e:
            print(f"DEBUG: Error fetching JSON: {str(e)}")
        return None

    async def get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str) -> Optional[List[WoRMSRecord]]:
        encoded_name = quote(scientific_name)
        endpoint = f"/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
        data = await self.fetch_json(client, endpoint)
        print(f"DEBUG: Raw data for {scientific_name}: {type(data)}")
        if not data:
            print(f"DEBUG: No data returned for {scientific_name}")
            return None
        try:
            if isinstance(data, list):
                for record in data:
                    # Fix boolean field handling
                    for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                        if bool_field in record:
                            record[bool_field] = bool(record[bool_field]) if record[bool_field] is not None else None
                records = [WoRMSRecord(**record) for record in data]
                print(f"DEBUG: Validated {len(records)} records for {scientific_name}")
                return records
            # Handle single record
            for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                if bool_field in data:
                    data[bool_field] = bool(data[bool_field]) if data[bool_field] is not None else None
            record = WoRMSRecord(**data)
            print(f"DEBUG: Validated single record for {scientific_name}")
            return [record]
        except ValidationError as e:
            print(f"DEBUG: Validation error for {scientific_name}: {str(e)}")
            return None

    async def get_vernaculars_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSVernacular]]:
        endpoint = f"/AphiaVernacularsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            return [WoRMSVernacular(**record) for record in data] if isinstance(data, list) else [WoRMSVernacular(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for vernaculars: {str(e)}")
            return None

    async def get_synonyms_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSynonym]]:
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            return [WoRMSSynonym(**record) for record in data] if isinstance(data, list) else [WoRMSSynonym(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for synonyms: {str(e)}")
            return None

    async def get_distributions_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSDistribution]]:
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            return [WoRMSDistribution(**record) for record in data] if isinstance(data, list) else [WoRMSDistribution(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for distributions: {str(e)}")
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
            return [WoRMSSource(**record) for record in data] if isinstance(data, list) else [WoRMSSource(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for sources: {str(e)}")
            return None

class MarineAgent(IChatBioAgent):
    def __init__(self):
        print("DEBUG: MarineAgent initialized")
        self.worms_client = WoRMSClient()

    @override
    def get_agent_card(self) -> AgentCard:
        print("DEBUG: get_agent_card called")
        return AgentCard(
            name="Marine Species Data Agent",
            description="Retrieves marine species information from WoRMS, including taxonomic data, common names, synonyms, distributions, attributes, and sources.",
            url="http://18.222.189.40:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_info",
                    description="Returns marine species data with taxonomic information, common names, synonyms, distributions, attributes, and sources",
                    parameters=MarineParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: BaseModel):
        """Executes the requested agent entrypoint using the provided context."""
        
        # Debug logging matching your friend's pattern
        print(f"=== DEBUG INFO ===")
        print(f"request: {request}")
        print(f"entrypoint: {entrypoint}")
        print(f"params type: {type(params)}")
        print(f"params: {params}")
        print(f"==================")
        
        if entrypoint == "get_marine_info":
            await self.run_get_marine_info(context, params)
        else:
            # Handle unexpected entrypoints 
            await context.reply(f"Unknown entrypoint '{entrypoint}' received. Request was: '{request}'")
            raise ValueError(f"Unsupported entrypoint: {entrypoint}")

    async def run_get_marine_info(self, context: ResponseContext, params: MarineParameters):
        """Workflow for retrieving marine species information from WoRMS"""
        
        # Determine what we're searching for
        search_term = None
        
        if params and params.species_name and params.species_name.strip():
            search_term = params.species_name.strip()
            print(f"DEBUG: Using species_name from params: {search_term}")
        
        if not search_term:
            print("DEBUG: No search term available")
            await context.reply("Please provide a marine species name to search for.")
            return

        async with context.begin_process(f"Retrieving marine species data for: {search_term}") as process:
            try:
                await process.log("Extracted search parameters", data=params.model_dump(exclude_defaults=True))

                # Extract query information
                marine_query = await self.extract_query_info(search_term)
                await process.log("Identified marine species", data=marine_query.model_dump())
                
                if not marine_query.scientificname and not marine_query.common_name:
                    await context.reply(f"Could not identify a marine species name in: '{search_term}'")
                    return
                    
                search_name = marine_query.scientificname or marine_query.common_name
                await process.log(f"Searching WoRMS for: {search_name}")
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Search for species records
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    search_url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{quote(search_name)}?like=false&marine_only=true"
                    await process.log("WoRMS search completed", data={"search_url": search_url})
                    
                    if not records or len(records) == 0:
                        await context.reply(f"No marine species found matching '{search_name}' in WoRMS database.")
                        return
                        
                    primary_species = records[0]
                    aphia_id = primary_species.AphiaID
                    await process.log(f"Found species: {primary_species.scientificname} (AphiaID: {aphia_id})")
                    
                    # Fetch additional data based on parameters
                    await process.log("Retrieving comprehensive marine species data...")
                    
                    # Use executor pattern like your friend for potentially blocking operations
                    loop = asyncio.get_event_loop()
                    
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
                        await process.log("Error processing species data", data={"error": str(e)})
                        await context.reply(f"Error processing marine species data: {str(e)}")
                        return

                    # Count collected data for summary
                    total_vernaculars = len(vernaculars) if vernaculars else 0
                    total_synonyms = len(synonyms) if synonyms else 0
                    total_distributions = len(distributions) if distributions else 0
                    total_sources = len(sources) if sources else 0

                    await process.log(f"Successfully compiled data for {primary_species.scientificname}")

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
                    
                    # Prepare user-friendly summary following your friend's pattern
                    vernacular_names = [v.vernacular for v in vernaculars] if vernaculars else []
                    distribution_locations = [d.locality for d in distributions if d.locality] if distributions else []
                    
                    response_text = (
                        f"Retrieved comprehensive data for **{primary_species.scientificname}** (AphiaID: {aphia_id}) from WoRMS.\n\n"
                        f"**Taxonomic Classification:**\n"
                        f"- Kingdom: {primary_species.kingdom or 'N/A'}\n"
                        f"- Phylum: {primary_species.phylum or 'N/A'}\n"
                        f"- Class: {getattr(primary_species, 'class_', None) or 'N/A'}\n"
                        f"- Order: {primary_species.order or 'N/A'}\n"
                        f"- Family: {primary_species.family or 'N/A'}\n"
                        f"- Genus: {primary_species.genus or 'N/A'}\n\n"
                    )
                    
                    if vernacular_names:
                        sample_names = ', '.join(vernacular_names[:3])
                        if len(vernacular_names) > 3:
                            sample_names += f" and {len(vernacular_names) - 3} more"
                        response_text += f"**Common Names:** {sample_names}\n\n"
                    
                    if total_synonyms > 0:
                        response_text += f"**Synonyms:** {total_synonyms} found\n\n"
                    
                    if distribution_locations:
                        sample_locations = ', '.join(distribution_locations[:3])
                        if len(distribution_locations) > 3:
                            sample_locations += f" and {len(distribution_locations) - 3} more"
                        response_text += f"**Distribution:** {sample_locations}\n\n"
                    
                    if total_sources > 0:
                        response_text += f"**Sources:** {total_sources} references available\n\n"
                    
                    response_text += "The complete dataset has been compiled in the attached artifact."
                    
                    await context.reply(response_text)
                    
            except Exception as e:
                print(f"DEBUG: General error in run_get_marine_info: {str(e)}")
                import traceback
                print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                await process.log("Error during marine species workflow", data={"error": str(e)})
                await context.reply(f"An error occurred while processing the marine species request: {str(e)}")

    async def extract_query_info(self, request: Optional[str]) -> MarineQueryModel:
        print(f"DEBUG: extract_query_info called with request: '{request}'")
        
        # Handle None or empty request
        if not request or not request.strip():
            return MarineQueryModel(scientificname=None, common_name=None)
        
        # If the request looks like a scientific name (two words), use it directly
        request_clean = request.strip()
        words = request_clean.split()
        if len(words) == 2 and words[0][0].isupper() and words[1][0].islower():
            print(f"DEBUG: Direct scientific name detected: {request_clean}")
            return MarineQueryModel(scientificname=request_clean, common_name=None)
        
        # Use OpenAI to extract species information
        if not Config.GROQ_API_KEY:
            print("DEBUG: No GROQ API key, using direct name")
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
            print(f"DEBUG: Extracted query: {marine_query.model_dump()}")
            return marine_query
        except Exception as e:
            print(f"DEBUG: Query extraction error: {str(e)}")
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
print("DEBUG: marine_agent.py module imported successfully")