import os
import json
import dotenv
import instructor
from typing_extensions import override
from openai import AsyncOpenAI
from pydantic import ValidationError, BaseModel
from typing import Optional, List
import httpx
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
    MarineQueryModel,
    MarineParameters
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
            print(f"DEBUG: Calling WoRMS API: {url}")
            resp = await client.get(url)
            if resp.status_code != 200:
                print(f"ERROR: WoRMS API {resp.status_code} for {endpoint}")
                return None
            data = resp.json()
            print(f"SUCCESS: Got data from WoRMS")
            return data
        except Exception as e:
            print(f"ERROR: WoRMS fetch failed: {str(e)}")
            return None

    async def get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str):
        encoded_name = quote(scientific_name)
        endpoint = f"/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                records = []
                for record in data:
                    for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                        if bool_field in record and record[bool_field] is not None:
                            record[bool_field] = bool(record[bool_field])
                    records.append(WoRMSRecord(**record))
                print(f"SUCCESS: Found {len(records)} records")
                return records
            else:
                for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                    if bool_field in data and data[bool_field] is not None:
                        data[bool_field] = bool(data[bool_field])
                record = WoRMSRecord(**data)
                print(f"SUCCESS: Found 1 record")
                return [record]
        except ValidationError as e:
            print(f"ERROR: Validation failed: {str(e)}")
            return None

    async def get_synonyms_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSSynonym(**record) for record in data] if isinstance(data, list) else [WoRMSSynonym(**data)]
            print(f"SUCCESS: Found {len(result)} synonyms")
            return result
        except ValidationError as e:
            print(f"ERROR: Synonym validation failed: {str(e)}")
            return None

    async def get_vernaculars_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaVernacularsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSVernacular(**record) for record in data] if isinstance(data, list) else [WoRMSVernacular(**data)]
            print(f"SUCCESS: Found {len(result)} vernacular names")
            return result
        except ValidationError as e:
            print(f"ERROR: Vernacular validation failed: {str(e)}")
            return None

    async def get_distributions_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            result = [WoRMSDistribution(**record) for record in data] if isinstance(data, list) else [WoRMSDistribution(**data)]
            print(f"SUCCESS: Found {len(result)} distributions")
            return result
        except ValidationError as e:
            print(f"ERROR: Distribution validation failed: {str(e)}")
            return None

class MarineAgent(IChatBioAgent):
    def __init__(self):
        print("INIT: MarineAgent initialized")
        self.worms_client = WoRMSClient()

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Marine Species Data Agent",
            description="Retrieves marine species information from WoRMS database",
            url="http://18.222.189.40:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_data",
                    description="Get marine species data from WoRMS",
                    parameters=MarineParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: MarineParameters):
        print(f"DEBUG: Entrypoint '{entrypoint}' called with request: {request}")
        print(f"DEBUG: Params: {params}")
        
        if entrypoint == "get_marine_data":
            await self.get_marine_data(context, params, request)
        else:
            print(f"ERROR: Unknown entrypoint: {entrypoint}")
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def get_marine_data(self, context: ResponseContext, params: MarineParameters, request: str):
        """Main entrypoint - gets marine species data based on what user wants"""
        print(f"DEBUG: get_marine_data called")
        
        # Extract scientific name from request
        scientific_name = await self.extract_scientific_name(request, params)
        if not scientific_name:
            await context.reply("Could not identify a marine species name from your request.")
            return
        
        print(f"DEBUG: Searching for species: {scientific_name}")
        
        async with context.begin_process(f"Getting marine data for {scientific_name}") as process:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Get species data from WoRMS
                    records = await self.worms_client.get_species_by_name(client, scientific_name)
                    
                    if not records:
                        await context.reply(f"No marine species found for '{scientific_name}' in WoRMS database.")
                        return
                    
                    # Use first record
                    species = records[0]
                    aphia_id = species.AphiaID
                    print(f"DEBUG: Found species {species.scientificname} with AphiaID {aphia_id}")
                    
                    # Determine what data to get based on request or params
                    get_synonyms = "synonym" in request.lower() or params.include_synonyms
                    get_vernacular = "vernacular" in request.lower() or "common" in request.lower() or params.include_vernaculars
                    get_distribution = "distribution" in request.lower() or "location" in request.lower() or params.include_distribution
                    
                    # Get additional data if requested
                    synonyms = None
                    vernaculars = None
                    distributions = None
                    
                    if get_synonyms:
                        synonyms = await self.worms_client.get_synonyms_by_aphia_id(client, aphia_id)
                    if get_vernacular:
                        vernaculars = await self.worms_client.get_vernaculars_by_aphia_id(client, aphia_id)
                    if get_distribution:
                        distributions = await self.worms_client.get_distributions_by_aphia_id(client, aphia_id)
                    
                    # Create data structure
                    species_data = {
                        "species": species.model_dump(),
                        "aphia_id": aphia_id,
                        "scientific_name": species.scientificname,
                        "search_term": scientific_name,
                        "synonyms": [s.model_dump() for s in synonyms] if synonyms else None,
                        "vernacular_names": [v.model_dump() for v in vernaculars] if vernaculars else None,
                        "distributions": [d.model_dump() for d in distributions] if distributions else None
                    }
                    
                    # Create artifact
                    content = json.dumps(species_data, indent=2)
                    content_bytes = content.encode('utf-8')
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species data for {species.scientificname}",
                        content=content_bytes,
                        uris=[f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"],
                        metadata={
                            "data_source": "WoRMS",
                            "aphia_id": aphia_id,
                            "scientific_name": species.scientificname,
                            "retrieved_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    # Send response
                    response = f"Found {species.scientificname} (AphiaID: {aphia_id}) in WoRMS database. "
                    response += f"Classification: {species.kingdom} > {species.phylum} > {getattr(species, 'class_', 'N/A')} > {species.family}. "
                    
                    if synonyms:
                        response += f"Found {len(synonyms)} synonyms. "
                    if vernaculars:
                        response += f"Found {len(vernaculars)} common names. "
                    if distributions:
                        response += f"Found {len(distributions)} distribution records. "
                    
                    response += "Complete data has been saved to artifact."
                    
                    await context.reply(response)
                    print(f"SUCCESS: Completed request for {species.scientificname}")
                    
            except Exception as e:
                print(f"ERROR: Failed to get marine data: {str(e)}")
                await context.reply(f"Error retrieving marine data: {str(e)}")

    async def extract_scientific_name(self, request: str, params: MarineParameters) -> Optional[str]:
        """Extract scientific name using instructor"""
        print(f"DEBUG: Extracting scientific name from: {request}")
        
        # Check if params has species_name
        if params and params.species_name and params.species_name.strip():
            print(f"DEBUG: Using species_name from params: {params.species_name}")
            return params.species_name.strip()
        
        # Check if request looks like scientific name (Genus species)
        if request and request.strip():
            words = request.strip().split()
            if len(words) == 2 and words[0][0].isupper() and words[1][0].islower():
                print(f"DEBUG: Request looks like scientific name: {request.strip()}")
                return request.strip()
        
        # Use instructor to extract
        if not Config.GROQ_API_KEY:
            print(f"DEBUG: No API key, using request as-is: {request}")
            return request.strip() if request else None
        
        try:
            print(f"DEBUG: Using instructor to extract scientific name")
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
                            "Extract the scientific name or common name of a marine species from the user input. "
                            "Return JSON with 'scientificname' and 'common_name' fields. "
                            "If you find a scientific name (like 'Acropora cervicornis'), put it in scientificname. "
                            "If you find a common name (like 'staghorn coral'), put it in common_name."
                        )
                    },
                    {"role": "user", "content": request}
                ],
                max_retries=2
            )
            
            result = marine_query.scientificname or marine_query.common_name
            print(f"DEBUG: Instructor extracted: {result}")
            return result
            
        except Exception as e:
            print(f"WARN: Instructor failed, using request directly: {str(e)}")
            return request.strip() if request else None

print("INIT: Marine agent with single get_marine_data entrypoint loaded")