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
            description="Retrieves marine species synonyms, vernacular names, and distribution from WoRMS",
            url="http://18.222.189.40:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Get synonyms for marine species",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_vernacular",
                    description="Get vernacular/common names for marine species",
                    parameters=MarineParameters
                ),
                AgentEntrypoint(
                    id="get_distribution",
                    description="Get distribution data for marine species",
                    parameters=MarineParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: BaseModel):
        print(f"=== RECEIVED REQUEST ===")
        print(f"entrypoint: '{entrypoint}'")
        print(f"entrypoint repr: {repr(entrypoint)}")
        print(f"entrypoint bytes: {entrypoint.encode()}")
        print(f"request: '{request}'")
        print(f"params: {params}")
        print(f"========================")
        
        # Convert params to MarineParameters
        if isinstance(params, dict):
            marine_params = MarineParameters(**params)
        elif hasattr(params, 'model_dump'):
            marine_params = MarineParameters(**params.model_dump())
        else:
            marine_params = params
        
        # Route to methods
        if entrypoint == "get_synonyms":
            await self.get_synonyms(context, marine_params, request)
        elif entrypoint == "get_vernacular":
            await self.get_vernacular(context, marine_params, request)
        elif entrypoint == "get_distribution":
            await self.get_distribution(context, marine_params, request)
        else:
            print(f"ERROR: Unknown entrypoint: {entrypoint}")
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def get_synonyms(self, context: ResponseContext, params: MarineParameters, request: str):
        """Get synonyms for marine species"""
        print(f"DEBUG: get_synonyms called")
        
        scientific_name = await self.extract_scientific_name(request, params)
        if not scientific_name:
            await context.reply("Could not identify a marine species name.")
            return
        
        async with context.begin_process(f"Getting synonyms for {scientific_name}") as process:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # First get species record
                    records = await self.worms_client.get_species_by_name(client, scientific_name)
                    if not records:
                        await context.reply(f"No marine species found for '{scientific_name}'")
                        return
                    
                    species = records[0]
                    aphia_id = species.AphiaID
                    print(f"DEBUG: Found species {species.scientificname}, getting synonyms")
                    
                    # Get synonyms
                    synonyms = await self.worms_client.get_synonyms_by_aphia_id(client, aphia_id)
                    if not synonyms:
                        await context.reply(f"No synonyms found for {species.scientificname}")
                        return
                    
                    # Create artifact with synonyms data
                    synonyms_data = {
                        "species_name": species.scientificname,
                        "aphia_id": aphia_id,
                        "synonyms": [synonym.model_dump() for synonym in synonyms],
                        "count": len(synonyms)
                    }
                    
                    content = json.dumps(synonyms_data, indent=2)
                    content_bytes = content.encode('utf-8')
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Synonyms for {species.scientificname}",
                        content=content_bytes,
                        uris=[f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{aphia_id}"],
                        metadata={
                            "data_source": "WoRMS",
                            "data_type": "synonyms",
                            "aphia_id": aphia_id,
                            "count": len(synonyms)
                        }
                    )
                    
                    # Send response
                    synonym_names = [s.scientificname for s in synonyms[:5]]
                    response = f"Found {len(synonyms)} synonyms for {species.scientificname}: {', '.join(synonym_names)}"
                    if len(synonyms) > 5:
                        response += f" and {len(synonyms) - 5} more. Complete data in artifact."
                    
                    await context.reply(response)
                    print(f"SUCCESS: Sent {len(synonyms)} synonyms")
                    
            except Exception as e:
                print(f"ERROR: Failed to get synonyms: {str(e)}")
                await context.reply(f"Error getting synonyms: {str(e)}")

    async def get_vernacular(self, context: ResponseContext, params: MarineParameters, request: str):
        """Get vernacular names for marine species"""
        print(f"DEBUG: get_vernacular called")
        
        scientific_name = await self.extract_scientific_name(request, params)
        if not scientific_name:
            await context.reply("Could not identify a marine species name.")
            return
        
        async with context.begin_process(f"Getting vernacular names for {scientific_name}") as process:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # First get species record
                    records = await self.worms_client.get_species_by_name(client, scientific_name)
                    if not records:
                        await context.reply(f"No marine species found for '{scientific_name}'")
                        return
                    
                    species = records[0]
                    aphia_id = species.AphiaID
                    print(f"DEBUG: Found species {species.scientificname}, getting vernacular names")
                    
                    # Get vernacular names
                    vernaculars = await self.worms_client.get_vernaculars_by_aphia_id(client, aphia_id)
                    if not vernaculars:
                        await context.reply(f"No vernacular names found for {species.scientificname}")
                        return
                    
                    # Create artifact with vernacular data
                    vernacular_data = {
                        "species_name": species.scientificname,
                        "aphia_id": aphia_id,
                        "vernacular_names": [vernacular.model_dump() for vernacular in vernaculars],
                        "count": len(vernaculars)
                    }
                    
                    content = json.dumps(vernacular_data, indent=2)
                    content_bytes = content.encode('utf-8')
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Vernacular names for {species.scientificname}",
                        content=content_bytes,
                        uris=[f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aphia_id}"],
                        metadata={
                            "data_source": "WoRMS",
                            "data_type": "vernacular_names",
                            "aphia_id": aphia_id,
                            "count": len(vernaculars)
                        }
                    )
                    
                    # Send response
                    names = [v.vernacular for v in vernaculars[:5]]
                    response = f"Found {len(vernaculars)} vernacular names for {species.scientificname}: {', '.join(names)}"
                    if len(vernaculars) > 5:
                        response += f" and {len(vernaculars) - 5} more. Complete data in artifact."
                    
                    await context.reply(response)
                    print(f"SUCCESS: Sent {len(vernaculars)} vernacular names")
                    
            except Exception as e:
                print(f"ERROR: Failed to get vernacular names: {str(e)}")
                await context.reply(f"Error getting vernacular names: {str(e)}")

    async def get_distribution(self, context: ResponseContext, params: MarineParameters, request: str):
        """Get distribution data for marine species"""
        print(f"DEBUG: get_distribution called")
        
        scientific_name = await self.extract_scientific_name(request, params)
        if not scientific_name:
            await context.reply("Could not identify a marine species name.")
            return
        
        async with context.begin_process(f"Getting distribution for {scientific_name}") as process:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    # First get species record
                    records = await self.worms_client.get_species_by_name(client, scientific_name)
                    if not records:
                        await context.reply(f"No marine species found for '{scientific_name}'")
                        return
                    
                    species = records[0]
                    aphia_id = species.AphiaID
                    print(f"DEBUG: Found species {species.scientificname}, getting distribution")
                    
                    # Get distribution data
                    distributions = await self.worms_client.get_distributions_by_aphia_id(client, aphia_id)
                    if not distributions:
                        await context.reply(f"No distribution data found for {species.scientificname}")
                        return
                    
                    # Create artifact with distribution data
                    distribution_data = {
                        "species_name": species.scientificname,
                        "aphia_id": aphia_id,
                        "distributions": [dist.model_dump() for dist in distributions],
                        "count": len(distributions)
                    }
                    
                    content = json.dumps(distribution_data, indent=2)
                    content_bytes = content.encode('utf-8')
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Distribution data for {species.scientificname}",
                        content=content_bytes,
                        uris=[f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{aphia_id}"],
                        metadata={
                            "data_source": "WoRMS",
                            "data_type": "distribution",
                            "aphia_id": aphia_id,
                            "count": len(distributions)
                        }
                    )
                    
                    # Send response
                    locations = [d.locality for d in distributions if d.locality][:5]
                    response = f"Found {len(distributions)} distribution records for {species.scientificname}"
                    if locations:
                        response += f": {', '.join(locations)}"
                        if len(distributions) > 5:
                            response += f" and {len(distributions) - 5} more locations"
                    response += ". Complete data in artifact."
                    
                    await context.reply(response)
                    print(f"SUCCESS: Sent {len(distributions)} distribution records")
                    
            except Exception as e:
                print(f"ERROR: Failed to get distribution: {str(e)}")
                await context.reply(f"Error getting distribution: {str(e)}")

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

print("INIT: Marine agent with synonyms, vernacular, and distribution entrypoints loaded")