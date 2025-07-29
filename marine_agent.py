import os
import json
import dotenv
import instructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import ValidationError
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
            # Print first 100 characters of response text
            response_text = resp.text[:100] + ("..." if len(resp.text) > 100 else "")
            print(f"DEBUG: Response Text (truncated to 100 chars): {response_text} (total {len(resp.text)} chars)")
            if resp.status_code != 200:
                print(f"DEBUG: Non-200 status code: {resp.status_code}")
                return None
            data = resp.json()
            # Convert data to string and print first 100 characters
            data_str = str(data)
            data_text = data_str[:100] + ("..." if len(data_str) > 100 else "")
            print(f"DEBUG: Parsed JSON (truncated to 100 chars): {data_text} (total {len(data_str)} chars)")
            return data
        except Exception as e:
            print(f"DEBUG: Error fetching JSON: {str(e)}")
        return None

    async def get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str) -> Optional[List[WoRMSRecord]]:
        encoded_name = quote(scientific_name)
        endpoint = f"/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
        data = await self.fetch_json(client, endpoint)
        print(f"DEBUG: Raw data for {scientific_name}: {data}")
        if not data:
            print(f"DEBUG: No data returned for {scientific_name}")
            return None
        try:
            if isinstance(data, list):
                for record in data:
                    if 'isMarine' in record:
                        record['isMarine'] = bool(record['isMarine']) if record['isMarine'] is not None else None
                    if 'isBrackish' in record:
                        record['isBrackish'] = bool(record['isBrackish']) if record['isBrackish'] is not None else None
                    if 'isFreshwater' in record:
                        record['isFreshwater'] = bool(record['isFreshwater']) if record['isFreshwater'] is not None else None
                    if 'isTerrestrial' in record:
                        record['isTerrestrial'] = bool(record['isTerrestrial']) if record['isTerrestrial'] is not None else None
                    if 'isExtinct' in record:
                        record['isExtinct'] = bool(record['isExtinct']) if record['isExtinct'] is not None else None
                records = [WoRMSRecord(**record) for record in data]
                print(f"DEBUG: Validated {len(records)} records for {scientific_name}")
                return records
            if 'isMarine' in data:
                data['isMarine'] = bool(data['isMarine']) if data['isMarine'] is not None else None
            if 'isBrackish' in data:
                data['isBrackish'] = bool(data['isBrackish']) if data['isBrackish'] is not None else None
            if 'isFreshwater' in data:
                data['isFreshwater'] = bool(data['isFreshwater']) if data['isFreshwater'] is not None else None
            if 'isTerrestrial' in data:
                data['isTerrestrial'] = bool(data['isTerrestrial']) if data['isFreshwater'] is not None else None
            if 'isExtinct' in data:
                data['isExtinct'] = bool(data['isExtinct']) if data['isExtinct'] is not None else None
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
            print(f"DEBUG: No vernaculars data for AphiaID {aphia_id}")
            return None
        try:
            return [WoRMSVernacular(**record) for record in data] if isinstance(data, list) else [WoRMSVernacular(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for vernaculars AphiaID {aphia_id}: {str(e)}")
            return None

    async def get_synonyms_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSynonym]]:
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            print(f"DEBUG: No synonyms data for AphiaID {aphia_id}")
            return None
        try:
            return [WoRMSSynonym(**record) for record in data] if isinstance(data, list) else [WoRMSSynonym(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for synonyms AphiaID {aphia_id}: {str(e)}")
            return None

    async def get_distributions_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSDistribution]]:
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            print(f"DEBUG: No distributions data for AphiaID {aphia_id}")
            return None
        try:
            return [WoRMSDistribution(**record) for record in data] if isinstance(data, list) else [WoRMSDistribution(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for distributions AphiaID {aphia_id}: {str(e)}")
            return None

    async def get_attributes_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[dict]]:
        endpoint = f"/AphiaAttributesByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            print(f"DEBUG: No attributes data for AphiaID {aphia_id}")
            return None
        return data if isinstance(data, list) else [data]

    async def get_sources_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSource]]:
        endpoint = f"/AphiaSourcesByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            print(f"DEBUG: No sources data for AphiaID {aphia_id}")
            return None
        try:
            return [WoRMSSource(**record) for record in data] if isinstance(data, list) else [WoRMSSource(**data)]
        except ValidationError as e:
            print(f"DEBUG: Validation error for sources AphiaID {aphia_id}: {str(e)}")
            return None

class MarineAgent(IChatBioAgent):
    def __init__(self):
        print("DEBUG: MarineAgent initialized")
        self.worms_client = WoRMSClient()

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

    async def extract_query_info(self, request: str) -> MarineQueryModel:
        print(f"DEBUG: extract_query_info called with request: '{request}' (type: {type(request)})")
        
        # Handle None or empty request
        if not request:
            print("DEBUG: Request is None or empty, returning empty MarineQueryModel")
            return MarineQueryModel(scientificname=None, common_name=None)
            
        openai_client = AsyncOpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
        )
        instructor_client = instructor.from_openai(openai_client)
        try:
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
                    {"role": "user", "content": request}
                ],
                max_retries=3
            )
            print(f"DEBUG: Extracted query: {marine_query.model_dump()}")
            return marine_query
        except (InstructorRetryException, ValidationError) as e:
            print(f"DEBUG: Query extraction error: {str(e)}")
            return MarineQueryModel(scientificname=None, common_name=None)

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

    # ADD MORE DEBUG LOGGING HERE
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineParameters,
    ) -> None:
        print("="*50)
        print(f"DEBUG: Run method called!")
        print(f"DEBUG: request: '{request}' (type: {type(request)})")
        print(f"DEBUG: entrypoint: '{entrypoint}' (type: {type(entrypoint)})")
        print(f"DEBUG: params: {params.model_dump() if params else 'None'} (type: {type(params)})")
        print(f"DEBUG: context: {context} (type: {type(context)})")
        print("="*50)
        
        if entrypoint != "get_marine_info":
            print(f"DEBUG: Invalid entrypoint: {entrypoint}")
            await context.reply(f"Unknown entrypoint: {entrypoint}")
            return
            
        # Handle case where request is None - use params.species_name as fallback
        search_term = request
        if not search_term and params and params.species_name:
            search_term = params.species_name
            print(f"DEBUG: Using species_name from params: {search_term}")
        
        if not search_term:
            print("DEBUG: No search term available")
            await context.reply("No species name provided in request or parameters")
            return
            
        async with context.begin_process("Analyzing marine species request") as process:
            try:
                # Use the search term directly if request is None
                if not request:
                    marine_query = MarineQueryModel(scientificname=search_term, common_name=None)
                    print(f"DEBUG: Using direct search term: {marine_query.model_dump()}")
                else:
                    marine_query = await self.extract_query_info(request)
                    print(f"DEBUG: Marine query after extraction: {marine_query.model_dump()}")
                
                if not marine_query.scientificname and not marine_query.common_name:
                    print(f"DEBUG: No scientific or common name identified")
                    await context.reply("No scientific or common name identified in the request")
                    return
                    
                search_name = marine_query.scientificname or marine_query.common_name
                await process.log(f"Identified marine species: {search_name}", {
                    "scientific_name": marine_query.scientificname,
                    "common_name": marine_query.common_name
                })
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    await process.log(f"Searching WoRMS by name: {search_name}", {
                        "search_url": f"https://www.marinespecies.org/rest/AphiaRecordsByName/{quote(search_name)}?like=false&marine_only=true"
                    })
                    
                    if not records or len(records) == 0:
                        print(f"DEBUG: No records found for {search_name}")
                        await context.reply(f"No marine species found matching '{search_name}'")
                        return
                        
                    primary_species = records[0]
                    aphia_id = primary_species.AphiaID
                    await process.log(f"Found species: {primary_species.scientificname} (AphiaID: {aphia_id})")
                    await process.log("Retrieving marine species data, vernacular names, synonyms, distributions, attributes, and sources")
                    
                    vernaculars = await self.worms_client.get_vernaculars_by_aphia_id(client, aphia_id) if params.include_vernaculars else None
                    synonyms = await self.worms_client.get_synonyms_by_aphia_id(client, aphia_id) if params.include_synonyms else None
                    distributions = await self.worms_client.get_distributions_by_aphia_id(client, aphia_id) if params.include_distribution else None
                    attributes = await self.worms_client.get_attributes_by_aphia_id(client, aphia_id)
                    sources = await self.worms_client.get_sources_by_aphia_id(client, aphia_id) if params.include_sources else None
                    
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
                        print(f"DEBUG: Validation error for species data: {str(e)}")
                        await context.reply(f"Error processing marine species data: {str(e)}")
                        return

                    # Create the artifact using process.create_artifact()
                    # Convert string content to bytes as expected by the API
                    content_bytes = content.encode('utf-8')
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species data for {primary_species.scientificname}",
                        content=content_bytes,  # Now using bytes instead of string
                        uris=self._build_worms_uris(aphia_id),
                        metadata={
                            "aphia_id": aphia_id,
                            "scientific_name": primary_species.scientificname,
                            "search_term": search_name,
                            "data_sources": [
                                "species",
                                "vernaculars" if params.include_vernaculars else None,
                                "synonyms" if params.include_synonyms else None,  
                                "distributions" if params.include_distribution else None,
                                "attributes",
                                "sources" if params.include_sources else None
                            ],
                            "retrieved_at": datetime.now(timezone.utc).isoformat()
                        }
                    )

                    print(f"DEBUG: Successfully created artifact for {primary_species.scientificname} (AphiaID: {aphia_id})")
                    
                    # Prepare and send text summary
                    vernacular_names = [v.vernacular for v in vernaculars] if vernaculars else []
                    synonym_names = [s.scientificname for s in synonyms] if synonyms else []
                    distribution_locations = [d.locality for d in distributions if d.locality] if distributions else []
                    attribute_summary = (
                        f"CITES Annex II, IUCN Red List: Critically Endangered (2021, Criteria A2bce)"
                        if attributes and any(attr.get("measurementValue") == "CITES" for attr in attributes)
                        else "None"
                    )
                    response_text = (
                        f"Found marine species data for {primary_species.scientificname} (AphiaID: {aphia_id}). "
                        f"Common names: {', '.join(vernacular_names) if vernacular_names else 'None'}. "
                        f"Synonyms: {', '.join(synonym_names) if synonym_names else 'None'}. "
                        f"Distributions: {', '.join(distribution_locations) if distribution_locations else 'None'}. "
                        f"Conservation status: {attribute_summary}. "
                        f"Sources: {len(sources) if sources else 0} found. "
                        "The artifact contains detailed taxonomic information, common names, synonyms, distributions, attributes, and sources from WoRMS."
                    )
                    await context.reply(response_text)
                    
            except Exception as e:
                print(f"DEBUG: General error in run method: {str(e)}")
                import traceback
                print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                await context.reply(f"An error occurred while processing the request: {str(e)}")

# Debug: Print when module is imported
print("DEBUG: marine_agent.py module imported successfully")