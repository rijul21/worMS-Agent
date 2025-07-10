import os
import json
import dotenv
import instructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from typing import Optional, List, Dict, Any, override
import httpx
from datetime import datetime

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

# Import our data models
from worms_models import (
    WoRMSRecord, WoRMSSynonym, WoRMSDistribution, WoRMSVernacular,
    WoRMSClassification, WoRMSChild, CompleteMarineSpeciesData,
    MarineQueryModel, MarineParameters
)

dotenv.load_dotenv()

# Configs for LLM and woRMS
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    WORMS_BASE_URL = "https://www.marinespecies.org/rest"
    MODEL_NAME = "llama3-70b-8192"

# woRMS API Client with proper data models
class WoRMSClient:
    def __init__(self, base_url: str = Config.WORMS_BASE_URL):
        self.base_url = base_url

    async def fetch_json(self, client: httpx.AsyncClient, endpoint: str):
        try:
            url = f"{self.base_url}{endpoint}"
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    async def get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str) -> Optional[List[WoRMSRecord]]:
        endpoint = f"/AphiaRecordsByName/{scientific_name}?like=false&marine_only=true"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSRecord(**record) for record in data]
            else:
                return [WoRMSRecord(**data)]
        except ValidationError as e:
            print(f"Validation error for species data: {e}")
            return None

    async def get_species_by_common_name(self, client: httpx.AsyncClient, common_name: str) -> Optional[List[WoRMSRecord]]:
        endpoint = f"/AphiaRecordsByVernacular/{common_name}?like=true&offset=1"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSRecord(**record) for record in data]
            else:
                return [WoRMSRecord(**data)]
        except ValidationError as e:
            print(f"Validation error for vernacular search: {e}")
            return None

    async def get_synonyms(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSynonym]]:
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSSynonym(**record) for record in data]
            else:
                return [WoRMSSynonym(**data)]
        except ValidationError as e:
            print(f"Validation error for synonyms: {e}")
            return None

    async def get_distribution(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSDistribution]]:
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSDistribution(**record) for record in data]
            else:
                return [WoRMSDistribution(**data)]
        except ValidationError as e:
            print(f"Validation error for distribution: {e}")
            return None

    async def get_vernaculars(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSVernacular]]:
        endpoint = f"/AphiaVernacularsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSVernacular(**record) for record in data]
            else:
                return [WoRMSVernacular(**data)]
        except ValidationError as e:
            print(f"Validation error for vernaculars: {e}")
            return None

    async def get_classification(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSClassification]]:
        endpoint = f"/AphiaClassificationByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSClassification(**record) for record in data]
            else:
                return [WoRMSClassification(**data)]
        except ValidationError as e:
            print(f"Validation error for classification: {e}")
            return None

    async def get_children(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSChild]]:
        endpoint = f"/AphiaChildrenByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        
        try:
            if isinstance(data, list):
                return [WoRMSChild(**record) for record in data]
            else:
                return [WoRMSChild(**data)]
        except ValidationError as e:
            print(f"Validation error for children: {e}")
            return None

# Main Agent class
class MarineAgent(IChatBioAgent):
    def __init__(self):
        self.worms_client = WoRMSClient()

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Marine Species Data Agent",
            description="Retrieves detailed marine species information from WoRMS using scientific or common names.",
            url="http://localhost:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_info",
                    description="Returns detailed marine species data with comprehensive taxonomic information",
                    parameters=MarineParameters
                )
            ]
        )

    async def extract_query_info(self, request: str) -> MarineQueryModel:
        openai_client = AsyncOpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
        )
        instructor_client = instructor.patch(openai_client)

        marine_query = await instructor_client.chat.completions.create(
            model=Config.MODEL_NAME,
            response_model=MarineQueryModel,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract marine animal names from user questions. Look for both scientific names "
                        "(like 'Orcinus orca') and common names (like 'killer whale', 'great white shark'). "
                        "If you find either, extract it. Handle conversational queries naturally."
                    )
                },
                {"role": "user", "content": request}
            ],
            max_retries=3
        )
        return marine_query

    def _build_worms_uris(self, aphia_id: int) -> List[str]:
        """Build URIs for WoRMS data sources"""
        return [
            f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaRecordsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{aphia_id}",
            f"https://www.marinespecies.org/rest/AphiaChildrenByAphiaID/{aphia_id}"
        ]

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineParameters,
    ) -> None:
        if entrypoint != "get_marine_info":
            raise ValueError(f"Unknown entrypoint: {entrypoint}")

        async with context.begin_process("Analyzing marine species request") as process:
            try:
                # Extract marine species information from user request
                marine_query = await self.extract_query_info(request)

                if not marine_query.scientificname and not marine_query.common_name:
                    await context.reply("No marine species identified in the request")
                    return

                search_name = marine_query.scientificname or marine_query.common_name
                await process.log(f"Identified marine species: {search_name}", {
                    "scientific_name": marine_query.scientificname,
                    "common_name": marine_query.common_name
                })

                # Search for the species
                async with httpx.AsyncClient() as client:
                    records = None
                    
                    # Try scientific name first
                    if marine_query.scientificname:
                        records = await self.worms_client.get_species_by_name(client, marine_query.scientificname)
                        await process.log(f"Searching WoRMS by scientific name: {marine_query.scientificname}")
                    
                    # Try common name if scientific name failed
                    if not records and marine_query.common_name:
                        records = await self.worms_client.get_species_by_common_name(client, marine_query.common_name)
                        await process.log(f"Searching WoRMS by common name: {marine_query.common_name}")
                    
                    if not records:
                        await context.reply(f"No marine species found matching '{search_name}'")
                        return

                    # Get the primary species record
                    primary_species = records[0]
                    aphia_id = primary_species.AphiaID
                    scientific_name = primary_species.scientificname

                    await process.log(f"Found species: {scientific_name} (AphiaID: {aphia_id})")

                    # Fetch all additional data based on parameters
                    await process.log("Retrieving comprehensive marine species data")
                    
                    synonyms_data = None
                    distribution_data = None
                    vernaculars_data = None
                    classification_data = None
                    children_data = None
                    
                    if params.include_synonyms:
                        synonyms_data = await self.worms_client.get_synonyms(client, aphia_id)
                    
                    if params.include_distribution:
                        distribution_data = await self.worms_client.get_distribution(client, aphia_id)
                    
                    if params.include_vernaculars:
                        vernaculars_data = await self.worms_client.get_vernaculars(client, aphia_id)
                    
                    if params.include_classification:
                        classification_data = await self.worms_client.get_classification(client, aphia_id)
                    
                    if params.include_children:
                        children_data = await self.worms_client.get_children(client, aphia_id)

                    # Create the complete marine species data model
                    complete_data = CompleteMarineSpeciesData(
                        species=primary_species,
                        synonyms=synonyms_data,
                        distribution=distribution_data,
                        vernaculars=vernaculars_data,
                        classification=classification_data,
                        children=children_data,
                        aphia_id=aphia_id,
                        scientific_name=scientific_name,
                        search_term=search_name
                    )

                    # Create artifact with the structured data
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Complete marine species data for {scientific_name}",
                        content=complete_data.model_dump_json(indent=2),
                        uris=self._build_worms_uris(aphia_id),
                        metadata={
                            "aphia_id": aphia_id,
                            "scientific_name": scientific_name,
                            "search_term": search_name,
                            "retrieved_at": complete_data.retrieved_at.isoformat(),
                            "data_sources": [
                                source for source, included in [
                                    ("synonyms", params.include_synonyms),
                                    ("distribution", params.include_distribution),
                                    ("vernaculars", params.include_vernaculars),
                                    ("classification", params.include_classification),
                                    ("children", params.include_children)
                                ] if included
                            ]
                        }
                    )

                    await process.log(f"Successfully retrieved complete marine species data for {scientific_name}")

                    # Generate a summary for the reply
                    summary_parts = [f"Found marine species data for {scientific_name} (AphiaID: {aphia_id})"]
                    
                    if synonyms_data:
                        summary_parts.append(f"{len(synonyms_data)} synonyms")
                    if distribution_data:
                        summary_parts.append(f"{len(distribution_data)} distribution records")
                    if vernaculars_data:
                        summary_parts.append(f"{len(vernaculars_data)} vernacular names")
                    if classification_data:
                        summary_parts.append(f"taxonomic classification with {len(classification_data)} levels")
                    if children_data:
                        summary_parts.append(f"{len(children_data)} child taxa")

                    reply = f"{summary_parts[0]}. The artifact contains comprehensive taxonomic information from WoRMS including {', '.join(summary_parts[1:])}."
                    await context.reply(reply)

            except InstructorRetryException:
                await context.reply("Could not extract marine species information from the request")
            except ValidationError as e:
                await context.reply(f"Data validation error: {str(e)}")
            except Exception as e:
                await context.reply("An error occurred while retrieving marine species data", data={"error": str(e)})