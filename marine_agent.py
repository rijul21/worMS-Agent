import os
import json
import dotenv
import instructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, override
import httpx

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

dotenv.load_dotenv()

# Configs for LLM and woRMS
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    WORMS_BASE_URL = "https://www.marinespecies.org/rest"
    MODEL_NAME = "llama3-70b-8192"

# Data Models
class MarineQueryModel(BaseModel):
    scientificname: Optional[str] = Field(
        None, 
        description="Scientific binomial name of the marine species (e.g., 'Orcinus orca', 'Carcharodon carcharias', 'Balaenoptera musculus')"
    )
    common_name: Optional[str] = Field(
        None, 
        description="Common or vernacular name of the marine animal in English (e.g., 'killer whale', 'great white shark', 'blue whale')"
    )

class EmptyModel(BaseModel):
    """Empty model for endpoints that don't require parameters"""
    ...

# woRMS API Client
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

    async def get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str):
        endpoint = f"/AphiaRecordsByName/{scientific_name}?like=false&marine_only=true"
        return await self.fetch_json(client, endpoint)

    async def get_species_by_common_name(self, client: httpx.AsyncClient, common_name: str):
        endpoint = f"/AphiaRecordsByVernacular/{common_name}?like=true&offset=1"
        return await self.fetch_json(client, endpoint)

    async def get_synonyms(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        return await self.fetch_json(client, endpoint)

    async def get_distribution(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        return await self.fetch_json(client, endpoint)

    async def get_vernaculars(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaVernacularsByAphiaID/{aphia_id}"
        return await self.fetch_json(client, endpoint)

    async def get_classification(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaClassificationByAphiaID/{aphia_id}"
        return await self.fetch_json(client, endpoint)

    async def get_children(self, client: httpx.AsyncClient, aphia_id: int):
        endpoint = f"/AphiaChildrenByAphiaID/{aphia_id}"
        return await self.fetch_json(client, endpoint)

# Main Agent class
class MarineAgent(IChatBioAgent):
    def __init__(self):
        self.worms_client = WoRMSClient()
        self.agent_card = AgentCard(
            name="Marine Species Data Agent",
            description="Retrieves detailed marine species information from WoRMS using scientific or common names.",
            url="http://localhost:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_info",
                    description="Returns detailed marine species data",
                    parameters=EmptyModel
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

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
        params: Optional[BaseModel],
    ) -> None:
        async with context.begin_process("Analyzing marine species request") as process:
            process: IChatBioAgentProcess

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
                    species_data = records[0] if isinstance(records, list) else records
                    aphia_id = species_data.get('AphiaID')
                    scientific_name = species_data.get('scientificname', 'Unknown')

                    await process.log(f"Found species: {scientific_name} (AphiaID: {aphia_id})")

                    # Fetch all additional data
                    await process.log("Retrieving comprehensive marine species data")
                    
                    synonyms_data = await self.worms_client.get_synonyms(client, aphia_id)
                    distribution_data = await self.worms_client.get_distribution(client, aphia_id)
                    vernaculars_data = await self.worms_client.get_vernaculars(client, aphia_id)
                    classification_data = await self.worms_client.get_classification(client, aphia_id)
                    children_data = await self.worms_client.get_children(client, aphia_id)

                    # Combine all data
                    all_marine_data = {
                        'species': species_data,
                        'synonyms': synonyms_data,
                        'distribution': distribution_data,
                        'vernaculars': vernaculars_data,
                        'classification': classification_data,
                        'children': children_data
                    }

                    # Create artifact with all the marine data
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Complete marine species data for {scientific_name}",
                        content=json.dumps(all_marine_data, indent=2),
                        uris=self._build_worms_uris(aphia_id),
                        metadata={
                            "aphia_id": aphia_id,
                            "scientific_name": scientific_name,
                            "search_term": search_name,
                            "data_sources": ["species", "synonyms", "distribution", "vernaculars", "classification", "children"]
                        }
                    )

                    await process.log(f"Successfully retrieved complete marine species data for {scientific_name}")

                    # Simple reply like POWO agent
                    await context.reply(f"Found marine species data for {scientific_name} (AphiaID: {aphia_id}). The artifact contains comprehensive taxonomic information from WoRMS.")

            except InstructorRetryException:
                await context.reply("Could not extract marine species information from the request")
            except Exception as e:
                await context.reply("An error occurred while retrieving marine species data", data={"error": str(e)})