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
from ichatbio.agent_response import ResponseContext, ArtifactResponse, DirectResponse, ProcessLogResponse
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
                return None
            data = await resp.json()
            return data
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
            return [WoRMSRecord(**data)]
        except ValidationError:
            return None

    async def get_vernaculars_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSVernacular]]:
        endpoint = f"/AphiaVernacularsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            return [WoRMSVernacular(**record) for record in data] if isinstance(data, list) else [WoRMSVernacular(**data)]
        except ValidationError:
            return None

    async def get_synonyms_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSSynonym]]:
        endpoint = f"/AphiaSynonymsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            return [WoRMSSynonym(**record) for record in data] if isinstance(data, list) else [WoRMSSynonym(**data)]
        except ValidationError:
            return None

    async def get_distributions_by_aphia_id(self, client: httpx.AsyncClient, aphia_id: int) -> Optional[List[WoRMSDistribution]]:
        endpoint = f"/AphiaDistributionsByAphiaID/{aphia_id}"
        data = await self.fetch_json(client, endpoint)
        if not data:
            return None
        try:
            return [WoRMSDistribution(**record) for record in data] if isinstance(data, list) else [WoRMSDistribution(**data)]
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
            return [WoRMSSource(**record) for record in data] if isinstance(data, list) else [WoRMSSource(**data)]
        except ValidationError:
            return None

class MarineAgent(IChatBioAgent):
    def __init__(self):
        self.worms_client = WoRMSClient()

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Marine Species Data Agent",
            description="Retrieves marine species information from WoRMS, including taxonomic data, common names, synonyms, distributions, attributes, and sources.",
            url="http://localhost:9999",
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
            return marine_query
        except (InstructorRetryException, ValidationError):
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

    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineParameters,
    ) -> None:
        if entrypoint != "get_marine_info":
            await context.reply(f"Unknown entrypoint: {entrypoint}")
            return

        async with context.begin_process("Analyzing marine species request") as process:
            try:
                marine_query = await self.extract_query_info(request)

                if not marine_query.scientificname and not marine_query.common_name:
                    await context.reply("No scientific or common name identified in the request")
                    return

                search_name = marine_query.scientificname or marine_query.common_name
                await process.log(f"Identified marine species: {search_name}", {
                    "scientific_name": marine_query.scientificname,
                    "common_name": marine_query.common_name
                })

                async with httpx.AsyncClient() as client:
                    records = await self.worms_client.get_species_by_name(client, search_name)
                    await process.log(f"Searching WoRMS by name: {search_name}", {
                        "search_url": f"https://www.marinespecies.org/rest/AphiaRecordsByName/{quote(search_name)}?like=false&marine_only=true"
                    })

                    if not records:
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
                        "search_term": search_name,
                        "retrieved_at": datetime.now(timezone.utc)
                    }

                    try:
                        complete_data = CompleteMarineSpeciesData(**species_data)
                        content = complete_data.model_dump_json()
                    except ValidationError as e:
                        await context.reply(f"Error processing marine species data: {str(e)}")
                        return

                    artifact_response = ArtifactResponse(
                        mimetype="application/json",
                        description=f"Marine species data for {primary_species.scientificname}",
                        uris=self._build_worms_uris(aphia_id),
                        content=content,
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
                    await context.reply(artifact_response)

                    await process.log(f"Successfully retrieved marine species data for {primary_species.scientificname}")

                    vernacular_names = [v.vernacular for v in vernaculars] if vernaculars else []
                    synonym_names = [s.scientificname for s in synonyms] if synonyms else []
                    distribution_locations = [d.locality for d in distributions if d.locality] if distributions else []
                    response_text = (
                        f"Found marine species data for {primary_species.scientificname} (AphiaID: {aphia_id}). "
                        f"Common names: {', '.join(vernacular_names) if vernacular_names else 'None'}. "
                        f"Synonyms: {', '.join(synonym_names) if synonym_names else 'None'}. "
                        f"Distributions: {', '.join(distribution_locations) if distribution_locations else 'None'}. "
                        f"Attributes: {len(attributes) if attributes else 0} found. "
                        f"Sources: {len(sources) if sources else 0} found. "
                        "The artifact contains taxonomic information, common names, synonyms, distributions, attributes, and sources from WoRMS."
                    )
                    await context.reply(response_text)

            except Exception as e:
                await context.reply(f"An error occurred while processing the request: {str(e)}")