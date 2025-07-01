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
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess, TextPart
from ichatbio.types import AgentCard, AgentEntrypoint

dotenv.load_dotenv()

#Configs for LLM and woRMS

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    WORMS_BASE_URL = "https://www.marinespecies.org/rest"
    MODEL_NAME = "llama3-70b-8192"

#Data Models

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

class MarineSpecies(BaseModel):
    AphiaID: int = Field(description="Unique WoRMS identifier number for this taxonomic record")
    scientificname: str = Field(description="Currently accepted scientific binomial name of the species")
    authority: Optional[str] = Field(None, description="Author(s) who first described this species, with publication year")
    rank: Optional[str] = Field(None, description="Taxonomic rank (e.g., 'Species', 'Genus', 'Family', 'Subspecies')")
    status: Optional[str] = Field(None, description="Nomenclatural status (e.g., 'accepted', 'synonym', 'unaccepted')")
    kingdom: Optional[str] = Field(None, description="Kingdom-level taxonomic classification (e.g., 'Animalia')")
    phylum: Optional[str] = Field(None, description="Phylum-level taxonomic classification (e.g., 'Chordata', 'Cnidaria')")
    class_: Optional[str] = Field(None, alias="class", description="Class-level taxonomic classification (e.g., 'Mammalia', 'Actinopterygii')")
    order: Optional[str] = Field(None, description="Order-level taxonomic classification (e.g., 'Cetacea', 'Carcharhiniformes')")
    family: Optional[str] = Field(None, description="Family-level taxonomic classification (e.g., 'Delphinidae', 'Lamnidae')")
    genus: Optional[str] = Field(None, description="Genus-level taxonomic classification (e.g., 'Orcinus', 'Carcharodon')")
    citation: Optional[str] = Field(None, description="Full bibliographic citation for this taxonomic record")
    lsid: Optional[str] = Field(None, description="Life Science Identifier - persistent unique identifier URI")

class Synonym(BaseModel):
    AphiaID: int = Field(description="WoRMS identifier for this synonym record")
    scientificname: str = Field(description="Synonymous scientific name that refers to the same species")
    authority: Optional[str] = Field(None, description="Author and year who published this synonymous name")
    status: Optional[str] = Field(None, description="Status of this synonym (e.g., 'synonym', 'objective synonym')")

class Distribution(BaseModel):
    locality: str = Field(description="Geographic location or region where this species is found (e.g., 'Atlantic Ocean', 'Mediterranean Sea')")
    status: Optional[str] = Field(None, description="Presence status in this location (e.g., 'native', 'introduced', 'uncertain')")
    gazetteer: Optional[str] = Field(None, description="Geographic reference system or authority used for this location")

class Vernacular(BaseModel):
    vernacular: str = Field(description="Common name of the species in local language (e.g., 'killer whale', 'orca')")
    language: Optional[str] = Field(None, description="Language code or name for this vernacular name (e.g., 'eng', 'fra', 'Spanish')")

class Classification(BaseModel):
    rank: str = Field(description="Taxonomic rank level (e.g., 'Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus')")
    scientificname: str = Field(description="Scientific name at this taxonomic rank level")

class ChildTaxon(BaseModel):
    AphiaID: int = Field(description="WoRMS identifier for this child taxon")
    scientificname: str = Field(description="Scientific name of the subordinate taxon (species, subspecies, etc.)")
    rank: str = Field(description="Taxonomic rank of this child taxon (e.g., 'Species', 'Subspecies', 'Variety')")

#woRMS API Client

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

#Data Processing

class MarineDataProcessor:
    def safe_parse(self, model_class, data):
        try:
            return model_class(**data)
        except Exception as e:
            return None

    def process_list_data(self, model_class, raw_data):
        if not raw_data:
            return []
        parsed_items = [self.safe_parse(model_class, item) for item in raw_data]
        return [item for item in parsed_items if item]

    def process_all_marine_data(self, species_data, synonyms_data, distribution_data,
                                 vernaculars_data, classification_data, children_data):
        return {
            'synonyms': self.process_list_data(Synonym, synonyms_data),
            'distribution': self.process_list_data(Distribution, distribution_data),
            'vernaculars': self.process_list_data(Vernacular, vernaculars_data),
            'classification': self.process_list_data(Classification, classification_data),
            'children': self.process_list_data(ChildTaxon, children_data)
        }

#Main Agent class

class MarineAgent(IChatBioAgent):
    def __init__(self):
        self.worms_client = WoRMSClient()
        self.data_processor = MarineDataProcessor()
        self.agent_card = AgentCard(
            name="Marine Species Agent",
            description="Retrieves marine animal information from WoRMS using scientific names.",
            url="http://localhost:9999",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_info",
                    description="Returns detailed marine animal information by scientific name",
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

    async def fetch_all_marine_data(self, query: MarineQueryModel):
        async with httpx.AsyncClient() as client:
            records = None
            search_term = None
            
            #trying scientific name first
            if query.scientificname:
                records = await self.worms_client.get_species_by_name(client, query.scientificname)
                search_term = query.scientificname
            
            #common name if scientific name failed
            if not records and query.common_name:
                records = await self.worms_client.get_species_by_common_name(client, query.common_name)
                search_term = query.common_name
            
            if not records:
                return None, None, search_term

            species_data = records[0] if isinstance(records, list) else records
            species = MarineSpecies(**species_data)
            aphia_id = species.AphiaID

            #fetching all additional data
            synonyms_data = await self.worms_client.get_synonyms(client, aphia_id)
            distribution_data = await self.worms_client.get_distribution(client, aphia_id)
            vernaculars_data = await self.worms_client.get_vernaculars(client, aphia_id)
            classification_data = await self.worms_client.get_classification(client, aphia_id)
            children_data = await self.worms_client.get_children(client, aphia_id)

            #returning raw data instead of processed
            raw_data = {
                'species': species_data,
                'synonyms': synonyms_data,
                'distribution': distribution_data,
                'vernaculars': vernaculars_data,
                'classification': classification_data,
                'children': children_data
            }

            processed_data = self.data_processor.process_all_marine_data(
                species_data, synonyms_data, distribution_data,
                vernaculars_data, classification_data, children_data
            )

            return species, raw_data, processed_data, search_term

    def build_worms_uris(self, species: MarineSpecies) -> List[str]:
        uris = []
        uris.append(f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={species.AphiaID}")
        if species.lsid:
            uris.append(species.lsid)
        return uris

    @staticmethod
    def build_structured_prompt(species, processed_data):
        lines = [
            f"Scientific Name: {species.scientificname}",
            f"Authority: {species.authority or 'N/A'}",
            f"Rank: {species.rank or 'N/A'}",
            f"Status: {species.status or 'N/A'}",
            "",
            "Classification Hierarchy:",
            *[f"  - {cl.rank}: {cl.scientificname}" for cl in processed_data['classification']],
            "",
            "Synonyms:",
            *[f"  - {syn.scientificname} ({syn.status or 'N/A'})" for syn in processed_data['synonyms']],
            "",
            "Distribution:",
            *[f"  - {dist.locality} ({dist.status or 'N/A'})" for dist in processed_data['distribution']],
            "",
            "Vernacular Names:",
            *[f"  - {vern.vernacular} ({vern.language or 'N/A'})" for vern in processed_data['vernaculars']],
            "",
            "Children Taxa:",
            *[f"  - {ch.scientificname} ({ch.rank})" for ch in processed_data['children']],
        ]
        return "\n".join(lines)

    async def synthesize_response(self, species: MarineSpecies, processed_data: dict) -> str:
        openai_client = AsyncOpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
        )

        detailed_prompt = self.build_structured_prompt(species, processed_data)

        response = await openai_client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a marine biology assistant. Provide informative, conversational responses."},
                {"role": "user", "content": detailed_prompt}
            ],
            max_tokens=700,
            temperature=0.7,
        )

        return response.choices[0].message.content

    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: Optional[BaseModel],
    ) -> None:
        async with context.begin_process("Retrieving marine species data") as process:
            process: IChatBioAgentProcess

            try:
                await process.log("Extracting marine species info from user query")
                marine_query = await self.extract_query_info(request)

                if not marine_query.scientificname and not marine_query.common_name:
                    
                    #TextPart instead of TextMessage
                    await process.send(TextPart(
                        text="I couldn't identify a marine species in your question. Try asking about a specific marine animal like 'killer whale' or 'Orcinus orca'."
                    ))
                    return

                search_name = marine_query.scientificname or marine_query.common_name
                await process.log("Species identified", data={"search_term": search_name})

                species, raw_data, processed_data, search_term = await self.fetch_all_marine_data(marine_query)

                if not species:
                    suggestion_msg = f"No marine species found for '{search_term}'. "
                    if search_term and len(search_term) > 3:
                        suggestion_msg += "Try checking the spelling or using the scientific name."
                    else:
                        suggestion_msg += "Try being more specific, like 'great white shark' or 'Carcharodon carcharias'."
                    await process.send(TextPart(text=suggestion_msg))
                    return

                await process.log("Species found", data={"AphiaID": species.AphiaID})

                #artifact 1–6: Raw Data
                for key, endpoint in {
                    'species': f"AphiaRecordsByName/{species.scientificname}",
                    'synonyms': f"AphiaSynonymsByAphiaID/{species.AphiaID}",
                    'distribution': f"AphiaDistributionsByAphiaID/{species.AphiaID}",
                    'vernaculars': f"AphiaVernacularsByAphiaID/{species.AphiaID}",
                    'classification': f"AphiaClassificationByAphiaID/{species.AphiaID}",
                    'children': f"AphiaChildrenByAphiaID/{species.AphiaID}"
                }.items():
                    if raw_data.get(key):
                        await process.create_artifact(
                            mimetype="application/json",
                            description=f"Raw {key} data for {species.scientificname}",
                            content=json.dumps(raw_data[key], indent=2),
                            uris=[f"https://www.marinespecies.org/rest/{endpoint}"],
                            metadata={"api_endpoint": key, "aphia_id": species.AphiaID}
                        )

                await process.log("Generating natural language summary from species data")
                answer = await self.synthesize_response(species, processed_data)

                #artifact 7: Final summary
                await process.create_artifact(
                    mimetype="text/markdown",
                    description=f"Marine species summary for {species.scientificname}",
                    content=answer,
                    uris=self.build_worms_uris(species),
                    metadata={
                        "aphia_id": species.AphiaID,
                        "scientificname": species.scientificname
                    }
                )

                await process.send(TextPart(text=answer))

            except InstructorRetryException:
                await process.send(TextPart(text="I couldn't understand your question. Try asking about a specific marine animal."))
            except Exception as e:
                await process.send(TextPart(text=f"An error occurred while retrieving marine species info: {str(e)}"))

        await context.reply("Marine species data request completed.")