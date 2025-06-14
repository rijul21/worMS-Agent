import os
import dotenv
import instructor
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator, override
from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, AgentEntrypoint, ProcessMessage, Message, TextMessage, ArtifactMessage
import httpx

dotenv.load_dotenv()

##Configs for LLM and woRMS

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    WORMS_BASE_URL = "https://www.marinespecies.org/rest"
    MODEL_NAME = "llama3-70b-8192"

#Data Models

class MarineQueryModel(BaseModel):
    scientificname: str = Field(..., description="Scientific name of the marine animal, e.g. Orcinus orca")

class EmptyModel(BaseModel):
    ...

class MarineSpecies(BaseModel):
    AphiaID: int
    scientificname: str
    authority: Optional[str] = None
    rank: Optional[str] = None
    status: Optional[str] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    citation: Optional[str] = None
    lsid: Optional[str] = None

class Synonym(BaseModel):
    AphiaID: int
    scientificname: str
    authority: Optional[str] = None
    status: Optional[str] = None

class Distribution(BaseModel):
    locality: str
    status: Optional[str] = None
    gazetteer: Optional[str] = None

class Vernacular(BaseModel):
    vernacular: str
    language: Optional[str] = None

class Classification(BaseModel):
    rank: str
    scientificname: str

class ChildTaxon(BaseModel):
    AphiaID: int
    scientificname: str
    rank: str


## woRMS API Client

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


## Data Processing

class MarineDataProcessor:
    def safe_parse(self, model_class, data):
        try:
            return model_class(**data)
        except Exception as e:
            print(f"Skipping invalid records for {model_class.__name__}: {e}")
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

## Main Agent class
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

    async def extract_scientific_name(self, request: str) -> MarineQueryModel:
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
                        "You are a marine taxonomy expert that extracts the scientific name "
                        "of a marine animal from user questions. If none found, respond accordingly."
                    )
                },
                {"role": "user", "content": request}
            ],
            max_retries=3
        )
        return marine_query

    async def fetch_all_marine_data(self, scientific_name: str):
        async with httpx.AsyncClient() as client:
            # Get main species record
            records = await self.worms_client.get_species_by_name(client, scientific_name)
            if not records:
                return None, None
            
            species_data = records[0] if isinstance(records, list) else records
            species = MarineSpecies(**species_data)
            aphia_id = species.AphiaID
            
            # Fetch all related data
            synonyms_data = await self.worms_client.get_synonyms(client, aphia_id)
            distribution_data = await self.worms_client.get_distribution(client, aphia_id)
            vernaculars_data = await self.worms_client.get_vernaculars(client, aphia_id)
            classification_data = await self.worms_client.get_classification(client, aphia_id)
            children_data = await self.worms_client.get_children(client, aphia_id)
            
            # Process all data
            processed_data = self.data_processor.process_all_marine_data(
                species_data, synonyms_data, distribution_data, 
                vernaculars_data, classification_data, children_data
            )
            
            return species, processed_data

    async def synthesize_response(self, species: MarineSpecies, processed_data: dict) -> str:
        openai_client = AsyncOpenAI(
            api_key=Config.GROQ_API_KEY,
            base_url=Config.GROQ_BASE_URL,
        )
        
        prompt_lines = [
            f"User asked about: {species.scientificname}",
            "Here is the structured data from WoRMS:",
            f"Species info: {species.model_dump_json(indent=2)}",
            f"Synonyms: {[syn.scientificname for syn in processed_data['synonyms']]}",
            f"Distribution: {[dist.locality for dist in processed_data['distribution']]}",
            f"Vernacular names: {[vern.vernacular for vern in processed_data['vernaculars']]}",
            f"Classification hierarchy: {[f'{cl.rank}: {cl.scientificname}' for cl in processed_data['classification']]}",
            f"Children taxa: {[ch.scientificname for ch in processed_data['children']]}",
            "Please answer the user's question in a clear, friendly, scientific way using this information."
        ]
        detailed_prompt = "\n".join(prompt_lines)

        response = await openai_client.chat.completions.create(
            model=Config.MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a marine biology assistant."},
                {"role": "user", "content": detailed_prompt}
            ],
            max_tokens=700,
            temperature=0.7,
        )
        
        return response.choices[0].message.content

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[Message, None]:
        try:
            # Extracting scientific name
            yield ProcessMessage(summary="Analyzing request", description="Extracting scientific name")
            marine_query = await self.extract_scientific_name(request)
            
            if not marine_query.scientificname.strip():
                yield TextMessage(text="Sorry, I couldn't detect a scientific name in your prompt.")
                return

            yield ProcessMessage(summary="Species name extracted", description=marine_query.scientificname)

            # Fetching all marine data
            species, processed_data = await self.fetch_all_marine_data(marine_query.scientificname)
            
            if not species:
                yield TextMessage(text=f"No marine species found for: {marine_query.scientificname}")
                return

            yield ProcessMessage(summary="Species found", description=f"AphiaID: {species.AphiaID}")

            # Synthesize response
            answer = await self.synthesize_response(species, processed_data)

            # Returning results
            yield ArtifactMessage(
                mimetype="text/markdown",
                description=f"Marine species info for {species.scientificname}",
                content=answer,
                metadata={
                    "AphiaID": species.AphiaID,
                    "scientificname": species.scientificname,
                    "synonyms": [syn.dict() for syn in processed_data['synonyms']],
                    "distribution": [dist.dict() for dist in processed_data['distribution']],
                    "vernaculars": [vern.dict() for vern in processed_data['vernaculars']],
                    "classification": [cl.dict() for cl in processed_data['classification']],
                    "children": [ch.dict() for ch in processed_data['children']]
                }
            )

            yield TextMessage(text=answer)

        except InstructorRetryException:
            yield TextMessage(text="Sorry, I couldn't extract a scientific name from your request.")
        except Exception as e:
            yield TextMessage(text=f"An error occurred while retrieving marine species info: {str(e)}")