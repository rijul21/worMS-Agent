# worms_agent_server.py (Combined with worms_client)
import os
import yaml
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict
from urllib.parse import urlencode, quote
import cloudscraper
from typing_extensions import override
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext  
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint

# Import your existing agent workflow and Pydantic models
from worms_agent import (
    WoRMSiChatBioAgent,
    MarineSynonymsParams,
    MarineDistributionParams,
    MarineVernacularParams,
    MarineSourcesParams,
    MarineRecordParams,
    MarineClassificationParams,
    MarineChildrenParams
)

# ============================================================================
# WoRMS Client Code (from worms_client.py)
# ============================================================================

# Parameter Models - 7 endpoints total
class SpeciesSearchParams(BaseModel):
    """Parameters for searching marine species in WoRMS"""
    scientific_name: str = Field(..., 
        description="Scientific name to search for",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    
    like: Optional[bool] = Field(False,
        description="Use fuzzy matching for names"
    )
    
    marine_only: Optional[bool] = Field(True,
        description="Return only marine species"
    )

class SynonymsParams(BaseModel):
    """Parameters for getting synonyms of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get synonyms for",
        examples=[137205, 104625, 137094]
    )

class DistributionParams(BaseModel):
    """Parameters for getting distribution data of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get distribution for",
        examples=[137205, 104625, 137094]
    )

class VernacularParams(BaseModel):
    """Parameters for getting vernacular/common names of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get vernacular names for",
        examples=[137205, 104625, 137094]
    )

class SourcesParams(BaseModel):
    """Parameters for getting literature sources/references of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get sources for",
        examples=[137205, 104625, 137094]
    )

class RecordParams(BaseModel):
    """Parameters for getting basic taxonomic record of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get record for",
        examples=[137205, 104625, 137094]
    )

class ClassificationParams(BaseModel):
    """Parameters for getting taxonomic classification of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get classification for",
        examples=[137205, 104625, 137094]
    )

class ChildrenParams(BaseModel):
    """Parameters for getting child taxa of a species"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get child taxa for",
        examples=[137205, 104625, 137094]
    )

class NoParams(BaseModel):
    """An empty model for entrypoints that require no parameters."""
    pass

class WoRMS:
    def __init__(self):
        self.worms_api_base_url = self._get_config_value("WORMS_API_URL", "https://www.marinespecies.org/rest")
        
        self.session = cloudscraper.create_scraper()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    def _get_config_value(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get configuration value from environment or YAML file"""
        value = os.getenv(key)
        if value is None and os.path.exists('env.yaml'):
            with open('env.yaml', 'r') as f:
                config = yaml.safe_load(f) or {}
                value = config.get(key, default)
        return value if value is not None else default

    # URL Building Methods - 5 core endpoints + search
    def build_species_search_url(self, params: SpeciesSearchParams) -> str:
        """Build URL for searching species by name"""
        encoded_name = quote(params.scientific_name)
        query_params = {}
        
        if params.like is not None:
            query_params['like'] = str(params.like).lower()
        if params.marine_only is not None:
            query_params['marine_only'] = str(params.marine_only).lower()
            
        query_string = urlencode(query_params) if query_params else ''
        base_url = f"{self.worms_api_base_url}/AphiaRecordsByName/{encoded_name}"
        return f"{base_url}?{query_string}" if query_string else base_url

    def build_synonyms_url(self, params: SynonymsParams) -> str:
        """Build URL for getting species synonyms"""
        return f"{self.worms_api_base_url}/AphiaSynonymsByAphiaID/{params.aphia_id}"

    def build_distribution_url(self, params: DistributionParams) -> str:
        """Build URL for getting species distribution"""
        return f"{self.worms_api_base_url}/AphiaDistributionsByAphiaID/{params.aphia_id}"

    def build_vernacular_url(self, params: VernacularParams) -> str:
        """Build URL for getting species vernacular/common names"""
        return f"{self.worms_api_base_url}/AphiaVernacularsByAphiaID/{params.aphia_id}"

    def build_sources_url(self, params: SourcesParams) -> str:
        """Build URL for getting species literature sources/references"""
        return f"{self.worms_api_base_url}/AphiaSourcesByAphiaID/{params.aphia_id}"

    def build_record_url(self, params: RecordParams) -> str:
        """Build URL for getting basic species taxonomic record"""
        return f"{self.worms_api_base_url}/AphiaRecordByAphiaID/{params.aphia_id}"

    def build_classification_url(self, params: ClassificationParams) -> str:
        """Build URL for getting species taxonomic classification"""
        return f"{self.worms_api_base_url}/AphiaClassificationByAphiaID/{params.aphia_id}"

    def build_children_url(self, params: ChildrenParams) -> str:
        """Build URL for getting species child taxa"""
        return f"{self.worms_api_base_url}/AphiaChildrenByAphiaID/{params.aphia_id}"

    # Request execution methods (following ALA pattern)
    def execute_request(self, url: str) -> Dict:
        """Execute GET request and return JSON response"""
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                raise ConnectionError(f"API response was not JSON. Response: {response.text[:200]}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed: {e}")

    # Helper method for getting AphiaID from species name
    def get_species_aphia_id(self, scientific_name: str) -> Optional[int]:
        """Get AphiaID for a species name - synchronous helper"""
        params = SpeciesSearchParams(scientific_name=scientific_name)
        url = self.build_species_search_url(params)
        
        try:
            result = self.execute_request(url)
            if isinstance(result, list) and result:
                return result[0].get('AphiaID')
            elif isinstance(result, dict):
                return result.get('AphiaID')
            return None
        except Exception:
            return None

# ============================================================================
# Agent Server Code (from worms_agent_server.py)
# ============================================================================

# --- AgentCard definition with 7 endpoints ---
card = AgentCard(
    name="WoRMS Marine Species Agent",
    description="Retrieves detailed marine species information from WoRMS (World Register of Marine Species) database including synonyms, distribution, common names, literature sources, taxonomic records, classification, and child taxa.",
    icon="https://www.marinespecies.org/images/WoRMS_logo.png",
    url="http://localhost:9999",  
    entrypoints=[
        AgentEntrypoint(
            id="get_synonyms",
            description="Get synonyms and alternative names for a marine species from WoRMS.",
            parameters=MarineSynonymsParams
        ),
        AgentEntrypoint(
            id="get_distribution",
            description="Get distribution data and geographic locations for a marine species from WoRMS.",
            parameters=MarineDistributionParams
        ),
        AgentEntrypoint(
            id="get_vernacular_names",
            description="Get vernacular/common names for a marine species in different languages from WoRMS.",
            parameters=MarineVernacularParams
        ),
        AgentEntrypoint(
            id="get_sources",
            description="Get literature sources and references for a marine species from WoRMS.",
            parameters=MarineSourcesParams
        ),
        AgentEntrypoint(
            id="get_record",
            description="Get basic taxonomic record and classification for a marine species from WoRMS.",
            parameters=MarineRecordParams
        ),
        AgentEntrypoint(
            id="get_taxonomy",
            description="Get complete taxonomic classification hierarchy for a marine species from WoRMS.",
            parameters=MarineClassificationParams
        ),
        AgentEntrypoint(
            id="get_marine_info",
            description="Get child taxa (subspecies, varieties, forms) for a marine species from WoRMS.",
            parameters=MarineChildrenParams
        )
    ]
)

# --- Implement the iChatBio agent class ---
class WoRMSAgent(IChatBioAgent):
    def __init__(self):
        self.workflow_agent = WoRMSiChatBioAgent()

    @override
    def get_agent_card(self) -> AgentCard:
        """Returns the agent's metadata card."""
        return card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: BaseModel):
        """Executes the requested agent entrypoint using the provided context."""
        
        # Debug logging 
        print(f"=== WoRMS DEBUG INFO ===")
        print(f"request: {request}")
        print(f"entrypoint: {entrypoint}")
        print(f"params type: {type(params)}")
        print(f"params: {params}")
        print(f"========================")
        
        if entrypoint == "get_synonyms":
            await self.workflow_agent.run_get_synonyms(context, params)
        elif entrypoint == "get_distribution":
            await self.workflow_agent.run_get_distribution(context, params)
        elif entrypoint == "get_vernacular_names":
            await self.workflow_agent.run_get_vernacular_names(context, params)
        elif entrypoint == "get_sources":
            await self.workflow_agent.run_get_sources(context, params)
        elif entrypoint == "get_record":
            await self.workflow_agent.run_get_record(context, params)
        elif entrypoint == "get_taxonomy":
            await self.workflow_agent.run_get_classification(context, params)
        elif entrypoint == "get_marine_info":
            await self.workflow_agent.run_get_children(context, params)
        else:
            # Handle unexpected entrypoints 
            await context.reply(f"Unknown entrypoint '{entrypoint}' received. Request was: '{request}'")
            raise ValueError(f"Unsupported entrypoint: {entrypoint}")

if __name__ == "__main__":
    agent = WoRMSAgent()
    print(f"Starting iChatBio agent server for '{card.name}' at http://localhost:9999")
    print("Available endpoints:")
    for ep in card.entrypoints:
        print(f"  - {ep.id}: {ep.description}")
    run_agent_server(agent, host="0.0.0.0", port=9999)