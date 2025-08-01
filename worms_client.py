import os
import yaml
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict
from urllib.parse import urlencode, quote
import cloudscraper

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