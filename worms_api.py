import os
import yaml
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict
from urllib.parse import urlencode, quote
import cloudscraper

# Define parameter classes for different WoRMS API endpoints
class SpeciesSearchParams(BaseModel):
    """Search parameters for finding marine species in the WoRMS database"""
    scientific_name: str = Field(..., 
        description="The scientific name you want to search for",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    
    like: Optional[bool] = Field(False,
        description="Whether to use fuzzy matching when searching names"
    )
    
    marine_only: Optional[bool] = Field(True,
        description="Limit results to marine species only"
    )

class SynonymsParams(BaseModel):
    """What we need to get synonym information for a species"""
    aphia_id: int = Field(...,
        description="WoRMS AphiaID number for the species you want synonyms for",
        examples=[137205, 104625, 137094]
    )

class DistributionParams(BaseModel):
    """Parameters needed to fetch where a species is found geographically"""
    aphia_id: int = Field(...,
        description="WoRMS AphiaID to get distribution data for",
        examples=[137205, 104625, 137094]
    )

class VernacularParams(BaseModel):
    """Get common names in different languages for a species"""
    aphia_id: int = Field(...,
        description="The species AphiaID to look up common names for",
        examples=[137205, 104625, 137094]
    )

class SourcesParams(BaseModel):
    """Parameters to retrieve scientific literature about a species"""
    aphia_id: int = Field(...,
        description="AphiaID of the species to find literature sources for",
        examples=[137205, 104625, 137094]
    )

class RecordParams(BaseModel):
    """Get the basic taxonomic information for a species"""
    aphia_id: int = Field(...,
        description="Species AphiaID to retrieve basic record information",
        examples=[137205, 104625, 137094]
    )

class ClassificationParams(BaseModel):
    """Parameters for getting the full taxonomic hierarchy of a species"""
    aphia_id: int = Field(...,
        description="AphiaID to get complete taxonomic classification for",
        examples=[137205, 104625, 137094]
    )

class ChildrenParams(BaseModel):
    """Find subspecies or other child taxa under a given species"""
    aphia_id: int = Field(...,
        description="Parent AphiaID to find child taxa for",
        examples=[137205, 104625, 137094]
    )

class NoParams(BaseModel):
    """Used when an endpoint doesn't need any special parameters"""
    pass

class WoRMS:
    def __init__(self):
        # Set up the base URL for WoRMS API calls
        self.worms_api_base_url = self._load_config_setting("WORMS_API_URL", "https://www.marinespecies.org/rest")
        
        # Create a session that can handle CloudFlare protection
        self.session = cloudscraper.create_scraper()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    def _load_config_setting(self, setting_name: str, fallback_value: Optional[str] = None) -> Optional[str]:
        """Try to get a setting from environment variables or YAML config file"""
        # First check if it's in the environment
        env_value = os.getenv(setting_name)
        if env_value is None and os.path.exists('env.yaml'):
            # If not found, try loading from YAML file
            with open('env.yaml', 'r') as f:
                yaml_config = yaml.safe_load(f) or {}
                env_value = yaml_config.get(setting_name, fallback_value)
        return env_value if env_value is not None else fallback_value

    # Methods to build the proper URLs for each type of API call
    def create_species_search_url(self, params: SpeciesSearchParams) -> str:
        """search species"""
        safe_name = quote(params.scientific_name)
        extra_params = {}
        
        if params.like is not None:
            extra_params['like'] = str(params.like).lower()
        if params.marine_only is not None:
            extra_params['marine_only'] = str(params.marine_only).lower()
            
        query_part = urlencode(extra_params) if extra_params else ''
        final_url = f"{self.worms_api_base_url}/AphiaRecordsByName/{safe_name}"
        
        if query_part:
            return f"{final_url}?{query_part}"
        else:
            return final_url

    def create_synonyms_url(self, params: SynonymsParams) -> str:
        """synonyms list"""
        aphia_id = params.aphia_id
        return f"{self.worms_api_base_url}/AphiaSynonymsByAphiaID/{aphia_id}"

    def create_distribution_url(self, params: DistributionParams) -> str:
        """geographic data"""
        return f"{self.worms_api_base_url}/AphiaDistributionsByAphiaID/{params.aphia_id}"

    def create_vernacular_url(self, params: VernacularParams) -> str:
        """common names"""
        id_num = params.aphia_id
        url = f"{self.worms_api_base_url}/AphiaVernacularsByAphiaID/{id_num}"
        return url

    def create_sources_url(self, params: SourcesParams) -> str:
        """literature refs"""
        return f"{self.worms_api_base_url}/AphiaSourcesByAphiaID/{params.aphia_id}"

    def create_record_url(self, params: RecordParams) -> str:
        """basic record"""
        base_url = self.worms_api_base_url
        aphia_id = params.aphia_id
        return f"{base_url}/AphiaRecordByAphiaID/{aphia_id}"

    def create_classification_url(self, params: ClassificationParams) -> str:
        """taxonomic tree"""
        return f"{self.worms_api_base_url}/AphiaClassificationByAphiaID/{params.aphia_id}"

    def create_children_url(self, params: ChildrenParams) -> str:
        """child taxa"""
        url = f"{self.worms_api_base_url}/AphiaChildrenByAphiaID/{params.aphia_id}"
        return url

    # Actually make the HTTP requests and handle responses
    def make_api_call(self, url: str) -> Dict:
        """Make a GET request to the WoRMS API and return the JSON data"""
        try:
            api_response = self.session.get(url, timeout=60)
            api_response.raise_for_status()
            try:
                return api_response.json()
            except ValueError:
                raise ConnectionError(f"WoRMS API didn't return valid JSON. Got: {api_response.text[:200]}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to WoRMS API: {e}")

    # Convenience method to quickly get an AphiaID from a species name
    def lookup_species_id(self, scientific_name: str) -> Optional[int]:
        """Quick way to get the AphiaID for a species name"""
        search_params = SpeciesSearchParams(scientific_name=scientific_name)
        search_url = self.create_species_search_url(search_params)
        
        try:
            api_result = self.make_api_call(search_url)
            # Handle both single results and arrays
            if isinstance(api_result, list) and api_result:
                return api_result[0].get('AphiaID')
            elif isinstance(api_result, dict):
                return api_result.get('AphiaID')
            return None
        except Exception:
            return None