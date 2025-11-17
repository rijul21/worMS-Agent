import os
import yaml
import requests
from pydantic import BaseModel, Field
from typing import Optional, Dict
from urllib.parse import urlencode, quote
import cloudscraper


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

class ExternalIDParams(BaseModel):
    """Parameters for getting external database IDs"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get external IDs for",
        examples=[137205, 104625, 137094]
    )
    id_type: Optional[str] = Field(None,
        description="Specific database type (e.g., 'fishbase', 'ncbi', 'tsn', 'bold', 'gisd')",
        examples=["fishbase", "ncbi", "tsn"]
    )

class AttributesParams(BaseModel):
    """Parameters for getting species attributes/traits"""
    aphia_id: int = Field(...,
        description="The AphiaID of the species to get attributes for",
        examples=[137205, 104625, 137094]
    )

class VernacularSearchParams(BaseModel):
    """Parameters for searching species by common/vernacular name"""
    vernacular_name: str = Field(...,
        description="Common name to search for",
        examples=["killer whale", "great white shark", "bottlenose dolphin"]
    )
    like: Optional[bool] = Field(False,
        description="Use fuzzy matching for names"
    )

class MatchNamesParams(BaseModel):
    """Parameters for batch matching multiple species names"""
    scientific_names: list[str] = Field(...,
        description="List of scientific names to match (max 50)",
        examples=[["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]],
        max_length=50
    )
    authorships: Optional[list[str]] = Field(None,
        description="Optional list of authorities for each name"
    )
    marine_only: Optional[bool] = Field(True,
        description="Limit to marine taxa"
    )
    extant_only: Optional[bool] = Field(True,
        description="Limit to extant (non-extinct) taxa"
    )
    match_authority: Optional[bool] = Field(True,
        description="Use the authority in the matching process"
    )


class AttributeKeysParams(BaseModel):
    """Parameters for getting attribute definition tree"""
    attribute_id: int = Field(0,
        description="The attribute definition ID to search for (0 for root items)",
        examples=[0, 1, 7]
    )
    include_children: Optional[bool] = Field(True,
        description="Include the tree of children"
    )


class AttributeValuesByCategoryParams(BaseModel):
    """Parameters for getting attribute values grouped by category"""
    category_id: int = Field(...,
        description="The CategoryID to search for",
        examples=[1, 7, 9]
    )

class RecordsByDateParams(BaseModel):
    """Parameters for getting species records modified during a time period"""
    startdate: str = Field(...,
        description="ISO 8601 formatted start date(time) - e.g., 2024-01-01T00:00:00+00:00",
        examples=["2024-01-01T00:00:00+00:00", "2025-11-08T00:00:00+00:00"]
    )
    enddate: Optional[str] = Field(None,
        description="ISO 8601 formatted end date(time) - defaults to today if not provided"
    )
    marine_only: Optional[bool] = Field(True,
        description="Limit to marine taxa"
    )
    extant_only: Optional[bool] = Field(True,
        description="Limit to extant (non-extinct) taxa"
    )
    offset: Optional[int] = Field(1,
        description="Starting record number for pagination (default: 1)"
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
    
    def build_external_id_url(self, params: ExternalIDParams) -> str:
        """Build URL for getting external database IDs"""
        base_url = f"{self.worms_api_base_url}/AphiaExternalIDByAphiaID/{params.aphia_id}"
        if params.id_type:
            return f"{base_url}?type={params.id_type}"
        return base_url
    
    def build_attributes_url(self, params: AttributesParams) -> str:
        """Build URL for getting species attributes/traits"""
        return f"{self.worms_api_base_url}/AphiaAttributesByAphiaID/{params.aphia_id}"
    
    def build_vernacular_search_url(self, params: VernacularSearchParams) -> str:
        """Build URL for searching species by vernacular/common name"""
        encoded_name = quote(params.vernacular_name)
        query_params = {}
        
        if params.like is not None:
            query_params['like'] = str(params.like).lower()
            
        query_string = urlencode(query_params) if query_params else ''
        base_url = f"{self.worms_api_base_url}/AphiaRecordsByVernacular/{encoded_name}"
        return f"{base_url}?{query_string}" if query_string else base_url
    
    def build_match_names_url(self, params: MatchNamesParams) -> str:
        """Build URL for batch matching multiple species names using TAXAMATCH fuzzy matching"""
        query_params = []
        
        for name in params.scientific_names:
            query_params.append(('scientificnames[]', name))
        

        if params.authorships:
            for auth in params.authorships:
                query_params.append(('authorships[]', auth))
        
    
        if params.marine_only is not None:
            query_params.append(('marine_only', str(params.marine_only).lower()))
        if params.extant_only is not None:
            query_params.append(('extant_only', str(params.extant_only).lower()))
        if params.match_authority is not None:
            query_params.append(('match_authority', str(params.match_authority).lower()))
        
      
        query_string = '&'.join([f"{k}={quote(str(v))}" for k, v in query_params])
        
        return f"{self.worms_api_base_url}/AphiaRecordsByMatchNames?{query_string}"
    
 
    def build_attribute_keys_url(self, params: AttributeKeysParams) -> str:
        """Build URL for getting attribute definition tree"""
        query_params = {}
        if params.include_children is not None:
            query_params['include_children'] = str(params.include_children).lower()
        
        query_string = urlencode(query_params) if query_params else ''
        base_url = f"{self.worms_api_base_url}/AphiaAttributeKeysByID/{params.attribute_id}"
        return f"{base_url}?{query_string}" if query_string else base_url
    
 
    def build_attribute_values_by_category_url(self, params: AttributeValuesByCategoryParams) -> str:
        """Build URL for getting attribute values grouped by category"""
        return f"{self.worms_api_base_url}/AphiaAttributeValuesByCategoryID/{params.category_id}"
    
  
    def build_records_by_date_url(self, params: RecordsByDateParams) -> str:
        """Build URL for getting records modified during a specific time period"""
        query_params = {'startdate': params.startdate}
        
        if params.enddate:
            query_params['enddate'] = params.enddate
        if params.marine_only is not None:
            query_params['marine_only'] = str(params.marine_only).lower()
        if params.extant_only is not None:
            query_params['extant_only'] = str(params.extant_only).lower()
        if params.offset is not None:
            query_params['offset'] = str(params.offset)
        
        query_string = urlencode(query_params)
        return f"{self.worms_api_base_url}/AphiaRecordsByDate?{query_string}"


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
