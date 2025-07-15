from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from datetime import datetime

# Core WoRMS Record Model
class WoRMSRecord(BaseModel):
    """Main species record from WoRMS"""
    model_config = ConfigDict(populate_by_name=True)
    
    AphiaID: int
    url: Optional[str] = None
    scientificname: str
    authority: Optional[str] = None
    status: Optional[str] = None
    rank: Optional[str] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    isMarine: Optional[bool] = None

# Synonym Model
class WoRMSSynonym(BaseModel):
    """Synonym record from WoRMS"""
    model_config = ConfigDict(populate_by_name=True)
    
    AphiaID: int
    scientificname: str
    authority: Optional[str] = None

# Distribution Model
class WoRMSDistribution(BaseModel):
    """Distribution record from WoRMS"""
    AphiaID: int
    locality: Optional[str] = None
    country: Optional[str] = None
    geographical_scale: Optional[str] = Field(None, alias="geographicalScale")

# Vernacular Names Model
class WoRMSVernacular(BaseModel):
    """Vernacular (common) name record from WoRMS"""
    AphiaID: int
    vernacular: str
    language: Optional[str] = None

# Source Model
class WoRMSSource(BaseModel):
    """Source (reference) record from WoRMS"""
    reference: Optional[str] = None
    author: Optional[str] = None
    year: Optional[str] = None

# Complete Marine Species Data Model
class CompleteMarineSpeciesData(BaseModel):
    """Complete marine species data combining all WoRMS information"""
    species: WoRMSRecord
    synonyms: Optional[List[WoRMSSynonym]] = None
    distribution: Optional[List[WoRMSDistribution]] = None
    vernaculars: Optional[List[WoRMSVernacular]] = None
    sources: Optional[List[WoRMSSource]] = None
    attributes: Optional[List[dict]] = None
    
    # Metadata
    aphia_id: int
    scientific_name: str
    search_term: str
    retrieved_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('aphia_id', mode='before')
    @classmethod
    def set_aphia_id(cls, v, info):
        if hasattr(info, 'data') and 'species' in info.data and info.data['species']:
            return info.data['species'].AphiaID
        return v
    
    @field_validator('scientific_name', mode='before')
    @classmethod
    def set_scientific_name(cls, v, info):
        if hasattr(info, 'data') and 'species' in info.data and info.data['species']:
            return info.data['species'].scientificname
        return v

# Query Models for the Agent
class MarineQueryModel(BaseModel):
    """Model for extracting marine species information from user queries"""
    scientificname: Optional[str] = Field(
        None, 
        description="Scientific binomial name of the marine species (e.g., 'Orcinus orca', 'Carcharodon carcharias', 'Balaenoptera musculus')"
    )
    common_name: Optional[str] = Field(
        None, 
        description="Common or vernacular name of the marine animal in English (e.g., 'killer whale', 'great white shark', 'blue whale')"
    )

class MarineParameters(BaseModel):
    """Parameters for marine species queries"""
    species_name: Optional[str] = Field(
        None, 
        description="Name of the marine species (scientific or common name)"
    )
    include_synonyms: bool = Field(
        True, 
        description="Include synonyms in the response"
    )
    include_distribution: bool = Field(
        True, 
        description="Include distribution data in the response"
    )
    include_vernaculars: bool = Field(
        True, 
        description="Include vernacular names in the response"
    )
    include_sources: bool = Field(
        True, 
        description="Include source references in the response"
    )