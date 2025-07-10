from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List, Union
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
    unacceptreason: Optional[str] = None
    taxonRankID: Optional[int] = None
    rank: Optional[str] = None
    valid_AphiaID: Optional[int] = None
    valid_name: Optional[str] = None
    valid_authority: Optional[str] = None
    parentNameUsageID: Optional[int] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    citation: Optional[str] = None
    lsid: Optional[str] = None
    isMarine: Optional[bool] = None
    isBrackish: Optional[bool] = None
    isFreshwater: Optional[bool] = None
    isTerrestrial: Optional[bool] = None
    isExtinct: Optional[bool] = None
    match_type: Optional[str] = None
    modified: Optional[str] = None

# Synonym Model
class WoRMSSynonym(BaseModel):
    """Synonym record from WoRMS"""
    model_config = ConfigDict(populate_by_name=True)
    
    AphiaID: int
    url: Optional[str] = None
    scientificname: str
    authority: Optional[str] = None
    status: Optional[str] = None
    unacceptreason: Optional[str] = None
    taxonRankID: Optional[int] = None
    rank: Optional[str] = None
    valid_AphiaID: Optional[int] = None
    valid_name: Optional[str] = None
    valid_authority: Optional[str] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    citation: Optional[str] = None
    lsid: Optional[str] = None
    isMarine: Optional[bool] = None
    isBrackish: Optional[bool] = None
    isFreshwater: Optional[bool] = None
    isTerrestrial: Optional[bool] = None
    isExtinct: Optional[bool] = None
    match_type: Optional[str] = None
    modified: Optional[str] = None

# Distribution Model
class WoRMSDistribution(BaseModel):
    """Distribution record from WoRMS"""
    AphiaID: int
    locationID: Optional[str] = None
    locality: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    ocean: Optional[str] = None
    FAOarea: Optional[str] = None
    generalarea: Optional[str] = None
    specificarea: Optional[str] = None
    recordtype: Optional[str] = None
    startdate: Optional[str] = None
    enddate: Optional[str] = None
    datecreated: Optional[str] = None
    datemodified: Optional[str] = None
    qualitystatus: Optional[str] = None
    establishmentMeans: Optional[str] = None
    note: Optional[str] = None

# Vernacular Names Model
class WoRMSVernacular(BaseModel):
    """Vernacular (common) name record from WoRMS"""
    AphiaID: int
    vernacular: str
    language_code: Optional[str] = None
    language: Optional[str] = None
    country_code: Optional[str] = None
    country: Optional[str] = None

# Classification Model
class WoRMSClassification(BaseModel):
    """Taxonomic classification record from WoRMS"""
    model_config = ConfigDict(populate_by_name=True)
    
    AphiaID: int
    rank: str
    scientificname: str
    child: Optional['WoRMSClassification'] = None

# Update forward references for recursive model
WoRMSClassification.model_rebuild()

# Children Model (for taxa that have subtaxa)
class WoRMSChild(BaseModel):
    """Child taxon record from WoRMS"""
    model_config = ConfigDict(populate_by_name=True)
    
    AphiaID: int
    url: Optional[str] = None
    scientificname: str
    authority: Optional[str] = None
    status: Optional[str] = None
    unacceptreason: Optional[str] = None
    taxonRankID: Optional[int] = None
    rank: Optional[str] = None
    valid_AphiaID: Optional[int] = None
    valid_name: Optional[str] = None
    valid_authority: Optional[str] = None
    parentNameUsageID: Optional[int] = None
    kingdom: Optional[str] = None
    phylum: Optional[str] = None
    class_: Optional[str] = Field(None, alias="class")
    order: Optional[str] = None
    family: Optional[str] = None
    genus: Optional[str] = None
    citation: Optional[str] = None
    lsid: Optional[str] = None
    isMarine: Optional[bool] = None
    isBrackish: Optional[bool] = None
    isFreshwater: Optional[bool] = None
    isTerrestrial: Optional[bool] = None
    isExtinct: Optional[bool] = None
    match_type: Optional[str] = None
    modified: Optional[str] = None

# Complete Marine Species Data Model
class CompleteMarineSpeciesData(BaseModel):
    """Complete marine species data combining all WoRMS information"""
    species: WoRMSRecord
    synonyms: Optional[List[WoRMSSynonym]] = None
    distribution: Optional[List[WoRMSDistribution]] = None
    vernaculars: Optional[List[WoRMSVernacular]] = None
    classification: Optional[List[WoRMSClassification]] = None
    children: Optional[List[WoRMSChild]] = None
    
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
    include_classification: bool = Field(
        True, 
        description="Include taxonomic classification in the response"
    )
    include_children: bool = Field(
        True, 
        description="Include child taxa in the response"
    )