from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime

class WoRMSRecord(BaseModel):
    """Main WoRMS species record"""
    AphiaID: int
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

class WoRMSSynonym(BaseModel):
    """WoRMS synonym record"""
    AphiaID: int
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

class WoRMSVernacular(BaseModel):
    """WoRMS vernacular/common name record"""
    vernacular: str
    language_code: Optional[str] = None
    language: Optional[str] = None

class WoRMSDistribution(BaseModel):
    """WoRMS distribution record"""
    locationID: Optional[str] = None
    locality: Optional[str] = None
    country: Optional[str] = None
    country_code: Optional[str] = None
    establishmentMeans: Optional[str] = None
    locationRemarks: Optional[str] = None
    recordType: Optional[str] = None

class WoRMSSource(BaseModel):
    """WoRMS source/reference record"""
    source_id: Optional[int] = None
    reference: Optional[str] = None
    page: Optional[str] = None
    url: Optional[str] = None
    type: Optional[str] = None
    fulltext: Optional[str] = None
    doi: Optional[str] = None
    author: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    source: Optional[str] = None

class MarineQueryModel(BaseModel):
    """Model for extracting species names using instructor"""
    scientificname: Optional[str] = None
    common_name: Optional[str] = None

class MarineParameters(BaseModel):
    """Parameters for marine agent entrypoints"""
    species_name: Optional[str] = None
    include_synonyms: bool = False
    include_vernaculars: bool = False
    include_distribution: bool = False
    include_sources: bool = False

class CompleteMarineSpeciesData(BaseModel):
    """Complete marine species data structure for artifacts"""
    species: WoRMSRecord
    aphia_id: int
    scientific_name: str
    search_term: str
    synonyms: Optional[List[WoRMSSynonym]] = None
    distribution: Optional[List[WoRMSDistribution]] = None
    vernaculars: Optional[List[WoRMSVernacular]] = None
    sources: Optional[List[WoRMSSource]] = None
    attributes: Optional[List[Any]] = None