from pydantic import BaseModel, Field
import asyncio
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from worms_client import (
    WoRMS,

    AttributesParams,
    SynonymsParams, 
    DistributionParams,

)

# Simple Agent Parameter Models
class MarineAttributesParams(BaseModel):
    """Parameters for getting marine species attributes"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )

class MarineSynonymsParams(BaseModel):
    """Parameters for getting marine species synonyms"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )

class MarineDistributionParams(BaseModel):
    """Parameters for getting marine species distribution"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )

class WoRMSiChatBioAgent:
    """The iChatBio agent implementation for WoRMS - Simple 3 endpoint version"""

    def __init__(self):
        self.worms_logic = WoRMS()

    async def run_get_attributes(self, context, params: MarineAttributesParams):
        """Workflow for getting marine species attributes"""
        async with context.begin_process(f"Getting attributes for '{params.species_name}'") as process:
            await process.log("Attributes search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get attributes
                attr_params = AttributesParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_attributes_url(attr_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    attributes = raw_response
                elif isinstance(raw_response, dict):
                    attributes = [raw_response]
                else:
                    attributes = []

                attribute_count = len(attributes)
                
                if attribute_count > 0:
                    await process.log(f"Found {attribute_count} attributes")
                    
                    # Extract sample attributes for display
                    sample_attributes = []
                    for attr in attributes[:5]:  # Show first 5
                        if isinstance(attr, dict):
                            attr_name = attr.get('measurementType', attr.get('attribute', 'Unknown'))
                            attr_value = attr.get('measurementValue', attr.get('value', 'N/A'))
                            sample_attributes.append(f"{attr_name}: {attr_value}")
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species attributes for {params.species_name} (AphiaID: {aphia_id}) - {attribute_count} attributes",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Attributes",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "attribute_count": attribute_count
                        }
                    )
                    
                    # Create detailed reply
                    reply = f"Found {attribute_count} attributes for {params.species_name} (AphiaID: {aphia_id})"
                    if sample_attributes:
                        reply += f". Examples: {'; '.join(sample_attributes[:3])}"
                        if attribute_count > 3:
                            reply += f" and {attribute_count - 3} more"
                    reply += ". I've created an artifact with the complete data."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No attributes found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving attributes: {e}")

    async def run_get_synonyms(self, context, params: MarineSynonymsParams):
        """Workflow for getting marine species synonyms"""
        async with context.begin_process(f"Getting synonyms for '{params.species_name}'") as process:
            await process.log("Synonyms search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get synonyms
                syn_params = SynonymsParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_synonyms_url(syn_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    synonyms = raw_response
                elif isinstance(raw_response, dict):
                    synonyms = [raw_response]
                else:
                    synonyms = []

                synonym_count = len(synonyms)
                
                if synonym_count > 0:
                    await process.log(f"Found {synonym_count} synonyms")
                    
                    # Extract sample synonyms for display
                    sample_synonyms = []
                    for syn in synonyms[:8]:  # Show first 8
                        if isinstance(syn, dict):
                            syn_name = syn.get('scientificname', 'Unknown')
                            syn_status = syn.get('status', '')
                            if syn_status and syn_status != 'accepted':
                                sample_synonyms.append(f"{syn_name} ({syn_status})")
                            else:
                                sample_synonyms.append(syn_name)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species synonyms for {params.species_name} (AphiaID: {aphia_id}) - {synonym_count} synonyms",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Synonyms",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "synonym_count": synonym_count
                        }
                    )
                    
                    # Create detailed reply
                    reply = f"Found {synonym_count} synonyms for {params.species_name} (AphiaID: {aphia_id})"
                    if sample_synonyms:
                        reply += f". Examples: {', '.join(sample_synonyms[:5])}"
                        if synonym_count > 5:
                            reply += f" and {synonym_count - 5} more"
                    reply += ". I've created an artifact with all the synonyms."
                    
                    await context.reply(reply)
                else:
                    await context.reply(f"No synonyms found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving synonyms: {e}")

    async def run_get_distribution(self, context, params: MarineDistributionParams):
        """Workflow for getting marine species distribution"""
        async with context.begin_process(f"Getting distribution for '{params.species_name}'") as process:
            await process.log("Distribution search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get distribution
                dist_params = DistributionParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_distribution_url(dist_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    distributions = raw_response
                elif isinstance(raw_response, dict):
                    distributions = [raw_response]
                else:
                    distributions = []

                distribution_count = len(distributions)
                
                if distribution_count > 0:
                    await process.log(f"Found {distribution_count} distribution records")
                    
                    # Extract location details for user-friendly response
                    countries = set()
                    localities = []
                    
                    for dist in distributions:
                        if isinstance(dist, dict):
                            if dist.get('country'):
                                countries.add(dist['country'])
                            if dist.get('locality'):
                                localities.append(dist['locality'])
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species distribution for {params.species_name} (AphiaID: {aphia_id}) - {distribution_count} locations",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Distribution",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "distribution_count": distribution_count
                        }
                    )
                    
                    # Create detailed user-friendly response
                    reply_parts = [f"Found distribution data for {params.species_name} (AphiaID: {aphia_id}) across {distribution_count} locations"]
                    
                    if countries:
                        country_list = sorted(list(countries))
                        if len(country_list) <= 10:
                            reply_parts.append(f"Countries/regions: {', '.join(country_list)}")
                        else:
                            reply_parts.append(f"Countries/regions: {', '.join(country_list[:10])} and {len(country_list)-10} more")
                    
                    if localities:
                        if len(localities) <= 5:
                            reply_parts.append(f"Specific locations include: {', '.join(localities)}")
                        else:
                            reply_parts.append(f"Specific locations include: {', '.join(localities[:5])} and {len(localities)-5} more")
                    
                    reply_parts.append("I've created an artifact with the complete distribution data.")
                    
                    await context.reply(". ".join(reply_parts))
                else:
                    await context.reply(f"No distribution data found for {params.species_name} in WoRMS.")

            except ConnectionError as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving distribution data: {e}")