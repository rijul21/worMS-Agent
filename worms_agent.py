from pydantic import BaseModel, Fieldimport ,asyncio
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from worms_client import (
    WoRMS,
    SpeciesSearchParams,
    AttributesParams,
    SynonymsParams, 
    DistributionParams,
    NoParams
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
                    
                    await context.reply(f"Found {attribute_count} attributes for {params.species_name} (AphiaID: {aphia_id}). I've created an artifact with the results.")
                else:
                    await context.reply(f"No attributes found for {params.species_name} in WoRMS.")

            except ConnectionError as e:
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
                    
                    await context.reply(f"Found {synonym_count} synonyms for {params.species_name} (AphiaID: {aphia_id}). I've created an artifact with the results.")
                else:
                    await context.reply(f"No synonyms found for {params.species_name} in WoRMS.")

            except ConnectionError as e:
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
                    
                    await context.reply(f"Found distribution data for {params.species_name} (AphiaID: {aphia_id}) across {distribution_count} locations. I've created an artifact with the results.")
                else:
                    await context.reply(f"No distribution data found for {params.species_name} in WoRMS.")

            except ConnectionError as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving distribution data: {e}")