import os
import json
import dotenv
import instructor
from typing import Optional, List, Dict, Any, override
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
import httpx
from urllib.parse import quote

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

from worms_models import (
    WoRMSRecord,
    WoRMSSynonym,
    WoRMSDistribution,
    WoRMSVernacular
)

dotenv.load_dotenv()

class MarineQueryModel(BaseModel):
    """Extracted marine species information from user message"""
    scientific_name: str = Field(..., description="Scientific name of the marine species, e.g. 'Orcinus orca'")

class MarineAgent(IChatBioAgent):
    def __init__(self):
        self.agent_card = AgentCard(
            name="WoRMS Marine Species Agent",
            description="Retrieves detailed marine species information from WoRMS (World Register of Marine Species) database.",
            icon=None,
            url="http://18.222.189.40:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="get_marine_info",
                    description="Returns detailed marine species information",
                    parameters=None
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        print(f"DEBUG: Agent run called with request: {request[:100]}...")
        print(f"DEBUG: Entrypoint: {entrypoint}")
        
        async with context.begin_process(summary="Analyzing marine species request") as process:
            try:
                print("DEBUG: Starting instructor extraction...")
                
                # Extract marine species information using instructor
                openai_client = AsyncOpenAI(
                    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                    api_key=os.getenv("GROQ_API_KEY")
                )
                instructor_client = instructor.patch(openai_client)

                print("DEBUG: Calling instructor...")
                
                marine_query: MarineQueryModel = await instructor_client.chat.completions.create(
                    model="llama3-70b-8192",
                    response_model=MarineQueryModel,
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            You are a marine biology expert that extracts species names from user messages.
                            Instructions:
                            1. Identify the scientific name of the marine species from the user's message
                            2. If common name is given, try to identify the scientific name
                            3. Always respond in this strict JSON format:

                            ```json
                            {
                            "scientific_name": "Genus species"
                            }```
                            """
                        },
                        {"role": "user", "content": request}
                    ],
                    max_retries=3
                )

                print(f"DEBUG: Instructor result: {marine_query.scientific_name}")

                await process.log(f"Identified marine species: {marine_query.scientific_name}", {
                    "scientific_name": marine_query.scientific_name,
                })

                # Search WoRMS database
                await process.log(f"Searching WoRMS database for: {marine_query.scientific_name}")
                print(f"DEBUG: Starting WoRMS search for: {marine_query.scientific_name}")

                async with httpx.AsyncClient(timeout=30.0) as client:
                    # Get species record
                    print("DEBUG: Calling _get_species_by_name...")
                    species_data = await self._get_species_by_name(client, marine_query.scientific_name)
                    
                    if not species_data:
                        print("DEBUG: No species data found")
                        await context.reply(f"No marine species found matching '{marine_query.scientific_name}' in WoRMS database.")
                        return

                    species = species_data[0]  # Use first result
                    aphia_id = species.AphiaID
                    print(f"DEBUG: Found species: {species.scientificname} (AphiaID: {aphia_id})")

                    await process.log(f"Found species: {species.scientificname} (AphiaID: {aphia_id})")

                    # Get additional data
                    await process.log("Retrieving synonyms, vernacular names, and distribution data")
                    print("DEBUG: Getting additional data...")
                    
                    synonyms = await self._get_synonyms(client, aphia_id)
                    print(f"DEBUG: Got {len(synonyms or [])} synonyms")
                    
                    vernaculars = await self._get_vernaculars(client, aphia_id)
                    print(f"DEBUG: Got {len(vernaculars or [])} vernaculars")
                    
                    distributions = await self._get_distributions(client, aphia_id)
                    print(f"DEBUG: Got {len(distributions or [])} distributions")

                    # Compile complete data
                    print("DEBUG: Compiling data...")
                    complete_data = {
                        "species": species.model_dump(),
                        "aphia_id": aphia_id,
                        "scientific_name": species.scientificname,
                        "search_term": marine_query.scientific_name,
                        "synonyms": [s.model_dump() for s in synonyms] if synonyms else [],
                        "vernacular_names": [v.model_dump() for v in vernaculars] if vernaculars else [],
                        "distributions": [d.model_dump() for d in distributions] if distributions else []
                    }
                    print("DEBUG: Data compiled successfully")

                    await process.log(f"Data compilation complete. Found {len(synonyms or [])} synonyms, {len(vernaculars or [])} vernacular names, {len(distributions or [])} distribution records.")

                    # Create artifact - try with smaller data first
                    await process.log("Starting artifact creation...")
                    print("DEBUG: Starting artifact creation...")
                    
                    # Create a smaller test dataset first
                    smaller_data = {
                        "species": {
                            "AphiaID": species.AphiaID,
                            "scientificname": species.scientificname,
                            "kingdom": species.kingdom,
                            "phylum": species.phylum,
                            "family": species.family
                        },
                        "aphia_id": aphia_id,
                        "scientific_name": species.scientificname,
                        "search_term": marine_query.scientific_name,
                        "counts": {
                            "synonyms": len(synonyms or []),
                            "vernacular_names": len(vernaculars or []),
                            "distributions": len(distributions or [])
                        }
                    }
                    
                    content = json.dumps(smaller_data, indent=2)
                    await process.log(f"JSON content created, size: {len(content)} characters")
                    print(f"DEBUG: JSON content created, size: {len(content)} characters")
                    
                    content_bytes = content.encode('utf-8')
                    await process.log(f"Content encoded to bytes, size: {len(content_bytes)} bytes")
                    print(f"DEBUG: Content encoded to bytes, size: {len(content_bytes)} bytes")

                    print("DEBUG: About to call create_artifact...")
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species summary for {species.scientificname}",
                        content=content_bytes,
                        uris=[f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"],
                        metadata={
                            "data_source": "WoRMS",
                            "aphia_id": aphia_id,
                            "scientific_name": species.scientificname
                        }
                    )
                    print("DEBUG: create_artifact call completed!")
                    
                    await process.log("Artifact creation completed successfully!")
                    print("DEBUG: Artifact creation completed!")

                    # Create user response
                    response_parts = [
                        f"Found {species.scientificname} (AphiaID: {aphia_id}) in WoRMS database.",
                        f"Classification: {species.kingdom} > {species.phylum} > {getattr(species, 'class_', 'N/A')} > {species.family}."
                    ]
                    
                    if synonyms:
                        response_parts.append(f"Found {len(synonyms)} synonyms.")
                    if vernaculars:
                        response_parts.append(f"Found {len(vernaculars)} vernacular names.")
                    if distributions:
                        response_parts.append(f"Found {len(distributions)} distribution records.")
                    
                    response_parts.append("Complete marine species data has been compiled in the artifact.")
                    
                    print("DEBUG: Sending reply...")
                    await context.reply(" ".join(response_parts))
                    print("DEBUG: Reply sent successfully!")

            except InstructorRetryException as e:
                print(f"DEBUG: InstructorRetryException: {e}")
                await context.reply("Sorry, I couldn't extract marine species information from your request.", data={"error": str(e)})
            except Exception as e:
                print(f"DEBUG: General Exception: {e}")
                print(f"DEBUG: Exception type: {type(e)}")
                import traceback
                print(f"DEBUG: Traceback: {traceback.format_exc()}")
                await context.reply("An error occurred while retrieving marine species information", data={"error": str(e)})

    async def _get_species_by_name(self, client: httpx.AsyncClient, scientific_name: str):
        """Get species data from WoRMS"""
        try:
            encoded_name = quote(scientific_name)
            url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
            
            response = await client.get(url)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
            
            # Handle single record or list
            if isinstance(data, list):
                records = []
                for record in data:
                    # Fix boolean fields
                    for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                        if bool_field in record and record[bool_field] is not None:
                            record[bool_field] = bool(record[bool_field])
                    records.append(WoRMSRecord(**record))
                return records
            else:
                # Single record
                for bool_field in ['isMarine', 'isBrackish', 'isFreshwater', 'isTerrestrial', 'isExtinct']:
                    if bool_field in data and data[bool_field] is not None:
                        data[bool_field] = bool(data[bool_field])
                return [WoRMSRecord(**data)]
                
        except Exception as e:
            print(f"Error getting species: {e}")
            return None

    async def _get_synonyms(self, client: httpx.AsyncClient, aphia_id: int):
        """Get synonyms from WoRMS"""
        try:
            url = f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{aphia_id}"
            response = await client.get(url)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
            
            if isinstance(data, list):
                return [WoRMSSynonym(**record) for record in data]
            else:
                return [WoRMSSynonym(**data)]
        except Exception:
            return None

    async def _get_vernaculars(self, client: httpx.AsyncClient, aphia_id: int):
        """Get vernacular names from WoRMS"""
        try:
            url = f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aphia_id}"
            response = await client.get(url)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
            
            if isinstance(data, list):
                return [WoRMSVernacular(**record) for record in data]
            else:
                return [WoRMSVernacular(**data)]
        except Exception:
            return None

    async def _get_distributions(self, client: httpx.AsyncClient, aphia_id: int):
        """Get distribution data from WoRMS"""
        try:
            url = f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{aphia_id}"
            response = await client.get(url)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
            
            if isinstance(data, list):
                return [WoRMSDistribution(**record) for record in data]
            else:
                return [WoRMSDistribution(**data)]
        except Exception:
            return None

print("INIT: WoRMS Marine Species Agent loaded successfully")