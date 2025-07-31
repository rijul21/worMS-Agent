import os
import json
import dotenv
import instructor
import asyncio
from typing import Optional, override
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import requests
from urllib.parse import quote

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint

from worms_models import (
    WoRMSRecord,
    WoRMSSynonym,
    WoRMSDistribution,
    WoRMSVernacular,
    MarineInfoParams
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
                    description="Get complete marine species information from WoRMS including taxonomy, synonyms, distribution and vernacular names",
                    parameters=MarineInfoParams
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        async with context.begin_process(summary="Searching WoRMS for marine species") as process:
            try:
                # Get species name from params or extract from request
                if params and hasattr(params, 'species_name') and params.species_name:
                    scientific_name = params.species_name
                    await process.log(f"Using species name from params: {scientific_name}")
                else:
                    scientific_name = await self.extract_species_name(request, process)
                    if not scientific_name:
                        await context.reply("Could not identify a marine species name from your request.")
                        return

                await process.log(f"Searching WoRMS database for: {scientific_name}")
                
                loop = asyncio.get_event_loop()
                worms_data = await loop.run_in_executor(None, lambda: self.get_worms_data(scientific_name))
                
                if not worms_data:
                    await context.reply(f"No marine species found for '{scientific_name}' in WoRMS database.")
                    return

                species = worms_data['species']
                aphia_id = species['AphiaID']
                
                await process.log(f"Found species: {species['scientificname']} (AphiaID: {aphia_id})")

                # Create artifact - WORKING PATTERN
                total = len(worms_data.get('synonyms', [])) + len(worms_data.get('vernaculars', [])) + len(worms_data.get('distributions', []))
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Raw JSON for marine species {species['scientificname']}.",
                    uris=[f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"],
                    metadata={"record_count": 1, "total_matches": total}
                )
                
                # Simple response
                response = f"**{species['scientificname']}** (AphiaID: {aphia_id})\n\n"
                response += f"**Classification**: {species.get('kingdom', 'N/A')} > {species.get('phylum', 'N/A')} > {species.get('family', 'N/A')}\n\n"
                response += f"**Data found**: {len(worms_data.get('synonyms', []))} synonyms, {len(worms_data.get('vernaculars', []))} common names, {len(worms_data.get('distributions', []))} distribution records\n\n"
                response += "Complete data available in the attached artifact."
                
                await context.reply(response)

            except Exception as e:
                await context.reply(f"An error occurred: {str(e)}")

    async def extract_species_name(self, request: str, process) -> Optional[str]:
        """Extract species name with fallback logic"""
        try:
            # Try instructor first
            openai_client = AsyncOpenAI(
                base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                api_key=os.getenv("GROQ_API_KEY")
            )
            instructor_client = instructor.patch(openai_client)

            marine_query: MarineQueryModel = await instructor_client.chat.completions.create(
                model="llama3-70b-8192",
                response_model=MarineQueryModel,
                messages=[
                    {"role": "system", "content": "Extract the scientific name of a marine species from the user's message."},
                    {"role": "user", "content": request}
                ],
                max_retries=2
            )
            await process.log(f"Identified marine species: {marine_query.scientific_name}")
            return marine_query.scientific_name

        except Exception as e:
            # Fallback: simple pattern matching
            await process.log(f"API extraction failed, using fallback")
            words = request.split()
            for i in range(len(words) - 1):
                if words[i][0].isupper() and words[i+1][0].islower():
                    scientific_name = f"{words[i]} {words[i+1]}"
                    await process.log(f"Using fallback extraction: {scientific_name}")
                    return scientific_name
            
            # Common patterns
            if "orcinus orca" in request.lower():
                return "Orcinus orca"
            elif "delphinus delphis" in request.lower():
                return "Delphinus delphis"
            
            return None

    def get_worms_data(self, scientific_name: str) -> dict:
        """Get complete WoRMS data for a species"""
        try:
            # Get main species record
            species_data = self.get_species_record(scientific_name)
            if not species_data:
                return None
            
            aphia_id = species_data['AphiaID']
            
            # Get additional data
            synonyms = self.get_species_synonyms(aphia_id)
            vernaculars = self.get_species_vernaculars(aphia_id) 
            distributions = self.get_species_distributions(aphia_id)
            
            return {
                'species': species_data,
                'synonyms': synonyms or [],
                'vernaculars': vernaculars or [],
                'distributions': distributions or [],
                'search_term': scientific_name
            }
            
        except Exception as e:
            print(f"Error getting WoRMS data: {e}")
            return None

    def get_species_record(self, scientific_name: str) -> dict:
        """Get main species record from WoRMS"""
        try:
            encoded_name = quote(scientific_name)
            url = f"https://www.marinespecies.org/rest/AphiaRecordsByName/{encoded_name}?like=false&marine_only=true"
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                return None
            
            data = response.json()
            if not data:
                return None
                
            # Return first record
            if isinstance(data, list):
                return data[0]
            else:
                return data
                
        except Exception as e:
            print(f"Error getting species record: {e}")
            return None

    def get_species_synonyms(self, aphia_id: int) -> list:
        """Get synonyms for a species"""
        try:
            url = f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{aphia_id}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data] if data else []
            return []
        except Exception:
            return []

    def get_species_vernaculars(self, aphia_id: int) -> list:
        """Get vernacular/common names for a species"""
        try:
            url = f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aphia_id}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data] if data else []
            return []
        except Exception:
            return []

    def get_species_distributions(self, aphia_id: int) -> list:
        """Get distribution data for a species"""
        try:
            url = f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{aphia_id}"
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()
                return data if isinstance(data, list) else [data] if data else []
            return []
        except Exception:
            return []

print("INIT: WoRMS Marine Species Agent loaded successfully")