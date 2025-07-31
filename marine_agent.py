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
    SynonymSearchParams,
    DistributionSearchParams,
    VernacularSearchParams,
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
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Get synonyms and alternative scientific names for a marine species from WoRMS",
                    parameters=SynonymSearchParams
                ),
                AgentEntrypoint(
                    id="get_distribution",
                    description="Get geographic distribution and occurrence data for a marine species from WoRMS",
                    parameters=DistributionSearchParams
                ),
                AgentEntrypoint(
                    id="get_vernacular_names",
                    description="Get vernacular names and common names in different languages for a marine species from WoRMS",
                    parameters=VernacularSearchParams
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        if entrypoint == "get_marine_info":
            await self.get_marine_info(context, request)
        elif entrypoint == "get_synonyms":
            await self.get_synonyms(context, request)
        elif entrypoint == "get_distribution":
            await self.get_distribution(context, request)
        elif entrypoint == "get_vernacular_names":
            await self.get_vernacular(context, request)
        else:
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def get_synonyms(self, context: ResponseContext, request: str, params: Optional[SynonymSearchParams]):
        """Get synonyms for a marine species"""
        async with context.begin_process("Getting synonyms from WoRMS") as process:
            try:
                # Get species name from params or extract from request
                if params and params.species_name:
                    scientific_name = params.species_name
                    await process.log(f"Using species name from params: {scientific_name}")
                else:
                    scientific_name = await self.extract_species_name(request, process)
                    if not scientific_name:
                        await context.reply("Could not identify a marine species name from your request.")
                        return

                await process.log(f"Getting synonyms for: {scientific_name}")
                
                loop = asyncio.get_event_loop()
                synonyms_data = await loop.run_in_executor(None, lambda: self.get_synonyms_data(scientific_name))
                
                if not synonyms_data:
                    await context.reply(f"No synonyms found for '{scientific_name}' in WoRMS.")
                    return

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Synonyms for {scientific_name}",
                    uris=[f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{synonyms_data['aphia_id']}"],
                    metadata={"synonym_count": len(synonyms_data['synonyms'])}
                )
                
                # Simple response
                synonyms = synonyms_data['synonyms']
                response = f"**Synonyms for {scientific_name}**\n\n"
                for i, syn in enumerate(synonyms[:10], 1):
                    name = syn.get('scientificname', 'Unknown')
                    response += f"{i}. {name}\n"
                
                if len(synonyms) > 10:
                    response += f"\n... and {len(synonyms) - 10} more synonyms in the artifact."
                
                response += f"\n**Total**: {len(synonyms)} synonyms found."
                
                await context.reply(response)

            except Exception as e:
                await context.reply(f"Error getting synonyms: {str(e)}")

    async def get_distribution(self, context: ResponseContext, request: str, params: Optional[DistributionSearchParams]):
        """Get distribution for a marine species"""
        async with context.begin_process("Getting distribution from WoRMS") as process:
            try:
                # Get species name from params or extract from request
                if params and params.species_name:
                    scientific_name = params.species_name
                    await process.log(f"Using species name from params: {scientific_name}")
                else:
                    scientific_name = await self.extract_species_name(request, process)
                    if not scientific_name:
                        await context.reply("Could not identify a marine species name from your request.")
                        return

                await process.log(f"Getting distribution for: {scientific_name}")
                
                loop = asyncio.get_event_loop()
                distribution_data = await loop.run_in_executor(None, lambda: self.get_distribution_data(scientific_name))
                
                if not distribution_data:
                    await context.reply(f"No distribution data found for '{scientific_name}' in WoRMS.")
                    return

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Distribution for {scientific_name}",
                    uris=[f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{distribution_data['aphia_id']}"],
                    metadata={"distribution_count": len(distribution_data['distributions'])}
                )
                
                # Simple response
                distributions = distribution_data['distributions']
                response = f"**Distribution for {scientific_name}**\n\n"
                
                regions = []
                for dist in distributions[:10]:
                    if dist.get('locality'):
                        regions.append(dist['locality'])
                    elif dist.get('country'):
                        regions.append(dist['country'])
                
                if regions:
                    response += f"**Found in**: {', '.join(regions)}"
                    if len(distributions) > 10:
                        response += f" and {len(distributions) - 10} more locations"
                    response += "\n\n"
                
                response += f"**Total**: {len(distributions)} distribution records found."
                
                await context.reply(response)

            except Exception as e:
                await context.reply(f"Error getting distribution: {str(e)}")

    async def get_vernacular(self, context: ResponseContext, request: str, params: Optional[VernacularSearchParams]):
        """Get vernacular names for a marine species"""
        async with context.begin_process("Getting vernacular names from WoRMS") as process:
            try:
                # Get species name from params or extract from request
                if params and params.species_name:
                    scientific_name = params.species_name
                    await process.log(f"Using species name from params: {scientific_name}")
                else:
                    scientific_name = await self.extract_species_name(request, process)
                    if not scientific_name:
                        await context.reply("Could not identify a marine species name from your request.")
                        return

                await process.log(f"Getting vernacular names for: {scientific_name}")
                
                loop = asyncio.get_event_loop()
                vernacular_data = await loop.run_in_executor(None, lambda: self.get_vernacular_data(scientific_name))
                
                if not vernacular_data:
                    await context.reply(f"No vernacular names found for '{scientific_name}' in WoRMS.")
                    return

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Vernacular names for {scientific_name}",
                    uris=[f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{vernacular_data['aphia_id']}"],
                    metadata={"vernacular_count": len(vernacular_data['vernaculars'])}
                )
                
                # Simple response
                vernaculars = vernacular_data['vernaculars']
                response = f"**Common Names for {scientific_name}**\n\n"
                
                # Group by language
                by_language = {}
                for vern in vernaculars:
                    name = vern.get('vernacular', 'Unknown')
                    language = vern.get('language', 'Unknown language')
                    
                    if language not in by_language:
                        by_language[language] = []
                    by_language[language].append(name)
                
                # Show top languages
                for i, (language, names) in enumerate(list(by_language.items())[:5], 1):
                    response += f"**{language}**: {', '.join(names[:3])}"
                    if len(names) > 3:
                        response += f" (and {len(names) - 3} more)"
                    response += "\n"
                
                response += f"\n**Total**: {len(vernaculars)} names in {len(by_language)} languages."
                
                await context.reply(response)

            except Exception as e:
                await context.reply(f"Error getting vernacular names: {str(e)}")

    def get_distribution_data(self, scientific_name: str) -> dict:
        """Get distribution data for a species"""
        try:
            species_data = self.get_species_record(scientific_name)
            if not species_data:
                return None
            
            aphia_id = species_data['AphiaID']
            distributions = self.get_species_distributions(aphia_id)
            
            return {
                'aphia_id': aphia_id,
                'distributions': distributions or [],
                'species_name': species_data['scientificname']
            }
        except Exception:
            return None

    def get_vernacular_data(self, scientific_name: str) -> dict:
        """Get vernacular data for a species"""
        try:
            species_data = self.get_species_record(scientific_name)
            if not species_data:
                return None
            
            aphia_id = species_data['AphiaID']
            vernaculars = self.get_species_vernaculars(aphia_id)
            
            return {
                'aphia_id': aphia_id,
                'vernaculars': vernaculars or [],
                'species_name': species_data['scientificname']
            }
        except Exception:
            return None
        """Get synonyms data for a species"""
        try:
            species_data = self.get_species_record(scientific_name)
            if not species_data:
                return None
            
            aphia_id = species_data['AphiaID']
            synonyms = self.get_species_synonyms(aphia_id)
            
            return {
                'aphia_id': aphia_id,
                'synonyms': synonyms or [],
                'species_name': species_data['scientificname']
            }
        except Exception:
            return None

    async def get_marine_info(self, context: ResponseContext, request: str):
        async with context.begin_process(summary="Searching WoRMS for marine species") as process:
            try:
                # Get species name from params or extract from request
                if params and params.species_name:
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

                # Create artifact
                total = len(worms_data.get('synonyms', [])) + len(worms_data.get('vernaculars', [])) + len(worms_data.get('distributions', []))
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Raw JSON for marine species {species['scientificname']}.",
                    uris=[f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"],
                    metadata={"record_count": 1, "total_matches": total}
                )
                
                # Simple improved response
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
            if "delphinus delphis" in request.lower():
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