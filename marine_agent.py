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
    MarineInfoParams,
    CompleteMarineSpeciesData
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
                    id="get_species_info",
                    description="Get complete marine species information including basic data and synonyms",
                    parameters=MarineInfoParams
                ),
                AgentEntrypoint(
                    id="get_distribution",
                    description="Get distribution data for marine species", 
                    parameters=MarineInfoParams
                ),
                AgentEntrypoint(
                    id="get_common_names",
                    description="Get vernacular/common names for marine species",
                    parameters=MarineInfoParams
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        if entrypoint == "get_species_info":
            await self.get_species_info(context, request, params)
        elif entrypoint == "get_distribution":
            await self.get_distribution(context, request, params)
        elif entrypoint == "get_common_names":
            await self.get_common_names(context, request, params)
        else:
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def get_species_info(self, context: ResponseContext, request: str, params: Optional[MarineInfoParams]):
        """Get complete species info including basic data and synonyms"""
        async with context.begin_process(summary="Getting marine species information from WoRMS") as process:
            try:
                # Extract species name
                species_name = await self._extract_species_name(request, params)
                if not species_name:
                    await context.reply("Could not identify a marine species in your request.")
                    return

                await process.log(f"Searching for: {species_name}")

                # Get species data
                loop = asyncio.get_event_loop()
                species_data = await loop.run_in_executor(None, lambda: self.get_species_record(species_name))
                
                if not species_data:
                    await context.reply(f"No marine species found for '{species_name}' in WoRMS database.")
                    return

                aphia_id = species_data['AphiaID']
                await process.log(f"Found: {species_data['scientificname']} (ID: {aphia_id})")

                # Get synonyms
                synonyms = await loop.run_in_executor(None, lambda: self.get_species_synonyms(aphia_id))
                
                # Prepare complete data
                complete_data = {
                    'species': species_data,
                    'aphia_id': aphia_id,
                    'scientific_name': species_data['scientificname'],
                    'search_term': species_name,
                    'synonyms': synonyms or [],
                    'total_synonyms': len(synonyms or [])
                }

                # Create artifact with the actual data
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Complete marine species data for {species_data['scientificname']}",
                    content=json.dumps(complete_data, indent=2),
                    uris=[f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"],
                    metadata={"aphia_id": aphia_id, "synonym_count": len(synonyms or [])}
                )

                # Send summary to ichatbio
                summary = self._create_species_summary(species_data, synonyms)
                await context.reply(summary)

            except Exception as e:
                await context.reply(f"Error retrieving species information: {str(e)}")

    async def get_distribution(self, context: ResponseContext, request: str, params: Optional[MarineInfoParams]):
        """Get distribution data for marine species"""
        async with context.begin_process(summary="Getting distribution data from WoRMS") as process:
            try:
                species_name = await self._extract_species_name(request, params)
                if not species_name:
                    await context.reply("Could not identify a marine species in your request.")
                    return

                await process.log(f"Getting distribution for: {species_name}")

                loop = asyncio.get_event_loop()
                species_data = await loop.run_in_executor(None, lambda: self.get_species_record(species_name))
                
                if not species_data:
                    await context.reply(f"No marine species found for '{species_name}'.")
                    return

                aphia_id = species_data['AphiaID']
                distributions = await loop.run_in_executor(None, lambda: self.get_species_distributions(aphia_id))

                if not distributions:
                    await context.reply(f"No distribution data found for {species_data['scientificname']}.")
                    return

                # Prepare distribution data
                distribution_data = {
                    'species': species_data,
                    'aphia_id': aphia_id,
                    'scientific_name': species_data['scientificname'],
                    'distributions': distributions,
                    'total_locations': len(distributions)
                }

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Distribution data for {species_data['scientificname']}",
                    content=json.dumps(distribution_data, indent=2),
                    uris=[f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{aphia_id}"],
                    metadata={"aphia_id": aphia_id, "location_count": len(distributions)}
                )

                # Send distribution summary
                summary = self._create_distribution_summary(species_data, distributions)
                await context.reply(summary)

            except Exception as e:
                await context.reply(f"Error getting distribution data: {str(e)}")

    async def get_common_names(self, context: ResponseContext, request: str, params: Optional[MarineInfoParams]):
        """Get vernacular/common names for marine species"""
        async with context.begin_process(summary="Getting common names from WoRMS") as process:
            try:
                species_name = await self._extract_species_name(request, params)
                if not species_name:
                    await context.reply("Could not identify a marine species in your request.")
                    return

                await process.log(f"Getting common names for: {species_name}")

                loop = asyncio.get_event_loop()
                species_data = await loop.run_in_executor(None, lambda: self.get_species_record(species_name))
                
                if not species_data:
                    await context.reply(f"No marine species found for '{species_name}'.")
                    return

                aphia_id = species_data['AphiaID']
                vernaculars = await loop.run_in_executor(None, lambda: self.get_species_vernaculars(aphia_id))

                if not vernaculars:
                    await context.reply(f"No common names found for {species_data['scientificname']}.")
                    return

                # Prepare vernacular data
                vernacular_data = {
                    'species': species_data,
                    'aphia_id': aphia_id,
                    'scientific_name': species_data['scientificname'],
                    'vernaculars': vernaculars,
                    'total_names': len(vernaculars)
                }

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Common names for {species_data['scientificname']}",
                    content=json.dumps(vernacular_data, indent=2),
                    uris=[f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{aphia_id}"],
                    metadata={"aphia_id": aphia_id, "name_count": len(vernaculars)}
                )

                # Send common names summary
                summary = self._create_vernacular_summary(species_data, vernaculars)
                await context.reply(summary)

            except Exception as e:
                await context.reply(f"Error getting common names: {str(e)}")

    async def _extract_species_name(self, request: str, params: Optional[MarineInfoParams]) -> Optional[str]:
        """Extract species name from params or request"""
        # First try params if provided
        if params and hasattr(params, 'species_name') and params.species_name:
            return params.species_name

        # Fall back to LLM extraction
        try:
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
            return marine_query.scientific_name
        except:
            return None

    def _create_species_summary(self, species_data: dict, synonyms: list) -> str:
        """Create a summary of species information for ichatbio"""
        scientific_name = species_data['scientificname']
        authority = species_data.get('authority', '')
        status = species_data.get('status', 'Unknown')
        
        summary = f"**{scientific_name}** {authority}\n\n"
        summary += f"**Status:** {status}\n"
        summary += f"**AphiaID:** {species_data['AphiaID']}\n"
        
        if species_data.get('kingdom'):
            summary += f"**Kingdom:** {species_data['kingdom']}\n"
        if species_data.get('phylum'):
            summary += f"**Phylum:** {species_data['phylum']}\n"
        if species_data.get('class'):
            summary += f"**Class:** {species_data['class']}\n"
        if species_data.get('family'):
            summary += f"**Family:** {species_data['family']}\n"
        
        if synonyms:
            summary += f"\n**Synonyms ({len(synonyms)}):**\n"
            for syn in synonyms[:5]:  # Show first 5
                summary += f"- {syn.get('scientificname', 'Unknown')}\n"
            if len(synonyms) > 5:
                summary += f"- ... and {len(synonyms) - 5} more\n"
        
        return summary

    def _create_distribution_summary(self, species_data: dict, distributions: list) -> str:
        """Create distribution summary"""
        scientific_name = species_data['scientificname']
        summary = f"**Distribution for {scientific_name}**\n\n"
        summary += f"Found in **{len(distributions)} locations**:\n\n"
        
        countries = set()
        for dist in distributions:
            if dist.get('country'):
                countries.add(dist['country'])
        
        if countries:
            summary += f"**Countries/Regions:** {', '.join(sorted(list(countries))[:10])}\n"
            if len(countries) > 10:
                summary += f"... and {len(countries) - 10} more regions\n"
        
        return summary

    def _create_vernacular_summary(self, species_data: dict, vernaculars: list) -> str:
        """Create vernacular names summary"""
        scientific_name = species_data['scientificname']
        summary = f"**Common names for {scientific_name}**\n\n"
        
        by_language = {}
        for vern in vernaculars:
            lang = vern.get('language', 'Unknown')
            if lang not in by_language:
                by_language[lang] = []
            by_language[lang].append(vern.get('vernacular', ''))
        
        for lang, names in list(by_language.items())[:5]:
            summary += f"**{lang}:** {', '.join(names[:3])}\n"
        
        if len(by_language) > 5:
            summary += f"\n... and names in {len(by_language) - 5} more languages\n"
        
        return summary

    # Keep your existing API methods exactly the same
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