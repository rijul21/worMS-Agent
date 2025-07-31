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
                    description="Returns detailed marine species information from WoRMS",
                    parameters=None
                ),
                AgentEntrypoint(
                    id="get_synonyms",
                    description="Get synonyms for marine species",
                    parameters=None
                ),
                AgentEntrypoint(
                    id="get_distribution",
                    description="Get distribution data for marine species", 
                    parameters=None
                ),
                AgentEntrypoint(
                    id="get_vernacular",
                    description="Get vernacular/common names for marine species",
                    parameters=None
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
        elif entrypoint == "get_vernacular":
            await self.get_vernacular(context, request)
        else:
            await context.reply(f"Unknown entrypoint: {entrypoint}")

    async def get_marine_info(self, context: ResponseContext, request: str):
        async with context.begin_process(summary="Searching WoRMS for marine species") as process:
            try:
                # Extract species name using instructor
                openai_client = AsyncOpenAI(
                    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                    api_key=os.getenv("GROQ_API_KEY")
                )
                instructor_client = instructor.patch(openai_client)

                marine_query: MarineQueryModel = await instructor_client.chat.completions.create(
                    model="llama3-70b-8192",
                    response_model=MarineQueryModel,
                    messages=[
                        {
                            "role": "system",
                            "content": "Extract the scientific name of a marine species from the user's message. Return JSON format: {\"scientific_name\": \"Genus species\"}"
                        },
                        {"role": "user", "content": request}
                    ],
                    max_retries=3
                )

                await process.log(f"Identified marine species: {marine_query.scientific_name}")

                # Search WoRMS database using run_in_executor pattern
                await process.log(f"Searching WoRMS database for: {marine_query.scientific_name}")
                
                loop = asyncio.get_event_loop()
                worms_data = await loop.run_in_executor(None, lambda: self.get_worms_data(marine_query.scientific_name))
                
                if not worms_data:
                    await context.reply(f"No marine species found for '{marine_query.scientific_name}' in WoRMS database.")
                    return

                species = worms_data['species']
                aphia_id = species['AphiaID']
                
                await process.log(f"Found species: {species['scientificname']} (AphiaID: {aphia_id})")

                # Store raw_response like ALA does
                raw_response = worms_data
                total = len(worms_data.get('synonyms', [])) + len(worms_data.get('vernaculars', [])) + len(worms_data.get('distributions', []))
                returned = 1

                await process.log("Query successful, found species data.")
                
                # Create artifact exactly like ALA - NO content parameter
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Raw JSON for marine species {species['scientificname']}.",
                    uris=[f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"],
                    metadata={"record_count": returned, "total_matches": total}
                )
                
                # Create rich, informative response
                response = self.format_species_summary(worms_data)
                
                await context.reply(response)

            except InstructorRetryException as e:
                await context.reply("Sorry, I couldn't extract marine species information from your request.")
            except Exception as e:
                await context.reply(f"An error occurred while retrieving marine species information: {str(e)}")

    async def get_synonyms(self, context: ResponseContext, request: str):
        """Get synonyms for a marine species"""
        async with context.begin_process("Getting synonyms from WoRMS") as process:
            try:
                # Extract species name
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
                    max_retries=3
                )

                await process.log(f"Getting synonyms for: {marine_query.scientific_name}")
                
                loop = asyncio.get_event_loop()
                synonyms_data = await loop.run_in_executor(None, lambda: self.get_synonyms_data(marine_query.scientific_name))
                
                if not synonyms_data:
                    await context.reply(f"No synonyms found for '{marine_query.scientific_name}' in WoRMS.")
                    return

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Synonyms for {marine_query.scientific_name}",
                    uris=[f"https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/{synonyms_data['aphia_id']}"],
                    metadata={"synonym_count": len(synonyms_data['synonyms'])}
                )
                
                await context.reply(self.format_synonyms_response(synonyms_data))

            except Exception as e:
                await context.reply(f"Error getting synonyms: {str(e)}")

    async def get_distribution(self, context: ResponseContext, request: str):
        """Get distribution for a marine species"""
        async with context.begin_process("Getting distribution from WoRMS") as process:
            try:
                # Extract species name
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
                    max_retries=3
                )

                await process.log(f"Getting distribution for: {marine_query.scientific_name}")
                
                loop = asyncio.get_event_loop()
                distribution_data = await loop.run_in_executor(None, lambda: self.get_distribution_data(marine_query.scientific_name))
                
                if not distribution_data:
                    await context.reply(f"No distribution data found for '{marine_query.scientific_name}' in WoRMS.")
                    return

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Distribution for {marine_query.scientific_name}",
                    uris=[f"https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/{distribution_data['aphia_id']}"],
                    metadata={"distribution_count": len(distribution_data['distributions'])}
                )
                
                await context.reply(self.format_distribution_response(distribution_data))

            except Exception as e:
                await context.reply(f"Error getting distribution: {str(e)}")

    async def get_vernacular(self, context: ResponseContext, request: str):
        """Get vernacular names for a marine species"""
        async with context.begin_process("Getting vernacular names from WoRMS") as process:
            try:
                # Extract species name
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
                    max_retries=3
                )

                await process.log(f"Getting vernacular names for: {marine_query.scientific_name}")
                
                loop = asyncio.get_event_loop()
                vernacular_data = await loop.run_in_executor(None, lambda: self.get_vernacular_data(marine_query.scientific_name))
                
                if not vernacular_data:
                    await context.reply(f"No vernacular names found for '{marine_query.scientific_name}' in WoRMS.")
                    return

                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Vernacular names for {marine_query.scientific_name}",
                    uris=[f"https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/{vernacular_data['aphia_id']}"],
                    metadata={"vernacular_count": len(vernacular_data['vernaculars'])}
                )
                
                await context.reply(self.format_vernacular_response(vernacular_data))

            except Exception as e:
                await context.reply(f"Error getting vernacular names: {str(e)}")

    def get_synonyms_data(self, scientific_name: str) -> dict:
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

    def get_worms_data(self, scientific_name: str) -> dict:
        """Get complete WoRMS data for a species - synchronous for run_in_executor"""
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
                
            # Return first record (WoRMS API can return list or single record)
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

    def format_species_summary(self, worms_data: dict) -> str:
        """Format a comprehensive species summary"""
        species = worms_data['species']
        synonyms = worms_data.get('synonyms', [])
        vernaculars = worms_data.get('vernaculars', [])
        distributions = worms_data.get('distributions', [])
        
        # Build taxonomy chain
        taxonomy_parts = [
            f"Kingdom {species.get('kingdom', 'N/A')}",
            f"Phylum {species.get('phylum', 'N/A')}",
            f"Class {species.get('class', 'N/A')}",
            f"Family {species.get('family', 'N/A')}"
        ]
        taxonomy = " > ".join([part for part in taxonomy_parts if "N/A" not in part])
        
        # Extract key distribution regions (first 5)
        key_regions = []
        for dist in distributions[:5]:
            if dist.get('locality'):
                key_regions.append(dist['locality'])
            elif dist.get('country'):
                key_regions.append(dist['country'])
        
        # Extract primary common names (first 4)
        primary_names = []
        for vern in vernaculars[:4]:
            if vern.get('vernacular'):
                primary_names.append(vern['vernacular'])
        
        # Extract recent synonyms (first 3)
        recent_synonyms = []
        for syn in synonyms[:3]:
            if syn.get('scientificname'):
                recent_synonyms.append(syn['scientificname'])
        
        # Build formatted response
        response = f"**{species['scientificname']}** (AphiaID: {species['AphiaID']})\n\n"
        
        if taxonomy:
            response += f"**Taxonomy**: {taxonomy}\n\n"
        
        if species.get('authority'):
            response += f"**Authority**: {species['authority']}\n\n"
        
        if key_regions:
            response += f"**Distribution**: Found in {', '.join(key_regions)}"
            if len(distributions) > 5:
                response += f" and {len(distributions) - 5} additional regions"
            response += "\n\n"
        
        if primary_names:
            response += f"**Common Names**: {', '.join(primary_names)}"
            if len(vernaculars) > 4:
                response += f" ({len(vernaculars)} total names)"
            response += "\n\n"
        
        if recent_synonyms:
            response += f"**Key Synonyms**: {', '.join(recent_synonyms)}"
            if len(synonyms) > 3:
                response += f" ({len(synonyms)} total synonyms)"
            response += "\n\n"
        
        response += f"**Additional Data**: Complete taxonomic details, {len(distributions)} distribution records, and {len(vernaculars)} vernacular names available in the attached artifact."
        
        return response
    
    def format_synonyms_response(self, synonyms_data: dict) -> str:
        """Format synonyms response with key highlights"""
        synonyms = synonyms_data['synonyms']
        species_name = synonyms_data['species_name']
        
        if not synonyms:
            return f"No synonyms found for **{species_name}** in WoRMS."
        
        # Group by status if available
        accepted_synonyms = []
        other_synonyms = []
        
        for syn in synonyms:
            name = syn.get('scientificname', 'Unknown')
            status = syn.get('status', '').lower()
            
            if 'accepted' in status or 'valid' in status:
                accepted_synonyms.append(name)
            else:
                other_synonyms.append(name)
        
        response = f"**Synonyms for {species_name}**\n\n"
        
        # Show first 6 synonyms with their details
        key_synonyms = synonyms[:6]
        for i, syn in enumerate(key_synonyms, 1):
            name = syn.get('scientificname', 'Unknown')
            authority = syn.get('authority', '')
            status = syn.get('status', '')
            
            response += f"{i}. **{name}**"
            if authority:
                response += f" {authority}"
            if status:
                response += f" [{status}]"
            response += "\n"
        
        if len(synonyms) > 6:
            response += f"\n**Additional**: {len(synonyms) - 6} more synonyms available in the attached artifact."
        
        response += f"\n\n**Total**: {len(synonyms)} synonyms found in WoRMS database."
        
        return response
    
    def format_distribution_response(self, distribution_data: dict) -> str:
        """Format distribution response with geographic details"""
        distributions = distribution_data['distributions']
        species_name = distribution_data['species_name']
        
        if not distributions:
            return f"No distribution data found for **{species_name}** in WoRMS."
        
        # Group by establishment means if available
        native_regions = []
        introduced_regions = []
        other_regions = []
        
        for dist in distributions:
            locality = dist.get('locality') or dist.get('country', 'Unknown location')
            establishment = dist.get('establishmentMeans', '').lower()
            
            if 'native' in establishment:
                native_regions.append(locality)
            elif 'introduced' in establishment:
                introduced_regions.append(locality)
            else:
                other_regions.append(locality)
        
        response = f"**Distribution of {species_name}**\n\n"
        
        # Show regions by type
        if native_regions:
            response += f"**Native Range**: {', '.join(native_regions[:5])}"
            if len(native_regions) > 5:
                response += f" and {len(native_regions) - 5} more regions"
            response += "\n\n"
        
        if introduced_regions:
            response += f"**Introduced Range**: {', '.join(introduced_regions[:5])}"
            if len(introduced_regions) > 5:
                response += f" and {len(introduced_regions) - 5} more regions"
            response += "\n\n"
        
        if other_regions:
            response += f"**Other Locations**: {', '.join(other_regions[:5])}"
            if len(other_regions) > 5:
                response += f" and {len(other_regions) - 5} more regions"
            response += "\n\n"
        
        response += f"**Total**: {len(distributions)} distribution records found in WoRMS database."
        
        return response
    
    def format_vernacular_response(self, vernacular_data: dict) -> str:
        """Format vernacular names response by language"""
        vernaculars = vernacular_data['vernaculars']
        species_name = vernacular_data['species_name']
        
        if not vernaculars:
            return f"No vernacular names found for **{species_name}** in WoRMS."
        
        # Group by language
        by_language = {}
        for vern in vernaculars:
            name = vern.get('vernacular', 'Unknown')
            language = vern.get('language', 'Unknown language')
            
            if language not in by_language:
                by_language[language] = []
            by_language[language].append(name)
        
        response = f"**Common Names for {species_name}**\n\n"
        
        # Show top languages with their names
        for i, (language, names) in enumerate(list(by_language.items())[:5], 1):
            response += f"**{language}**: {', '.join(names[:3])}"
            if len(names) > 3:
                response += f" and {len(names) - 3} more"
            response += "\n"
        
        if len(by_language) > 5:
            response += f"\n**Additional**: Names in {len(by_language) - 5} more languages available in the attached artifact."
        
        response += f"\n\n**Total**: {len(vernaculars)} vernacular names in {len(by_language)} languages found in WoRMS database."
        
        return response

print("INIT: WoRMS Marine Species Agent loaded successfully")