from pydantic import BaseModel, Field
import asyncio
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from worms_client import (
    WoRMS,
    SynonymsParams, 
    DistributionParams,
    VernacularParams,
    SourcesParams,
    RecordParams,
)

# Simple Agent Parameter Models - 5 endpoints (removed attributes, added 3 new)
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

class MarineVernacularParams(BaseModel):
    """Parameters for getting marine species vernacular/common names"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_vernaculars: Optional[bool] = Field(True,
        description="Include vernacular names in results"
    )

class MarineSourcesParams(BaseModel):
    """Parameters for getting marine species literature sources/references"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_sources: Optional[bool] = Field(True,
        description="Include literature sources in results"
    )

class MarineRecordParams(BaseModel):
    """Parameters for getting marine species basic taxonomic record"""
    species_name: str = Field(...,
        description="Scientific name of the marine species",
        examples=["Orcinus orca", "Delphinus delphis", "Tursiops truncatus"]
    )
    include_record: Optional[bool] = Field(True,
        description="Include taxonomic record in results"
    )

class WoRMSiChatBioAgent:
    """The iChatBio agent implementation for WoRMS - 5 endpoint version"""

    def __init__(self):
        self.worms_logic = WoRMS()

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
                    
                    # Create engaging, detailed reply
                    accepted_names = [syn for syn in synonyms if isinstance(syn, dict) and syn.get('status') == 'accepted']
                    unaccepted_names = [syn for syn in synonyms if isinstance(syn, dict) and syn.get('status') != 'accepted']
                    
                    reply_parts = [f"🔍 **Taxonomic History of {params.species_name}**"]
                    reply_parts.append(f"I found **{synonym_count} historical names** for this species in the WoRMS database!")
                    
                    if accepted_names:
                        reply_parts.append(f"✅ **Currently accepted**: {len(accepted_names)} names")
                    
                    if unaccepted_names:
                        reply_parts.append(f"📚 **Historical/invalid names**: {len(unaccepted_names)} entries")
                        
                        # Group by common patterns
                        genus_changes = []
                        spelling_variants = []
                        for syn in unaccepted_names[:8]:
                            if isinstance(syn, dict):
                                name = syn.get('scientificname', '')
                                if name:
                                    if name.split()[0] != params.species_name.split()[0]:  # Different genus
                                        genus_changes.append(name)
                                    else:
                                        spelling_variants.append(name)
                        
                        if genus_changes:
                            reply_parts.append(f"🔄 **Genus transfers**: This species was previously classified in genera like *{', '.join(set([n.split()[0] for n in genus_changes[:3]]))}*")
                        
                        if spelling_variants:
                            reply_parts.append(f"✏️ **Spelling variants**: {', '.join(spelling_variants[:3])}")
                    
                    reply_parts.append(f"📊 **Complete dataset** with all {synonym_count} names, authorities, and publication years is available in the artifact above.")
                    
                    await context.reply("\n\n".join(reply_parts))
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
                    
                    # Extract rich location details for engaging response
                    countries = set()
                    localities = []
                    regions = set()
                    marine_areas = []
                    
                    for dist in distributions:
                        if isinstance(dist, dict):
                            if dist.get('country'):
                                countries.add(dist['country'])
                            if dist.get('locality'):
                                locality = dist['locality']
                                localities.append(locality)
                                # Categorize locations
                                if any(term in locality.lower() for term in ['sea', 'ocean', 'atlantic', 'pacific', 'mediterranean']):
                                    marine_areas.append(locality)
                            if dist.get('locationID'):
                                regions.add(dist.get('locationID', ''))
                    
                    # Create engaging, detailed reply
                    reply_parts = [f"🌍 **Global Distribution of {params.species_name}**"]
                    reply_parts.append(f"This species has been documented across **{distribution_count} distinct locations** worldwide!")
                    
                    if countries:
                        country_list = sorted(list(countries))
                        if len(country_list) <= 8:
                            reply_parts.append(f"🏴 **Countries/Regions**: {', '.join(country_list)}")
                        else:
                            reply_parts.append(f"🏴 **Countries/Regions**: {', '.join(country_list[:8])} and {len(country_list)-8} others")
                    
                    if marine_areas:
                        reply_parts.append(f"🌊 **Major Marine Areas**: {', '.join(marine_areas[:4])}")
                    
                    if localities:
                        coastal_areas = [loc for loc in localities if any(term in loc.lower() for term in ['coast', 'coastal', 'shore', 'estuary', 'bay'])]
                        if coastal_areas:
                            reply_parts.append(f"🏖️ **Coastal Presence**: Found in {len(coastal_areas)} coastal/estuarine areas")
                    
                    # Add ecological insight
                    if len(countries) > 15:
                        reply_parts.append(f"🌐 **Ecological Note**: With presence in {len(countries)}+ regions, this appears to be a **cosmopolitan species** with global distribution!")
                    elif len(countries) > 8:
                        reply_parts.append(f"📍 **Distribution Pattern**: **Wide-ranging species** found across multiple biogeographic regions")
                    else:
                        reply_parts.append(f"📍 **Distribution Pattern**: **Regional distribution** with documented presence in {len(countries)} areas")
                    
                    reply_parts.append(f"📊 **Detailed location data** with coordinates and specific locality information is available in the artifact above.")
                    
                    await context.reply("\n\n".join(reply_parts))
                else:
                    await context.reply(f"No distribution data found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving distribution data: {e}")

    async def run_get_vernacular_names(self, context, params: MarineVernacularParams):
        """Workflow for getting marine species vernacular/common names"""
        async with context.begin_process(f"Getting vernacular names for '{params.species_name}'") as process:
            await process.log("Vernacular names search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get vernacular names
                vern_params = VernacularParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_vernacular_url(vern_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    vernaculars = raw_response
                elif isinstance(raw_response, dict):
                    vernaculars = [raw_response]
                else:
                    vernaculars = []

                vernacular_count = len(vernaculars)
                
                if vernacular_count > 0:
                    await process.log(f"Found {vernacular_count} vernacular names")
                    
                    # Extract sample vernacular names by language for display
                    sample_names = []
                    languages = set()
                    
                    for vern in vernaculars[:10]:  # Show first 10
                        if isinstance(vern, dict):
                            name = vern.get('vernacular', 'Unknown')
                            lang = vern.get('language', 'Unknown')
                            if lang != 'Unknown':
                                languages.add(lang)
                                sample_names.append(f"{name} ({lang})")
                            else:
                                sample_names.append(name)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species vernacular names for {params.species_name} (AphiaID: {aphia_id}) - {vernacular_count} names",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Vernacular Names",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "vernacular_count": vernacular_count,
                            "languages": list(languages)
                        }
                    )
                    
                    # Extract rich vernacular data for engaging response
                    sample_names = []
                    languages = set()
                    language_groups = {}
                    interesting_names = []
                    
                    for vern in vernaculars:
                        if isinstance(vern, dict):
                            name = vern.get('vernacular', 'Unknown')
                            lang = vern.get('language', 'Unknown')
                            if lang != 'Unknown':
                                languages.add(lang)
                                if lang not in language_groups:
                                    language_groups[lang] = []
                                language_groups[lang].append(name)
                                
                                # Collect interesting/unique names
                                if len(name) > 15 or any(char in name for char in ['ä', 'ö', 'ü', 'ñ', 'ç', 'é', 'è']):
                                    interesting_names.append(f"{name} ({lang})")
                                
                                sample_names.append(f"**{name}** ({lang})")
                    
                    # Create engaging, detailed reply
                    reply_parts = [f"🗣️ **Common Names for {params.species_name}**"]
                    reply_parts.append(f"This species is known by **{vernacular_count} different names** across **{len(languages)} languages**!")
                    
                    # Highlight language diversity
                    if len(languages) > 10:
                        reply_parts.append(f"🌍 **Global Recognition**: This species has names in {len(languages)} languages, showing its **worldwide cultural significance**!")
                    
                    # Show interesting examples by language
                    lang_examples = []
                    for lang in sorted(list(languages))[:6]:
                        if lang in language_groups and language_groups[lang]:
                            examples = language_groups[lang][:2]
                            lang_examples.append(f"**{lang}**: {', '.join(examples)}")
                    
                    if lang_examples:
                        reply_parts.append("🏷️ **Examples by Language**:")
                        reply_parts.extend([f"  • {ex}" for ex in lang_examples])
                    
                    # Highlight unique/interesting names
                    if interesting_names:
                        reply_parts.append(f"✨ **Notable Names**: {', '.join(interesting_names[:3])}")
                    
                    # Cultural insight
                    indigenous_langs = [lang for lang in languages if lang.lower() in ['inuktitut', 'aleut', 'kalaallisut', 'yupik', 'inupiaq']]
                    if indigenous_langs:
                        reply_parts.append(f"🏔️ **Indigenous Knowledge**: Names documented in {len(indigenous_langs)} indigenous languages, reflecting traditional ecological knowledge")
                    
                    reply_parts.append(f"📚 **Complete multilingual dataset** with all {vernacular_count} names and language codes is available in the artifact above.")
                    
                    await context.reply("\n\n".join(reply_parts))
                else:
                    await context.reply(f"No vernacular names found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving vernacular names: {e}")

    async def run_get_sources(self, context, params: MarineSourcesParams):
        """Workflow for getting marine species literature sources/references"""
        async with context.begin_process(f"Getting sources for '{params.species_name}'") as process:
            await process.log("Sources search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get sources
                sources_params = SourcesParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_sources_url(sources_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format
                if isinstance(raw_response, list):
                    sources = raw_response
                elif isinstance(raw_response, dict):
                    sources = [raw_response]
                else:
                    sources = []

                source_count = len(sources)
                
                if source_count > 0:
                    await process.log(f"Found {source_count} sources")
                    
                    # Extract sample sources for display
                    sample_sources = []
                    authors = set()
                    years = set()
                    
                    for source in sources[:8]:  # Show first 8
                        if isinstance(source, dict):
                            title = source.get('title', 'Unknown')
                            author = source.get('authors', source.get('author', ''))
                            year = source.get('year', '')
                            
                            if author:
                                authors.add(author.split(',')[0])  # First author
                            if year:
                                years.add(str(year))
                            
                            if author and year:
                                sample_sources.append(f"{author} ({year})")
                            elif title:
                                sample_sources.append(title[:50] + "..." if len(title) > 50 else title)
                    
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Marine species literature sources for {params.species_name} (AphiaID: {aphia_id}) - {source_count} sources",
                        uris=[api_url],
                        metadata={
                            "data_source": "WoRMS Sources",
                            "aphia_id": aphia_id,
                            "scientific_name": params.species_name,
                            "source_count": source_count,
                            "authors": list(authors),
                            "years": sorted(list(years))
                        }
                    )
                    
                    # Extract rich source data for engaging response
                    sample_sources = []
                    authors = set()
                    years = set()
                    publications = []
                    source_types = set()
                    
                    for source in sources:
                        if isinstance(source, dict):
                            title = source.get('title', 'Unknown')
                            author = source.get('authors', source.get('author', ''))
                            year = source.get('year', '')
                            source_type = source.get('type', '')
                            
                            if author and author != 'Unknown':
                                # Extract first author surname
                                first_author = author.split(',')[0].strip()
                                authors.add(first_author)
                            
                            if year and str(year) != 'Unknown':
                                years.add(str(year))
                            
                            if title and title != 'Unknown' and len(title) > 5:
                                publications.append(title)
                            
                            if source_type:
                                source_types.add(source_type)
                            
                            # Create formatted citation
                            if author and year:
                                sample_sources.append(f"**{first_author}** ({year})")
                            elif title and len(title) < 80:
                                sample_sources.append(f"*{title}*")
                    
                    # Create engaging, detailed reply
                    reply_parts = [f"📚 **Scientific Literature for {params.species_name}**"]
                    reply_parts.append(f"Found **{source_count} academic sources** documenting this species!")
                    
                    # Publication timeline
                    if years:
                        year_list = sorted([int(y) for y in years if y.isdigit()])
                        if year_list:
                            earliest = min(year_list)
                            latest = max(year_list)
                            if latest - earliest > 50:
                                reply_parts.append(f"📅 **Research Timeline**: Scientific documentation spans **{latest - earliest} years** ({earliest}-{latest})")
                            else:
                                reply_parts.append(f"📅 **Publication Period**: {earliest}-{latest}")
                    
                    # Author diversity
                    if authors:
                        clean_authors = [a for a in authors if a != 'Unknown' and len(a) > 2]
                        if len(clean_authors) > 5:
                            reply_parts.append(f"👥 **Research Community**: {len(clean_authors)} different research groups/authors have published on this species")
                            reply_parts.append(f"🔬 **Key Contributors**: {', '.join(list(clean_authors)[:4])}")
                    
                    # Publication insights
                    if publications:
                        marine_terms = sum(1 for pub in publications if any(term in pub.lower() for term in ['marine', 'ocean', 'sea', 'fish', 'taxonomy', 'systematics']))
                        if marine_terms > len(publications) * 0.5:
                            reply_parts.append(f"🌊 **Research Focus**: Strong emphasis on **marine biology and taxonomy**")
                    
                    # Source examples
                    if sample_sources:
                        reply_parts.append(f"📖 **Sample References**: {', '.join(sample_sources[:4])}")
                        if source_count > 4:
                            reply_parts.append(f"   *...and {source_count - 4} additional sources*")
                    
                    # Research status insight
                    if source_count > 15:
                        reply_parts.append(f"⭐ **Research Status**: **Well-studied species** with extensive scientific literature")
                    elif source_count > 5:
                        reply_parts.append(f"📊 **Research Status**: **Moderately documented** species in scientific literature")
                    else:
                        reply_parts.append(f"🔍 **Research Status**: **Limited documentation** - potential area for further research")
                    
                    reply_parts.append(f"📋 **Complete bibliography** with full citations and publication details is available in the artifact above.")
                    
                    await context.reply("\n\n".join(reply_parts))
                else:
                    await context.reply(f"No literature sources found for {params.species_name} in WoRMS.")

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving sources: {e}")

    async def run_get_record(self, context, params: MarineRecordParams):
        """Workflow for getting marine species basic taxonomic record"""
        async with context.begin_process(f"Getting basic record for '{params.species_name}'") as process:
            await process.log("Record search parameters", data=params.model_dump(exclude_defaults=True))

            try:
                # Step 1: Get AphiaID
                await process.log(f"Looking up AphiaID for '{params.species_name}'...")
                loop = asyncio.get_event_loop()
                aphia_id = await loop.run_in_executor(None, lambda: self.worms_logic.get_species_aphia_id(params.species_name))
                
                if not aphia_id:
                    await context.reply(f"Could not find '{params.species_name}' in WoRMS database.")
                    return

                await process.log(f"Found AphiaID: {aphia_id}")

                # Step 2: Get record
                record_params = RecordParams(aphia_id=aphia_id)
                api_url = self.worms_logic.build_record_url(record_params)
                await process.log(f"Constructed API URL: {api_url}")

                raw_response = await loop.run_in_executor(None, lambda: self.worms_logic.execute_request(api_url))
                
                # Handle response format - this endpoint returns a single object
                if isinstance(raw_response, dict):
                    record = raw_response
                else:
                    await context.reply(f"Unexpected response format for {params.species_name} record.")
                    return

                await process.log("Found taxonomic record")
                
                # Extract key information for display
                scientific_name = record.get('scientificname', 'Unknown')
                authority = record.get('authority', '')
                status = record.get('status', 'Unknown')
                rank = record.get('rank', 'Unknown')
                kingdom = record.get('kingdom', '')
                phylum = record.get('phylum', '')
                class_name = record.get('class', '')
                order = record.get('order', '')
                family = record.get('family', '')
                genus = record.get('genus', '')
                
                await process.create_artifact(
                    mimetype="application/json",
                    description=f"Marine species taxonomic record for {params.species_name} (AphiaID: {aphia_id})",
                    uris=[api_url],
                    metadata={
                        "data_source": "WoRMS Taxonomic Record",
                        "aphia_id": aphia_id,
                        "scientific_name": scientific_name,
                        "status": status,
                        "rank": rank,
                        "taxonomy": {
                            "kingdom": kingdom,
                            "phylum": phylum,
                            "class": class_name,
                            "order": order,
                            "family": family,
                            "genus": genus
                        }
                    }
                )
                
                # Create detailed user-friendly response
                reply_parts = [f"Retrieved taxonomic record for {scientific_name}"]
                
                if authority:
                    reply_parts.append(f"Authority: {authority}")
                
                reply_parts.append(f"Status: {status}, Rank: {rank}")
                
                # Build taxonomy string
                taxonomy_parts = []
                for tax_rank, tax_name in [("Kingdom", kingdom), ("Phylum", phylum), ("Class", class_name), 
                                         ("Order", order), ("Family", family), ("Genus", genus)]:
                    if tax_name:
                        taxonomy_parts.append(f"{tax_rank}: {tax_name}")
                
                if taxonomy_parts:
                    reply_parts.append(f"Taxonomy - {'; '.join(taxonomy_parts)}")
                
                reply_parts.append("I've created an artifact with the complete taxonomic record.")
                
                await context.reply(". ".join(reply_parts))

            except Exception as e:
                await process.log("Error during API request", data={"error": str(e)})
                await context.reply(f"I encountered an error while retrieving the taxonomic record: {e}")