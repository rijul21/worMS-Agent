# worms_agent_server.py
from typing_extensions import override
from pydantic import BaseModel
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext  
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint

# Import your existing agent workflow and Pydantic models
from worms_agent import (
    WoRMSiChatBioAgent,
    MarineSynonymsParams,
    MarineDistributionParams,
    MarineVernacularParams,
    MarineSourcesParams,
    MarineRecordParams,
    MarineClassificationParams,
    MarineChildrenParams
)
from worms_client import NoParams

# --- AgentCard definition with 7 endpoints ---
card = AgentCard(
    name="WoRMS Marine Species Agent",
    description="Retrieves detailed marine species information from WoRMS (World Register of Marine Species) database including synonyms, distribution, common names, literature sources, taxonomic records, classification, and child taxa.",
    icon="https://www.marinespecies.org/images/WoRMS_logo.png",
    url="http://localhost:9999",  
    entrypoints=[
        AgentEntrypoint(
            id="get_synonyms",
            description="Get synonyms and alternative names for a marine species from WoRMS.",
            parameters=MarineSynonymsParams
        ),
        AgentEntrypoint(
            id="get_distribution",
            description="Get distribution data and geographic locations for a marine species from WoRMS.",
            parameters=MarineDistributionParams
        ),
        AgentEntrypoint(
            id="get_vernacular_names",
            description="Get vernacular/common names for a marine species in different languages from WoRMS.",
            parameters=MarineVernacularParams
        ),
        AgentEntrypoint(
            id="get_sources",
            description="Get literature sources and references for a marine species from WoRMS.",
            parameters=MarineSourcesParams
        ),
        AgentEntrypoint(
            id="get_record",
            description="Get basic taxonomic record and classification for a marine species from WoRMS.",
            parameters=MarineRecordParams
        ),
        AgentEntrypoint(
            id="get_taxonomy",
            description="Get complete taxonomic classification hierarchy for a marine species from WoRMS.",
            parameters=MarineClassificationParams
        ),
        AgentEntrypoint(
            id="get_marine_info",
            description="Get child taxa (subspecies, varieties, forms) for a marine species from WoRMS.",
            parameters=MarineChildrenParams
        )
    ]
)

# --- Implement the iChatBio agent class ---
class WoRMSAgent(IChatBioAgent):
    def __init__(self):
        self.workflow_agent = WoRMSiChatBioAgent()

    @override
    def get_agent_card(self) -> AgentCard:
        """Returns the agent's metadata card."""
        return card

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: BaseModel):
        """Executes the requested agent entrypoint using the provided context."""
        
        # Debug logging 
        print(f"=== WoRMS DEBUG INFO ===")
        print(f"request: {request}")
        print(f"entrypoint: {entrypoint}")
        print(f"params type: {type(params)}")
        print(f"params: {params}")
        print(f"========================")
        
        if entrypoint == "get_synonyms":
            await self.workflow_agent.run_get_synonyms(context, params)
        elif entrypoint == "get_distribution":
            await self.workflow_agent.run_get_distribution(context, params)
        elif entrypoint == "get_vernacular_names":
            await self.workflow_agent.run_get_vernacular_names(context, params)
        elif entrypoint == "get_sources":
            await self.workflow_agent.run_get_sources(context, params)
        elif entrypoint == "get_record":
            await self.workflow_agent.run_get_record(context, params)
        elif entrypoint == "get_taxonomy":
            await self.workflow_agent.run_get_classification(context, params)
        elif entrypoint == "get_marine_info":
            await self.workflow_agent.run_get_children(context, params)
        else:
            # Handle unexpected entrypoints 
            await context.reply(f"Unknown entrypoint '{entrypoint}' received. Request was: '{request}'")
            raise ValueError(f"Unsupported entrypoint: {entrypoint}")

if __name__ == "__main__":
    agent = WoRMSAgent()
    print(f"Starting iChatBio agent server for '{card.name}' at http://localhost:9999")
    print("Available endpoints:")
    for ep in card.entrypoints:
        print(f"  - {ep.id}: {ep.description}")
    run_agent_server(agent, host="0.0.0.0", port=9999)