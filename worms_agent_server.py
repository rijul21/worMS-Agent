# worms_agent_server.py
from typing_extensions import override
from pydantic import BaseModel
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext  
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint

# Import your existing agent workflow and Pydantic models
from worms_agent import WoRMSiChatBioAgent
from worms_client import (
    MarineAttributesParams,
    MarineSynonymsParams,
    MarineDistributionParams,
    NoParams
)

# --- AgentCard definition with url added ---
card = AgentCard(
    name="WoRMS Marine Species Agent",
    description="Retrieves detailed marine species information from WoRMS (World Register of Marine Species) database.",
    icon="https://www.marinespecies.org/images/WoRMS_logo.png",
    url="http://localhost:9999",  
    entrypoints=[
        AgentEntrypoint(
            id="get_attributes",
            description="Get attributes and characteristics for a marine species from WoRMS.",
            parameters=MarineAttributesParams
        ),
        AgentEntrypoint(
            id="get_synonyms",
            description="Get synonyms and alternative names for a marine species from WoRMS.",
            parameters=MarineSynonymsParams
        ),
        AgentEntrypoint(
            id="get_distribution",
            description="Get distribution data and geographic locations for a marine species from WoRMS.",
            parameters=MarineDistributionParams
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
        
        if entrypoint == "get_attributes":
            await self.workflow_agent.run_get_attributes(context, params)
        elif entrypoint == "get_synonyms":
            await self.workflow_agent.run_get_synonyms(context, params)
        elif entrypoint == "get_distribution":
            await self.workflow_agent.run_get_distribution(context, params)
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