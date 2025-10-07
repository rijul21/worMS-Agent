from typing import override
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
import dotenv
import asyncio

from worms_api import WoRMS, SynonymsParams

dotenv.load_dotenv()

class MarineResearchParams(BaseModel):
    """Parameters for marine species research requests"""
    species_names: list[str] = Field(
        default=[],
        description="Scientific names of marine species to research",
        examples=[["Orcinus orca"], ["Orcinus orca", "Delphinus delphis"]]
    )

AGENT_DESCRIPTION = "Marine species research assistant using WoRMS database"


class WoRMSReActAgent(IChatBioAgent):
    def __init__(self):
        self.worms_logic = WoRMS()
        
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="WoRMS Agent",
            description=AGENT_DESCRIPTION,
            icon="https://www.marinespecies.org/images/WoRMS_logo.png",
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="research_marine_species",
                    description=AGENT_DESCRIPTION,
                    parameters=MarineResearchParams
                )
            ]
        )
    
    @override
    async def run(
        self,
        context: ResponseContext,
        request: str,
        entrypoint: str,
        params: MarineResearchParams,
    ):
        """Main entry point - builds and executes the ReAct agent loop"""
        
        # Control tools
        @tool(return_direct=True)
        async def abort(reason: str):
            """Call if you cannot fulfill the request."""
            await context.reply(f"Unable to complete request: {reason}")

        @tool(return_direct=True)
        async def finish(summary: str):
            """Call when request is successfully completed."""
            await context.reply(summary)

        @tool
        async def get_species_synonyms(species_name: str) -> str:
            """Get synonyms and alternative scientific names for a marine species.
            
            Args:
                species_name: Scientific name (e.g., "Orcinus orca")
            """
            async with context.begin_process(f"Fetching synonyms: {species_name}") as process:
                try:
                    await process.log(f"Looking up species: {species_name}")
                    
                    loop = asyncio.get_event_loop()
                    aphia_id = await loop.run_in_executor(
                        None, 
                        lambda: self.worms_logic.get_species_aphia_id(species_name)
                    )
                    
                    if not aphia_id:
                        await process.log(f"Species not found: {species_name}")
                        return f"Species '{species_name}' not found in WoRMS database."
                    
                    await process.log(f" Found AphiaID: {aphia_id}")
                    
                    # Build API request
                    syn_params = SynonymsParams(aphia_id=aphia_id)
                    api_url = self.worms_logic.build_synonyms_url(syn_params)
                    await process.log(f"API endpoint: {api_url}")
                    
                    # Execute request
                    await process.log("Calling WoRMS API...")
                    raw_response = await loop.run_in_executor(
                        None,
                        lambda: self.worms_logic.execute_request(api_url)
                    )
                    
                    # Normalize response
                    synonyms = raw_response if isinstance(raw_response, list) else [raw_response] if raw_response else []
                    await process.log(f"Received {len(synonyms)} synonym records")
                    
                    if not synonyms:
                        await process.log(f"⚠️ No synonyms found for {species_name}")
                        return f"No synonyms found for {species_name}"
                    
                    # Create artifact
                    await process.log("Creating artifact with synonym data...")
                    await process.create_artifact(
                        mimetype="application/json",
                        description=f"Synonyms for {species_name} (AphiaID: {aphia_id}) - {len(synonyms)} total",
                        uris=[api_url],
                        metadata={
                            "aphia_id": aphia_id, 
                            "count": len(synonyms),
                            "species": species_name
                        }
                    )
                    
                    # Build summary
                    samples = [s.get('scientificname', 'Unknown') for s in synonyms[:3] if isinstance(s, dict)]
                    await process.log(f"Complete. Sample synonyms: {', '.join(samples)}")
                    
                    return f"Found {len(synonyms)} synonyms for {species_name}. Examples: {', '.join(samples)}. Full data in artifact."
                        
                except Exception as e:
                    await process.log(f"ERROR: {type(e).__name__}: {str(e)}")
                    return f"Error retrieving synonyms: {str(e)}"
        
        tools = [get_species_synonyms, abort, finish]
        
        # Execute agent
        async with context.begin_process("Processing request") as process:
            await process.log(f"User query: '{request}'")
            
            if params.species_names:
                await process.log(f"🐠 Species context: {', '.join(params.species_names)}")
            
            await process.log("Initializing LangChain ReAct agent...")
            llm = ChatOpenAI(model="gpt-4o-mini")
            system_prompt = self._make_system_prompt(params.species_names, request)
            agent = create_react_agent(llm, tools)
            
            try:
                await process.log("Starting agent reasoning loop...")
                await agent.ainvoke({
                    "messages": [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=request)
                    ]
                })
                await process.log("Agent completed successfully")
                
            except Exception as e:
                await process.log(f"Agent error: {type(e).__name__}: {str(e)}")
                await context.reply(f"An error occurred: {str(e)}")
    
    def _make_system_prompt(self, species_names: list[str], user_request: str) -> str:
        """Generate system prompt"""
        species_context = f"\n\nSpecies: {', '.join(species_names)}" if species_names else ""
        
        return f"""\
                        You are a marine biology assistant with access to the WoRMS database.

                        Request: "{user_request}"{species_context}

                        Instructions:
                        - Use get_species_synonyms to retrieve synonym data
                        - Call finish() with a summary when done
                        - Call abort() if you cannot complete the request

                        Always create artifacts when retrieving data.
                        """


if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS Agent Server")
    print("=" * 60)
    print(f"URL: http://localhost:9999")
    print(f"Status: Synonyms tool ready")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)