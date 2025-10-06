"""
WoRMS ReAct Agent - Intelligent marine species research assistant
Uses LangChain to enable natural language queries and multi-step research
"""

from typing import override
from pydantic import BaseModel, Field
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import run_agent_server
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain.tools import tool
from langchain_openai import ChatOpenAI
import langchain.agents
import dotenv
import asyncio

from .worms_api import WoRMS

dotenv.load_dotenv()

# Single flexible entrypoint parameter
class MarineResearchParams(BaseModel):
    """Parameters for marine species research requests"""
    species_names: list[str] = Field(
        default=[],
        description="Scientific names of marine species to research (optional - can be extracted from natural language)",
        examples=[["Orcinus orca"], ["Orcinus orca", "Delphinus delphis"]]
    )

AGENT_DESCRIPTION = """\
Intelligent marine species research assistant using the WoRMS (World Register of Marine Species) database.

Capabilities:
- Look up taxonomic information for single or multiple species
- Compare species data (distribution, taxonomy, synonyms, etc.)
- Retrieve synonyms, vernacular names, and literature sources
- Analyze taxonomic relationships and classifications
- Aggregate information across multiple data types

Simply describe what you want to know in natural language, like:
- "What are the synonyms for Orcinus orca?"
- "Compare the distribution of killer whales and dolphins"
- "Give me all taxonomic information for Tursiops truncatus"
"""


class WoRMSReActAgent(IChatBioAgent):
    def __init__(self):
        self.worms_logic = WoRMS()
        
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="WoRMS Research Agent (ReAct)",
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
        """
        Main entry point - builds and executes the ReAct agent loop
        """
        
        # Create control tools
        @tool(return_direct=True)
        async def abort(reason: str):
            """
            Call this if you cannot fulfill the user's request.
            Provide a clear explanation of why.
            """
            await context.reply(f"I couldn't complete your request: {reason}")
        
        @tool(return_direct=True)
        async def finish(summary: str):
            """
            Call this when you have successfully completed the user's request.
            Provide a summary of what you found.
            """
            await context.reply(summary)
        
        # Build tools list (will add WoRMS tools in next step)
        tools = [
            abort,
            finish,
            # TODO: Add WoRMS operation tools here in next step
        ]
        
        # Create LangChain agent
        llm = ChatOpenAI(model="gpt-4o-mini", tool_choice="required")
        
        system_message = self._make_system_prompt(params.species_names, request)
        agent = langchain.agents.create_agent(
            model=llm,
            tools=tools,
            prompt=system_message,
        )
        
        # Execute agent with logging
        async with context.begin_process("Analyzing your marine species request") as process:
            await process.log(f"User request: {request}")
            if params.species_names:
                await process.log(f"Species context: {', '.join(params.species_names)}")
            
            try:
                await agent.ainvoke({
                    "messages": [
                        {"role": "user", "content": request}
                    ]
                })
            except Exception as e:
                await process.log(f"Agent error: {str(e)}")
                await context.reply(f"An error occurred: {str(e)}")
    
    def _make_system_prompt(self, species_names: list[str], user_request: str) -> str:
        """Generate system prompt to guide the agent's reasoning"""
        
        species_context = ""
        if species_names:
            species_context = f"\n\nSpecies mentioned: {', '.join(species_names)}"
        
        return f"""\
You are a marine biology research assistant with access to the WoRMS (World Register of Marine Species) database.

Your mission: Help users research marine species by retrieving and analyzing taxonomic data from WoRMS.

Current request: "{user_request}"{species_context}

Guidelines:
1. Break complex requests into steps
2. Always verify species exist in WoRMS before fetching detailed data
3. When comparing species, retrieve the same data type for all species
4. Create artifacts for data you retrieve so users can examine details
5. If a request is ambiguous, make reasonable assumptions and explain them
6. If you cannot complete a request, call abort() with a clear reason
7. When finished, call finish() with a comprehensive summary

Available in next step: Tools to search species, get synonyms, distribution, vernacular names, sources, taxonomic records, classifications, and child taxa.
"""


if __name__ == "__main__":
    agent = WoRMSReActAgent()
    print("=" * 60)
    print("WoRMS ReAct Agent Server")
    print("=" * 60)
    print(f"Starting at: http://localhost:9999")
    print(f"Agent: {agent.get_agent_card().name}")
    print("Status: Tools not yet implemented (placeholder)")
    print("=" * 60)
    run_agent_server(agent, host="0.0.0.0", port=9999)