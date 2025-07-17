import pytest
import pytest_asyncio
import respx
import httpx
from unittest.mock import AsyncMock, patch
from marine_agent import MarineAgent, Config
from worms_models import MarineQueryModel, MarineParameters
from ichatbio.agent_response import ProcessBeginResponse, ProcessLogResponse, DirectResponse

@pytest.mark.asyncio
async def test_marine_agent_run_valid_request(context, messages):
    # Instantiate the agent
    agent = MarineAgent()

    # Mock extract_query_info to return a valid MarineQueryModel (bypassing Groq API)
    with patch.object(agent, "extract_query_info", new=AsyncMock()) as mock_extract:
        mock_extract.return_value = MarineQueryModel(
            scientificname="Orcinus orca",
            common_name="killer whale"
        )

        # Mock WoRMS API for species data
        with respx.mock:
            respx.get(
                f"{Config.WORMS_BASE_URL}/AphiaRecordsByName/Orcinus%20orca?like=false&marine_only=true"
            ).mock(
                return_value=httpx.Response(
                    200,
                    json=[{
                        "AphiaID": 137102,
                        "scientificname": "Orcinus orca",
                        "authority": "(Linnaeus, 1758)",
                        "status": "accepted",
                        "rank": "Species",
                        "isMarine": 1,
                        "isBrackish": 0,
                        "isFreshwater": 0,
                        "isTerrestrial": 0,
                        "isExtinct": 0
                    }]
                )
            )
            # Mock other WoRMS endpoints to return empty responses for simplicity
            respx.get(
                f"{Config.WORMS_BASE_URL}/AphiaVernacularsByAphiaID/137102"
            ).mock(return_value=httpx.Response(200, json=[]))
            respx.get(
                f"{Config.WORMS_BASE_URL}/AphiaSynonymsByAphiaID/137102"
            ).mock(return_value=httpx.Response(200, json=[]))
            respx.get(
                f"{Config.WORMS_BASE_URL}/AphiaDistributionsByAphiaID/137102"
            ).mock(return_value=httpx.Response(200, json=[]))
            respx.get(
                f"{Config.WORMS_BASE_URL}/AphiaAttributesByAphiaID/137102"
            ).mock(return_value=httpx.Response(200, json=[]))
            respx.get(
                f"{Config.WORMS_BASE_URL}/AphiaSourcesByAphiaID/137102"
            ).mock(return_value=httpx.Response(200, json=[]))

            # Run the agent
            params = MarineParameters(
                species_name="killer whale",
                include_synonyms=True,
                include_distribution=True,
                include_vernaculars=True,
                include_sources=True
            )
            await agent.run(context, "Tell me about the killer whale", "get_marine_info", params)

    # Verify the messages
    assert len(messages) == 7, f"Expected 7 messages, got {len(messages)}"

    # Check ProcessBeginResponse
    assert isinstance(messages[0], ProcessBeginResponse)
    assert messages[0].summary == "Analyzing marine species request"

    # Check ProcessLogResponses
    process_logs = [m for m in messages if isinstance(m, ProcessLogResponse)]
    assert len(process_logs) == 4, f"Expected 4 ProcessLogResponses, got {len(process_logs)}"
    assert process_logs[0].text == "Identified marine species: Orcinus orca"
    assert process_logs[0].data == {
        "scientific_name": "Orcinus orca",
        "common_name": "killer whale"
    }
    assert process_logs[1].text == "Searching WoRMS by name: Orcinus orca"
    assert process_logs[2].text == "Found species: Orcinus orca (AphiaID: 137102)"
    assert process_logs[3].text == "Retrieving marine species data, vernacular names, synonyms, distributions, attributes, and sources"

    # Check DirectResponses
    direct_responses = [m for m in messages if isinstance(m, DirectResponse)]
    assert len(direct_responses) == 2, f"Expected 2 DirectResponses, got {len(direct_responses)}"
    # Check artifact data in first DirectResponse
    assert direct_responses[0].text == "Marine species data for Orcinus orca"
    assert direct_responses[0].data == {
        "mimetype": "application/json",
        "description": "Marine species data for Orcinus orca",
        "uris": [
            "https://www.marinespecies.org/aphia.php?p=taxdetails&id=137102",
            "https://www.marinespecies.org/rest/AphiaRecordsByAphiaID/137102",
            "https://www.marinespecies.org/rest/AphiaVernacularsByAphiaID/137102",
            "https://www.marinespecies.org/rest/AphiaSynonymsByAphiaID/137102",
            "https://www.marinespecies.org/rest/AphiaDistributionsByAphiaID/137102",
            "https://www.marinespecies.org/rest/AphiaAttributesByAphiaID/137102",
            "https://www.marinespecies.org/rest/AphiaSourcesByAphiaID/137102"
        ],
        "metadata": {
            "aphia_id": 137102,
            "scientific_name": "Orcinus orca",
            "search_term": "Orcinus orca",  # Updated to match run method
            "data_sources": ["species", "vernaculars", "synonyms", "distributions", "attributes", "sources"],
            "retrieved_at": direct_responses[0].data["metadata"]["retrieved_at"]  # Dynamic timestamp
        }
    }
    # Check text summary in second DirectResponse
    assert "Found marine species data for Orcinus orca (AphiaID: 137102)" in direct_responses[1].text
    assert "Common names: None" in direct_responses[1].text
    assert "Synonyms: None" in direct_responses[1].text
    assert "Distributions: None" in direct_responses[1].text
    assert "Conservation status: None" in direct_responses[1].text
    assert "Sources: 0 found" in direct_responses[1].text
    assert "The artifact contains detailed taxonomic information" in direct_responses[1].text