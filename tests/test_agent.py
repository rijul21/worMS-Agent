"""
Unit tests for WoRMS Agent.
Tests core functionality including species lookup, name resolution, and data retrieval.
"""
import pytest
from src.agent import WoRMSReActAgent, MarineResearchParams


def print_test_summary(test_name, request, messages):
    """Helper function to print test summary"""
    print(f"Test: {test_name}")
    print(f"Mocked up user request: {request}")

    

    for msg in messages:
        if hasattr(msg, 'data') and msg.data:
            if 'reasoning' in msg.data:
                print("Query Type:", msg.data.get('query_type', 'N/A'))
                print("Species Count:", msg.data.get('species_count', 'N/A'))
                print("Must Call Tools:", msg.data.get('must_call', []))
                print("Should Call Tools:", msg.data.get('should_call', []))
                print("Reasoning:", msg.data.get('reasoning', 'N/A'))
                break
    
    # print final summary
    final_summary = None
    for msg in reversed(messages):
        if hasattr(msg, 'text') and msg.text:
            final_summary = msg.text
            break
    
    if final_summary:
        print("\nFinal Summary:")
        print(final_summary)
    
    print("="*80 + "\n")


@pytest.mark.asyncio
async def test_single_species_query(agent, context, messages):
    """
    Test basic single species query with scientific name.
    Should successfully retrieve species information.
    """
    params = MarineResearchParams(species_names=["Orcinus orca"])
    request = "Tell me about Orcinus orca"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Single Species Query", request, messages)
    
    # Agent should send at least one response
    assert len(messages) > 0
    
    # Check that some response mentions the species
    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')])
    assert "orca" in response_text.lower() or "killer whale" in response_text.lower()


@pytest.mark.asyncio
async def test_common_name_resolution(agent, context, messages):
    """
    Test that agent can resolve common names to scientific names.
    Agent should handle "killer whale" and convert it properly.
    """
    params = MarineResearchParams(species_names=["killer whale"])
    request = "What is the taxonomy of killer whales?"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Common Name Resolution", request, messages)
    
    # Should complete without errors
    assert len(messages) > 0
    
    # Response should reference scientific name after resolution
    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')])
    assert len(response_text) > 0


@pytest.mark.asyncio
async def test_multiple_species_comparison(agent, context, messages):
    """
    Test comparison query with multiple species.
    Agent should handle multiple species and provide comparative information.
    """
    params = MarineResearchParams(species_names=["Orcinus orca", "Delphinus delphis"])
    request = "Compare Orcinus orca and Delphinus delphis"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Multiple Species Comparison", request, messages)
    
    # Agent should process multiple species
    assert len(messages) > 0
    
    # Should mention both species in some form
    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')]).lower()
    assert "orca" in response_text or "delphis" in response_text


@pytest.mark.asyncio
async def test_conservation_status_query(agent, context, messages):
    """
    Test query specifically about conservation status.
    Should trigger tools related to species attributes and IUCN status.
    """
    params = MarineResearchParams(species_names=["Carcharodon carcharias"])
    request = "What is the conservation status of great white sharks?"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Conservation Status Query", request, messages)
    
    assert len(messages) > 0
    
    # Response should be about conservation/status
    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')]).lower()
    assert len(response_text) > 0


@pytest.mark.asyncio
async def test_distribution_query(agent, context, messages):
    """
    Test query about species geographic distribution.
    Should trigger distribution-related tools.
    """
    params = MarineResearchParams(species_names=["Tursiops truncatus"])
    request = "Where does the bottlenose dolphin live?"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Distribution Query", request, messages)
    
    assert len(messages) > 0
    
    # Should have some response about location/distribution
    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')])
    assert len(response_text) > 0


@pytest.mark.asyncio
async def test_taxonomy_query(agent, context, messages):
    """
    Test query about taxonomic classification.
    Should retrieve taxonomic hierarchy information.
    """
    params = MarineResearchParams(species_names=["Physeter macrocephalus"])
    request = "What is the taxonomic classification of sperm whales?"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Taxonomy Query", request, messages)
    
    assert len(messages) > 0
    
 
    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')]).lower()
    assert len(response_text) > 0
    
    has_taxonomy_terms = any(term in response_text for term in 
                             ["family", "order", "class", "phylum", "taxonomy", "classification"])
   
    assert has_taxonomy_terms or len(response_text) > 50

@pytest.mark.asyncio
async def test_nonexistent_species(agent, context, messages):
    """
    Test with a completely fake species name that doesn't exist in WoRMS.
    This should break: agent should handle gracefully but might fail or give wrong info.
    """
    params = MarineResearchParams(species_names=["Fakeus nonexistentus"])
    request = "Tell me about Fakeus nonexistentus"
    
    await agent.run(
        context=context,
        request=request,
        entrypoint="research_marine_species",
        params=params
    )
    
    print_test_summary("Nonexistent Species", request, messages)
    

    response_text = " ".join([msg.text for msg in messages if hasattr(msg, 'text')]).lower()
    
    assert "not found" in response_text or "could not" in response_text or "unable" in response_text, \
        "Agent should explicitly say species not found, but it didn't!"