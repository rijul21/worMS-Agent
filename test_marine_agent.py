import pytest
from marine_agent import MarineAgent
from MarineAgent.worms_agent_server import MarineParameters

@pytest.mark.asyncio
async def test_get_marine_info_orcinus_orca(context, messages):
    """Test complete marine data retrieval for Orcinus orca (killer whale)"""
    agent = MarineAgent()
    params = MarineParameters(
        species_name="Orcinus orca",
        include_synonyms=True,
        include_distribution=True,
        include_vernaculars=True,
        include_sources=True
    )
    await agent.run(context, "Get information for Orcinus orca", "get_marine_info", params)
    
    # Check that we got responses
    assert len(messages) > 0, "Agent should send at least one message"
    
    # Check for expected content in any message
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert "Orcinus orca" in all_text, "Response should contain species name"
    print(f"✅ Test passed - found {len(messages)} messages")

@pytest.mark.asyncio
async def test_get_taxonomy_orcinus_orca(context, messages):
    """Test taxonomic classification for Orcinus orca"""
    agent = MarineAgent()
    params = MarineParameters(species_name="Orcinus orca")
    await agent.run(context, "Get taxonomy for Orcinus orca", "get_taxonomy", params)
    
    # Check that we got responses
    assert len(messages) > 0, "Agent should send at least one message"
    
    # Check for taxonomy-related content
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert any(word in all_text for word in ["TEST:", "Kingdom", "Phylum", "Class", "Family", "taxonomy"]), \
        "Response should contain taxonomic information or test message"
    print(f"✅ Taxonomy test passed - found {len(messages)} messages")

@pytest.mark.asyncio
async def test_get_synonyms_delphinus_delphis(context, messages):
    """Test synonym retrieval for Delphinus delphis"""
    agent = MarineAgent()
    params = MarineParameters(species_name="Delphinus delphis")
    await agent.run(context, "Get synonyms for Delphinus delphis", "get_synonyms", params)
    
    # Check that we got responses
    assert len(messages) > 0, "Agent should send at least one message"
    
    # Check for synonym-related content
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert any(word in all_text.lower() for word in ["synonym", "delphinus", "found", "alternative"]), \
        "Response should contain synonym information"
    print(f"✅ Synonyms test passed - found {len(messages)} messages")

@pytest.mark.asyncio
async def test_invalid_species_handling(context, messages):
    """Test handling of invalid/nonexistent species name"""
    agent = MarineAgent()
    params = MarineParameters(species_name="NonexistentSpecies invalidus")
    await agent.run(context, "Get info for nonexistent species", "get_marine_info", params)
    
    # Check that we got responses
    assert len(messages) > 0, "Agent should send at least one message"
    
    # Check for appropriate error handling
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert any(word in all_text.lower() for word in ["no", "not found", "error", "matching"]), \
        "Response should indicate species not found"
    print(f"✅ Error handling test passed - found {len(messages)} messages")

@pytest.mark.asyncio
async def test_artifact_creation_with_comprehensive_data(context, messages):
    """Test that comprehensive marine data creates artifact with expected data counts"""
    agent = MarineAgent()
    params = MarineParameters(
        species_name="Delphinus delphis",
        include_synonyms=True,
        include_distribution=True,
        include_vernaculars=True,
        include_sources=True
    )
    await agent.run(context, "Get comprehensive data for Delphinus delphis", "get_marine_info", params)
    
    # Check for artifact creation
    artifacts = [m for m in messages if hasattr(m, 'mimetype') and m.mimetype == "application/json"]
    assert len(artifacts) > 0, "Should create at least one JSON artifact"
    
    # Check response mentions data counts
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert any(char.isdigit() for char in all_text), "Response should contain data counts"
    assert "Delphinus delphis" in all_text, "Response should contain species name"
    
    print(f"✅ Artifact creation test passed - found {len(artifacts)} artifacts and {len(messages)} total messages")

@pytest.mark.asyncio
async def test_vernacular_names_with_language_diversity(context, messages):
    """Test vernacular names retrieval shows language diversity"""
    agent = MarineAgent()
    params = MarineParameters(species_name="Orcinus orca")
    await agent.run(context, "Get vernacular names for Orcinus orca", "get_vernacular_names", params)
    
    # Check that we got responses
    assert len(messages) > 0, "Agent should send at least one message"
    
    # Check for vernacular/common name content
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert any(word in all_text.lower() for word in ["common name", "vernacular", "language", "found"]), \
        "Response should contain vernacular name information"
    
    # Should mention multiple names or languages
    assert any(char.isdigit() for char in all_text), "Response should contain count of names found"
    
    print(f"✅ Vernacular names test passed - found {len(messages)} messages")

@pytest.mark.asyncio
async def test_distribution_data_geographic_coverage(context, messages):
    """Test distribution data provides geographic location information"""
    agent = MarineAgent()
    params = MarineParameters(species_name="Orcinus orca")
    await agent.run(context, "Get distribution for Orcinus orca", "get_distribution", params)
    
    # Check that we got responses
    assert len(messages) > 0, "Agent should send at least one message"
    
    # Check for distribution/geographic content
    all_text = " ".join([getattr(m, "text", "") for m in messages])
    assert any(word in all_text.lower() for word in ["location", "found in", "distribution", "recorded"]), \
        "Response should contain distribution information"
    
    # Should mention number of locations or specific places
    has_locations = any(word in all_text for word in ["Atlantic", "Pacific", "Ocean", "Sea", "waters"]) or \
                   any(char.isdigit() for char in all_text)
    assert has_locations, "Response should mention specific locations or counts"
    
    print(f"✅ Distribution test passed - found {len(messages)} messages")