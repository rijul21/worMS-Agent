import pytest
from ichatbio.agent_response import DirectResponse, ProcessBeginResponse, ProcessLogResponse, ArtifactResponse
from marine_agent import MarineAgent


@pytest.mark.asyncio
async def test_marine_agent_scientific_name(context, messages):
    """Test marine agent with scientific name - basic functionality"""
    await MarineAgent().run(context, "I need information about Orcinus orca", "get_marine_info", None)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    # Test essential flow: should have process begin, species identification, and artifact creation
    assert len(messages) >= 4  # At minimum: begin, identify, artifact, reply
    assert isinstance(messages[0], ProcessBeginResponse)
    assert messages[0].summary == 'Analyzing marine species request'
    
    # Should identify the species
    assert any('Orcinus orca' in msg.text for msg in messages if isinstance(msg, ProcessLogResponse))
    
    # Should create an artifact
    artifact_msgs = [msg for msg in messages if isinstance(msg, ArtifactResponse)]
    assert len(artifact_msgs) == 1
    assert artifact_msgs[0].mimetype == 'application/json'
    assert 'Orcinus orca' in artifact_msgs[0].description
    assert artifact_msgs[0].metadata['aphia_id'] == 137102
    
    # Should end with a direct response
    assert isinstance(messages[-1], DirectResponse)
    assert 'Found marine species data for Orcinus orca' in messages[-1].text


@pytest.mark.asyncio
async def test_marine_agent_common_name(context, messages):
    """Test marine agent with common name - basic functionality"""
    await MarineAgent().run(context, "Tell me about great white sharks", "get_marine_info", None)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    # Test essential flow
    assert len(messages) >= 4
    assert isinstance(messages[0], ProcessBeginResponse)
    
    # Should identify the species (either by common or scientific name)
    assert any('great white' in msg.text.lower() or 'carcharodon carcharias' in msg.text.lower() 
              for msg in messages if isinstance(msg, ProcessLogResponse))
    
    # Should create an artifact
    artifact_msgs = [msg for msg in messages if isinstance(msg, ArtifactResponse)]
    assert len(artifact_msgs) == 1
    assert artifact_msgs[0].mimetype == 'application/json'
    assert artifact_msgs[0].metadata['aphia_id'] == 105838
    
    # Should end with a direct response
    assert isinstance(messages[-1], DirectResponse)
    assert 'Found marine species data for Carcharodon carcharias' in messages[-1].text


@pytest.mark.asyncio
async def test_marine_agent_no_species_found(context, messages):
    """Test marine agent when no species is identified"""
    await MarineAgent().run(context, "Tell me about unicorns", "get_marine_info", None)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    # Should have minimal response when no species found
    assert len(messages) == 2
    assert isinstance(messages[0], ProcessBeginResponse)
    assert isinstance(messages[1], DirectResponse)
    assert 'No marine species identified' in messages[1].text


@pytest.mark.asyncio
async def test_marine_agent_species_not_in_worms(context, messages):
    """Test marine agent when species is identified but not found in WoRMS"""
    await MarineAgent().run(context, "Tell me about Fakeus marineus", "get_marine_info", None)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    # Should identify species but fail to find in WoRMS
    assert len(messages) >= 3
    assert isinstance(messages[0], ProcessBeginResponse)
    
    # Should identify the fake species
    assert any('Fakeus marineus' in msg.text for msg in messages if isinstance(msg, ProcessLogResponse))
    
    # Should end with not found message
    assert isinstance(messages[-1], DirectResponse)
    assert 'No marine species found matching' in messages[-1].text
    assert 'Fakeus marineus' in messages[-1].text


@pytest.mark.asyncio
async def test_marine_agent_artifact_structure(context, messages):
    """Test that the artifact contains expected structure and URIs"""
    await MarineAgent().run(context, "Information about blue whales", "get_marine_info", None)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    # Find the artifact
    artifact_msgs = [msg for msg in messages if isinstance(msg, ArtifactResponse)]
    assert len(artifact_msgs) == 1
    
    artifact = artifact_msgs[0]
    assert artifact.mimetype == 'application/json'
    assert len(artifact.uris) == 7  # Should have 7 WoRMS URIs
    
    # Check required metadata fields
    assert 'aphia_id' in artifact.metadata
    assert 'scientific_name' in artifact.metadata
    assert 'search_term' in artifact.metadata
    assert 'data_sources' in artifact.metadata
    
    # Check that URIs are properly formatted
    for uri in artifact.uris:
        assert uri.startswith('https://www.marinespecies.org/')
        assert str(artifact.metadata['aphia_id']) in uri