import pytest
from ichatbio.agent_response import DirectResponse, ProcessBeginResponse, ProcessLogResponse, ArtifactResponse
from marine_agent import MarineAgent
from worms_models import MarineParameters


@pytest.mark.asyncio
async def test_marine_agent_basic(context, messages):
    """Test basic marine species lookup"""
    params = MarineParameters(
        include_synonyms=True,
        include_distribution=True,
        include_vernaculars=True,
        include_classification=True,
        include_children=True
    )
    
    await MarineAgent().run(context, "Tell me about killer whale", "get_marine_info", params)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    #basic messages
    assert len(messages) >= 4
    
    #first message should be process begin
    assert isinstance(messages[0], ProcessBeginResponse)
    assert messages[0].summary == "Analyzing marine species request"
    
    #should have some process log messages
    log_messages = [msg for msg in messages if isinstance(msg, ProcessLogResponse)]
    assert len(log_messages) > 0
    
    #should create an artifact
    artifacts = [msg for msg in messages if isinstance(msg, ArtifactResponse)]
    assert len(artifacts) == 1
    assert artifacts[0].mimetype == "application/json"
    
    #should end with a direct response
    assert isinstance(messages[-1], DirectResponse)
    assert len(messages[-1].text) > 0


@pytest.mark.asyncio
async def test_marine_agent_scientific_name(context, messages):
    """Test marine species lookup with scientific name"""
    params = MarineParameters(
        include_synonyms=False,
        include_distribution=False,
        include_vernaculars=True,
        include_classification=False,
        include_children=False
    )
    
    await MarineAgent().run(context, "What is Carcharodon carcharias?", "get_marine_info", params)
    
    messages: list[ProcessBeginResponse | ProcessLogResponse | ArtifactResponse | DirectResponse]
    
    #basic structure check
    assert len(messages) >= 3
    
    #should start with process begin
    assert isinstance(messages[0], ProcessBeginResponse)
    
    #should create exactly one artifact
    artifacts = [msg for msg in messages if isinstance(msg, ArtifactResponse)]
    assert len(artifacts) == 1
    assert artifacts[0].mimetype == "application/json"
    
    #should end with response
    assert isinstance(messages[-1], DirectResponse)