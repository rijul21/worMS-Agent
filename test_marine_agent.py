import pytest
import json
from unittest.mock import AsyncMock, patch
from ichatbio.agent_response import ProcessBeginResponse, ProcessLogResponse, ArtifactResponse, DirectResponse
from marine_agent import MarineAgent
from worms_models import WoRMSRecord, WoRMSSynonym, WoRMSDistribution, WoRMSVernacular, WoRMSSource, MarineQueryModel, MarineParameters

# Sample WoRMS API response for testing
SAMPLE_WORMS_RESPONSE = [
    {
        "AphiaID": 206989,
        "scientificname": "Acropora cervicornis",
        "authority": "(Lamarck, 1816)",
        "status": "accepted",
        "rank": "Species",
        "family": "Acroporidae",
        "kingdom": "Animalia",
        "phylum": "Cnidaria",
        "class": "Anthozoa",
        "order": "Scleractinia",
        "genus": "Acropora",
        "isMarine": True
    }
]

# Sample vernacular names response
SAMPLE_VERNACULAR_RESPONSE = [
    {"AphiaID": 206989, "vernacular": "staghorn coral", "language": "English"},
    {"AphiaID": 206989, "vernacular": "corail corne de cerf", "language": "French"}
]

# Sample synonyms response
SAMPLE_SYNONYMS_RESPONSE = [
    {"AphiaID": 206990, "scientificname": "Madrepora cervicornis", "authority": "Lamarck, 1816"},
    {"AphiaID": 206991, "scientificname": "Acropora prolifera", "authority": "Lamarck, 1816"}
]

# Sample distribution response
SAMPLE_DISTRIBUTION_RESPONSE = [
    {"AphiaID": 206989, "locality": "Caribbean Sea", "country": "Multiple", "geographicalScale": "Region"},
    {"AphiaID": 206989, "locality": "Gulf of Mexico", "country": "United States", "geographicalScale": "Region"}
]

# Sample attributes response
SAMPLE_ATTRIBUTES_RESPONSE = [
    {"attribute_name": "Habitat", "value": "Coral reefs"},
    {"attribute_name": "Depth range", "value": "0-40 meters"}
]

# Sample sources response
SAMPLE_SOURCES_RESPONSE = [
    {"reference": "Lamarck, J.B. (1816). Histoire naturelle des animaux sans vertèbres.", "author": "Lamarck, J.B.", "year": "1816"},
    {"reference": "Veron, J.E.N. (2000). Corals of the World.", "author": "Veron, J.E.N.", "year": "2000"}
]

@pytest.mark.asyncio
async def test_marine_agent_complete_data(context, messages):
    """
    Test: Verify all data (taxonomic, vernacular names, synonyms, distributions, attributes, sources) is retrieved correctly for a scientific name query
    """
    agent = MarineAgent()
    
    mock_records = [WoRMSRecord(**SAMPLE_WORMS_RESPONSE[0])]
    
    async def mock_get_species_by_name(client, name):
        return mock_records
    
    async def mock_get_vernaculars_by_aphia_id(client, aphia_id):
        return [WoRMSVernacular(**record) for record in SAMPLE_VERNACULAR_RESPONSE]
    
    async def mock_get_synonyms_by_aphia_id(client, aphia_id):
        return [WoRMSSynonym(**record) for record in SAMPLE_SYNONYMS_RESPONSE]
    
    async def mock_get_distributions_by_aphia_id(client, aphia_id):
        return [WoRMSDistribution(**record) for record in SAMPLE_DISTRIBUTION_RESPONSE]
    
    async def mock_get_attributes_by_aphia_id(client, aphia_id):
        return SAMPLE_ATTRIBUTES_RESPONSE
    
    async def mock_get_sources_by_aphia_id(client, aphia_id):
        return [WoRMSSource(**record) for record in SAMPLE_SOURCES_RESPONSE]
    
    async def mock_extract_query_info(request):
        return MarineQueryModel(scientificname="Acropora cervicornis", common_name=None)
    
    with patch.object(agent.worms_client, 'get_species_by_name', mock_get_species_by_name), \
         patch.object(agent.worms_client, 'get_vernaculars_by_aphia_id', mock_get_vernaculars_by_aphia_id), \
         patch.object(agent.worms_client, 'get_synonyms_by_aphia_id', mock_get_synonyms_by_aphia_id), \
         patch.object(agent.worms_client, 'get_distributions_by_aphia_id', mock_get_distributions_by_aphia_id), \
         patch.object(agent.worms_client, 'get_attributes_by_aphia_id', mock_get_attributes_by_aphia_id), \
         patch.object(agent.worms_client, 'get_sources_by_aphia_id', mock_get_sources_by_aphia_id), \
         patch.object(agent, 'extract_query_info', mock_extract_query_info):
        params = MarineParameters(
            species_name="Acropora cervicornis",
            include_synonyms=True,
            include_distribution=True,
            include_vernaculars=True,
            include_sources=True
        )
        query = "Tell me about Acropora cervicornis"
        await agent.run(context, query, "get_marine_info", params)
    
    assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}: {messages}"
    assert isinstance(messages[0], ProcessBeginResponse), "First message should be ProcessBeginResponse"
    
    # Check for artifact response (standalone or in DirectResponse.text)
    artifact_response = next(
        (msg for msg in messages if isinstance(msg, ArtifactResponse)),
        next(
            (msg.text for msg in messages if isinstance(msg, DirectResponse) and isinstance(msg.text, ArtifactResponse)),
            None
        )
    )
    assert artifact_response, f"Expected ArtifactResponse (standalone or in DirectResponse), got {messages}"
    
    # Parse artifact content
    artifact_data = json.loads(artifact_response.content)
    assert artifact_data["aphia_id"] == 206989, "Expected aphia_id to match"
    assert artifact_data["scientific_name"] == "Acropora cervicornis", "Expected scientific_name to match"
    assert artifact_data["search_term"] == "Acropora cervicornis", "Expected search_term to match"
    assert artifact_data["species"]["AphiaID"] == 206989, "Expected species.AphiaID to match"
    assert artifact_data["species"]["scientificname"] == "Acropora cervicornis", "Expected species.scientificname to match"
    assert artifact_data["species"]["authority"] == "(Lamarck, 1816)", "Expected species.authority to match"
    assert artifact_data["species"]["isMarine"] is True, "Expected species.isMarine to match"
    assert artifact_data["vernaculars"] == [
        {"AphiaID": 206989, "vernacular": "staghorn coral", "language": "English"},
        {"AphiaID": 206989, "vernacular": "corail corne de cerf", "language": "French"}
    ], "Expected vernaculars to match"
    assert artifact_data["synonyms"] == [
        {"AphiaID": 206990, "scientificname": "Madrepora cervicornis", "authority": "Lamarck, 1816"},
        {"AphiaID": 206991, "scientificname": "Acropora prolifera", "authority": "Lamarck, 1816"}
    ], "Expected synonyms to match"
    assert artifact_data["distribution"] == [
        {"AphiaID": 206989, "locality": "Caribbean Sea", "country": "Multiple", "geographical_scale": "Region"},
        {"AphiaID": 206989, "locality": "Gulf of Mexico", "country": "United States", "geographical_scale": "Region"}
    ], "Expected distributions to match"
    assert artifact_data["sources"] == [
        {"reference": "Lamarck, J.B. (1816). Histoire naturelle des animaux sans vertèbres.", "author": "Lamarck, J.B.", "year": "1816"},
        {"reference": "Veron, J.E.N. (2000). Corals of the World.", "author": "Veron, J.E.N.", "year": "2000"}
    ], "Expected sources to match"
    assert artifact_data["attributes"] == [
        {"attribute_name": "Habitat", "value": "Coral reefs"},
        {"attribute_name": "Depth range", "value": "0-40 meters"}
    ], "Expected attributes to match"
    
    # Check metadata
    assert sorted([x for x in artifact_response.metadata["data_sources"] if x]) == [
        "attributes", "distributions", "sources", "species", "synonyms", "vernaculars"
    ], "Expected data_sources to include all endpoints"
    
    # Check success message
    success_found = any(
        isinstance(msg, DirectResponse) and isinstance(msg.text, str) and 
        "Found marine species data" in msg.text and
        "staghorn coral, corail corne de cerf" in msg.text and
        "Madrepora cervicornis, Acropora prolifera" in msg.text and
        "Caribbean Sea, Gulf of Mexico" in msg.text and
        "Attributes: 2 found" in msg.text and
        "Sources: 2 found" in msg.text
        for msg in messages
    )
    assert success_found, f"Expected success message with vernacular names, synonyms, distributions, attributes, and sources, got {messages}"

# @pytest.mark.asyncio
# async def test_marine_agent_common_name(context, messages):
#     """
#     Test: Verify agent handles common name queries correctly
#     """
#     agent = MarineAgent()
#     
#     mock_records = [WoRMSRecord(**SAMPLE_WORMS_RESPONSE[0])]
#     
#     async def mock_get_species_by_name(client, name):
#         return mock_records
#     
#     async def mock_get_vernaculars_by_aphia_id(client, aphia_id):
#         return [WoRMSVernacular(**record) for record in SAMPLE_VERNACULAR_RESPONSE]
#     
#     async def mock_get_synonyms_by_aphia_id(client, aphia_id):
#         return [WoRMSSynonym(**record) for record in SAMPLE_SYNONYMS_RESPONSE]
#     
#     async def mock_get_distributions_by_aphia_id(client, aphia_id):
#         return [WoRMSDistribution(**record) for record in SAMPLE_DISTRIBUTION_RESPONSE]
#     
#     async def mock_get_attributes_by_aphia_id(client, aphia_id):
#         return SAMPLE_ATTRIBUTES_RESPONSE
#     
#     async def mock_get_sources_by_aphia_id(client, aphia_id):
#         return [WoRMSSource(**record) for record in SAMPLE_SOURCES_RESPONSE]
#     
#     async def mock_extract_query_info(request):
#         return MarineQueryModel(scientificname=None, common_name="staghorn coral")
#     
#     with patch.object(agent.worms_client, 'get_species_by_name', mock_get_species_by_name), \
#          patch.object(agent.worms_client, 'get_vernaculars_by_aphia_id', mock_get_vernaculars_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_synonyms_by_aphia_id', mock_get_synonyms_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_distributions_by_aphia_id', mock_get_distributions_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_attributes_by_aphia_id', mock_get_attributes_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_sources_by_aphia_id', mock_get_sources_by_aphia_id), \
#          patch.object(agent, 'extract_query_info', mock_extract_query_info):
#         params = MarineParameters(
#             species_name="staghorn coral",
#             include_synonyms=True,
#             include_distribution=True,
#             include_vernaculars=True,
#             include_sources=True
#         )
#         query = "Tell me about staghorn coral"
#         await agent.run(context, query, "get_marine_info", params)
#     
#     assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}: {messages}"
#     assert isinstance(messages[0], ProcessBeginResponse), "First message should be ProcessBeginResponse"
#     
#     # Check for artifact response (standalone or in DirectResponse.text)
#     artifact_response = next(
#         (msg for msg in messages if isinstance(msg, ArtifactResponse)),
#         next(
#             (msg.text for msg in messages if isinstance(msg, DirectResponse) and isinstance(msg.text, ArtifactResponse)),
#             None
#         )
#     )
#     assert artifact_response, f"Expected ArtifactResponse (standalone or in DirectResponse), got {messages}"
#     
#     # Parse artifact content
#     artifact_data = json.loads(artifact_response.content)
#     assert artifact_data["aphia_id"] == 206989, "Expected aphia_id to match"
#     assert artifact_data["scientific_name"] == "Acropora cervicornis", "Expected scientific_name to match"
#     assert artifact_data["search_term"] == "staghorn coral", "Expected search_term to match"
#     assert artifact_data["species"]["AphiaID"] == 206989, "Expected species.AphiaID to match"
#     
#     # Check success message
#     success_found = any(
#         isinstance(msg, DirectResponse) and isinstance(msg.text, str) and 
#         "Found marine species data for Acropora cervicornis" in msg.text and
#         "staghorn coral, corail corne de cerf" in msg.text
#         for msg in messages
#     )
#     assert success_found, f"Expected success message for common name query, got {messages}"

# @pytest.mark.asyncio
# async def test_marine_agent_invalid_species(context, messages):
#     """
#     Test: Verify agent handles invalid species names gracefully
#     """
#     agent = MarineAgent()
#     
#     async def mock_get_species_by_name(client, name):
#         return None
#     
#     async def mock_extract_query_info(request):
#         return MarineQueryModel(scientificname="Invalidus species", common_name=None)
#     
#     with patch.object(agent.worms_client, 'get_species_by_name', mock_get_species_by_name), \
#          patch.object(agent, 'extract_query_info', mock_extract_query_info):
#         params = MarineParameters(
#             species_name="Invalidus species",
#             include_synonyms=True,
#             include_distribution=True,
#             include_vernaculars=True,
#             include_sources=True
#         )
#         query = "Tell me about Invalidus species"
#         await agent.run(context, query, "get_marine_info", params)
#     
#     assert len(messages) >= 2, f"Expected at least 2 messages, got {len(messages)}: {messages}"
#     assert isinstance(messages[0], ProcessBeginResponse), "First message should be ProcessBeginResponse"
#     
#     error_found = any(
#         isinstance(msg, DirectResponse) and isinstance(msg.text, str) and 
#         "No marine species found matching 'Invalidus species'" in msg.text
#         for msg in messages
#     )
#     assert error_found, f"Expected error message for invalid species, got {messages}"

# @pytest.mark.asyncio
# async def test_marine_agent_partial_data(context, messages):
#     """
#     Test: Verify agent handles cases where some data types are missing
#     """
#     agent = MarineAgent()
#     
#     mock_records = [WoRMSRecord(**SAMPLE_WORMS_RESPONSE[0])]
#     
#     async def mock_get_species_by_name(client, name):
#         return mock_records
#     
#     async def mock_get_vernaculars_by_aphia_id(client, aphia_id):
#         return None
#     
#     async def mock_get_synonyms_by_aphia_id(client, aphia_id):
#         return None
#     
#     async def mock_get_distributions_by_aphia_id(client, aphia_id):
#         return [WoRMSDistribution(**SAMPLE_DISTRIBUTION_RESPONSE[0])]
#     
#     async def mock_get_attributes_by_aphia_id(client, aphia_id):
#         return SAMPLE_ATTRIBUTES_RESPONSE
#     
#     async def mock_get_sources_by_aphia_id(client, aphia_id):
#         return [WoRMSSource(**SAMPLE_SOURCES_RESPONSE[0])]
#     
#     async def mock_extract_query_info(request):
#         return MarineQueryModel(scientificname="Acropora cervicornis", common_name=None)
#     
#     with patch.object(agent.worms_client, 'get_species_by_name', mock_get_species_by_name), \
#          patch.object(agent.worms_client, 'get_vernaculars_by_aphia_id', mock_get_vernaculars_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_synonyms_by_aphia_id', mock_get_synonyms_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_distributions_by_aphia_id', mock_get_distributions_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_attributes_by_aphia_id', mock_get_attributes_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_sources_by_aphia_id', mock_get_sources_by_aphia_id), \
#          patch.object(agent, 'extract_query_info', mock_extract_query_info):
#         params = MarineParameters(
#             species_name="Acropora cervicornis",
#             include_synonyms=True,
#             include_distribution=True,
#             include_vernaculars=True,
#             include_sources=True
#         )
#         query = "Tell me about Acropora cervicornis"
#         await agent.run(context, query, "get_marine_info", params)
#     
#     assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}: {messages}"
#     assert isinstance(messages[0], ProcessBeginResponse), "First message should be ProcessBeginResponse"
#     
#     # Check for artifact response (standalone or in DirectResponse.text)
#     artifact_response = next(
#         (msg for msg in messages if isinstance(msg, ArtifactResponse)),
#         next(
#             (msg.text for msg in messages if isinstance(msg, DirectResponse) and isinstance(msg.text, ArtifactResponse)),
#             None
#         )
#     )
#     assert artifact_response, f"Expected ArtifactResponse (standalone or in DirectResponse), got {messages}"
#     
#     # Parse artifact content
#     artifact_data = json.loads(artifact_response.content)
#     assert artifact_data["aphia_id"] == 206989, "Expected aphia_id to match"
#     assert artifact_data["scientific_name"] == "Acropora cervicornis", "Expected scientific_name to match"
#     assert artifact_data["vernaculars"] is None, "Expected vernaculars to be None"
#     assert artifact_data["synonyms"] is None, "Expected synonyms to be None"
#     assert len(artifact_data["distribution"]) == 1, "Expected one distribution"
#     assert len(artifact_data["sources"]) == 1, "Expected one source"
#     assert len(artifact_data["attributes"]) == 2, "Expected two attributes"
#     
#     # Check success message
#     success_found = any(
#         isinstance(msg, DirectResponse) and isinstance(msg.text, str) and 
#         "Found marine species data" in msg.text and
#         "Common names: None" in msg.text and
#         "Synonyms: None" in msg.text and
#         "Distributions: Caribbean Sea" in msg.text and
#         "Attributes: 2 found" in msg.text and
#         "Sources: 1 found" in msg.text
#         for msg in messages
#     )
#     assert success_found, f"Expected success message with partial data, got {messages}"

# @pytest.mark.asyncio
# async def test_marine_agent_disabled_parameters(context, messages):
#     """
#     Test: Verify agent handles disabled parameters correctly
#     """
#     agent = MarineAgent()
#     
#     mock_records = [WoRMSRecord(**SAMPLE_WORMS_RESPONSE[0])]
#     
#     async def mock_get_species_by_name(client, name):
#         return mock_records
#     
#     async def mock_get_attributes_by_aphia_id(client, aphia_id):
#         return SAMPLE_ATTRIBUTES_RESPONSE
#     
#     async def mock_extract_query_info(request):
#         return MarineQueryModel(scientificname="Acropora cervicornis", common_name=None)
#     
#     with patch.object(agent.worms_client, 'get_species_by_name', mock_get_species_by_name), \
#          patch.object(agent.worms_client, 'get_vernaculars_by_aphia_id', AsyncMock(return_value=None)), \
#          patch.object(agent.worms_client, 'get_synonyms_by_aphia_id', AsyncMock(return_value=None)), \
#          patch.object(agent.worms_client, 'get_distributions_by_aphia_id', AsyncMock(return_value=None)), \
#          patch.object(agent.worms_client, 'get_attributes_by_aphia_id', mock_get_attributes_by_aphia_id), \
#          patch.object(agent.worms_client, 'get_sources_by_aphia_id', AsyncMock(return_value=None)), \
#          patch.object(agent, 'extract_query_info', mock_extract_query_info):
#         params = MarineParameters(
#             species_name="Acropora cervicornis",
#             include_synonyms=False,
#             include_distribution=False,
#             include_vernaculars=False,
#             include_sources=False
#         )
#         query = "Tell me about Acropora cervicornis"
#         await agent.run(context, query, "get_marine_info", params)
#     
#     assert len(messages) >= 3, f"Expected at least 3 messages, got {len(messages)}: {messages}"
#     assert isinstance(messages[0], ProcessBeginResponse), "First message should be ProcessBeginResponse"
#     
#     # Check for artifact response (standalone or in DirectResponse.text)
#     artifact_response = next(
#         (msg for msg in messages if isinstance(msg, ArtifactResponse)),
#         next(
#             (msg.text for msg in messages if isinstance(msg, DirectResponse) and isinstance(msg.text, ArtifactResponse)),
#             None
#         )
#     )
#     assert artifact_response, f"Expected ArtifactResponse (standalone or in DirectResponse), got {messages}"
#     
#     # Parse artifact content
#     artifact_data = json.loads(artifact_response.content)
#     assert artifact_data["aphia_id"] == 206989, "Expected aphia_id to match"
#     assert artifact_data["scientific_name"] == "Acropora cervicornis", "Expected scientific_name to match"
#     assert artifact_data["vernaculars"] is None, "Expected vernaculars to be None"
#     assert artifact_data["synonyms"] is None, "Expected synonyms to be None"
#     assert artifact_data["distribution"] is None, "Expected distribution to be None"
#     assert artifact_data["sources"] is None, "Expected sources to be None"
#     assert len(artifact_data["attributes"]) == 2, "Expected two attributes"
#     
#     # Check metadata
#     assert sorted([x for x in artifact_response.metadata["data_sources"] if x]) == [
#         "attributes", "species"
#     ], "Expected data_sources to include only species and attributes"
#     
#     # Check success message
#     success_found = any(
#         isinstance(msg, DirectResponse) and isinstance(msg.text, str) and 
#         "Found marine species data" in msg.text and
#         "Common names: None" in msg.text and
#         "Synonyms: None" in msg.text and
#         "Distributions: None" in msg.text and
#         "Attributes: 2 found" in msg.text and
#         "Sources: 0 found" in msg.text
#         for msg in messages
#     )
#     assert success_found, f"Expected success message with disabled parameters, got {messages}"