# WoRMS iChatBio Agent

A comprehensive iChatBio agent that integrates the World Register of Marine Species (WoRMS) database with the iChatBio ecosystem. This agent enables natural language queries for marine species data, leveraging iChatBio's agent framework to provide structured taxonomic and ecological information through the A2A protocol.

## Tech Stack

* **Python 3.12+** (required for iChatBio SDK)
* **iChatBio SDK** (`ichatbio-sdk`) for agent framework and A2A protocol integration
* **Pydantic** for data modeling and validation
* **cloudscraper** for anti-bot HTTP requests
* **requests** for HTTP client functionality
* **PyYAML** for configuration management
* **asyncio** for asynchronous operations

## About iChatBio Integration

This agent is built using the **iChatBio SDK**, which provides a layer of abstraction over the [A2A protocol](https://github.com/google/a2a) while exposing iChatBio-specific capabilities. The integration offers several key advantages:

### iChatBio Framework Benefits
* **Structured Parameters**: Direct access to validated parameters without unreliable NLP parsing
* **Agent-to-Agent Communication**: Compatible with other A2A agents in the ecosystem
* **Artifact Creation**: Generate persistent, structured data artifacts with metadata
* **Process Logging**: Detailed workflow tracking and debugging capabilities
* **Response Context**: Rich context management for complex multi-step operations

### A2A Protocol Compliance
Because this agent uses the iChatBio SDK, it automatically supports:
* **Agent Discovery**: Self-describing capabilities through agent cards
* **Standardized Communication**: Consistent request/response patterns
* **Interoperability**: Communication with other A2A-compliant agents
* **Service Integration**: Access to iChatBio ecosystem services and shared storage

## Key Components

* **WoRMSAgent** - Main iChatBio agent class implementing IChatBioAgent interface
* **WoRMSiChatBioAgent** - Core workflow logic and API orchestration
* **WoRMS** - REST client for WoRMS API with cloudscraper integration
* **Parameter Models** - Pydantic models for type-safe API interactions

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/worms-ichatbio-agent.git
cd worms-ichatbio-agent
```

### 2. Install dependencies
```bash
pip install ichatbio-sdk
pip install -r requirements.txt
```

### 3. Set up configuration (optional)
Create an `env.yaml` file to customize the WoRMS API URL:
```yaml
WORMS_API_URL: "https://www.marinespecies.org/rest"
```

### 4. Run the agent server
```bash
python agent.py
```

The agent will start on `http://localhost:9999` with the following output:
```bash
Starting iChatBio agent server for 'WoRMS Marine Species Agent' at http://localhost:9999
Available endpoints:
  - get_synonyms: Get synonyms and alternative names for a marine species from WoRMS.
  - get_distribution: Get distribution data and geographic locations for a marine species from WoRMS.
  - get_vernacular_names: Get vernacular/common names for a marine species in different languages from WoRMS.
  - get_sources: Get literature sources and references for a marine species from WoRMS.
  - get_record: Get basic taxonomic record and classification for a marine species from WoRMS.
  - get_taxonomy: Get complete taxonomic classification hierarchy for a marine species from WoRMS.
  - get_marine_info: Get child taxa (subspecies, varieties, forms) for a marine species from WoRMS.
```

### 5. Test with main.py (optional)
```bash
python main.py
```

## API Endpoints

The agent provides 7 distinct endpoints that map to WoRMS REST API services:

### Species Search & ID Resolution
* `GET /AphiaRecordsByName/{scientific_name}` - Convert species name to AphiaID

### Core Data Endpoints
* `GET /AphiaSynonymsByAphiaID/{aphia_id}` - Species synonyms and alternative names
* `GET /AphiaDistributionsByAphiaID/{aphia_id}` - Geographic distribution data
* `GET /AphiaVernacularsByAphiaID/{aphia_id}` - Common names in multiple languages
* `GET /AphiaSourcesByAphiaID/{aphia_id}` - Literature sources and references
* `GET /AphiaRecordByAphiaID/{aphia_id}` - Basic taxonomic record
* `GET /AphiaClassificationByAphiaID/{aphia_id}` - Complete taxonomic hierarchy
* `GET /AphiaChildrenByAphiaID/{aphia_id}` - Child taxa (subspecies, varieties)

## Test Queries

The agent supports natural language queries that are automatically parsed and routed to appropriate endpoints. Here are comprehensive test examples:

### Synonyms & Alternative Names
```
"Get synonyms for Delphinus delphis"
"Find alternative names for great white shark"
"What are the synonyms of spinner dolphin?"
```
*Expected: Historical taxonomic names, junior synonyms, and alternative scientific nomenclature*

### Geographic Distribution
```
"Find distribution for Gadus morhua"
"Show me distribution data for bluefin tuna"
"Where is Tursiops truncatus found?"
```
*Expected: Geographic locations, countries, marine regions, and specific localities*

### Taxonomic Classification
```
"Get taxonomic classification for Thunnus thynnus"
"Get classification hierarchy for killer whale"
"Show me the taxonomy of Atlantic cod"
```
*Expected: Complete hierarchical classification from Kingdom to Species level*

### Basic Species Records
```
"Get basic record for Tursiops truncatus"
"Show me information about bowhead whale"
"Get species data for Salmo salar"
```
*Expected: Core taxonomic information, authority, status, and basic classification*

### Common Names (Vernacular)
```
"Get vernacular names for Orcinus orca"
"Get common names for Atlantic cod"
"What are the local names for bluefin tuna?"
```
*Expected: Common names in multiple languages with language codes*

### Child Taxa & Subspecies
```
"Get children for Thunnus"
"Get marine info for dolphin genus"
"Find child taxa for Balaenoptera"
```
*Expected: Subspecies, varieties, forms, and lower taxonomic ranks*

### Literature Sources
```
"Find sources for Salmo salar"
"Get literature references for bowhead whale"
"Show me research papers about great white sharks"
```
*Expected: Scientific publications, references, and bibliographic data*

### Complex Natural Language Queries
```
"Tell me about the distribution and common names of killer whales"
"What are all the tuna species and where are they found?"
"Show me the complete taxonomic information for marine mammals"
```
*Expected: Multi-endpoint responses with comprehensive data integration*

## File Structure

```
worms-ichatbio-agent/
├── agent.py              # Main iChatBio agent implementation
├── worms_api.py          # WoRMS API client with Pydantic models
├── main.py               # Optional testing/demo script
├── requirements.txt      # Python dependencies
├── test_marine_agent.py  # Unit tests
├── conftest.py          # Test configuration
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## iChatBio Agent Architecture

### Agent Card Definition
The agent implements the iChatBio `IChatBioAgent` interface with a comprehensive agent card:

```python
from ichatbio.types import AgentCard, AgentEntrypoint

card = AgentCard(
    name="WoRMS Marine Species Agent",
    description="Retrieves detailed marine species information from WoRMS database...",
    icon="https://www.marinespecies.org/images/WoRMS_logo.png",
    url="http://localhost:9999",
    entrypoints=[
        AgentEntrypoint(id="get_synonyms", description="Get synonyms...", parameters=MarineSynonymsParams),
        AgentEntrypoint(id="get_distribution", description="Get distribution...", parameters=MarineDistributionParams),
        # ... 5 more entrypoints
    ]
)
```

### Workflow Processing
Each request follows the iChatBio workflow pattern:

1. **Context Initialization**: `ResponseContext` provides logging and artifact management
2. **Process Creation**: `context.begin_process()` creates tracked workflow processes
3. **Parameter Validation**: Pydantic models ensure type-safe inputs
4. **API Interaction**: WoRMS client handles HTTP communication with error handling
5. **Artifact Generation**: `process.create_artifact()` creates persistent JSON artifacts
6. **Response Reply**: `context.reply()` provides structured user feedback

### Response Processing Flow

1. **iChatBio Request**: Framework routes natural language request to appropriate entrypoint
2. **Parameter Parsing**: Pydantic models validate and structure input parameters
3. **Agent Processing**: WoRMSiChatBioAgent orchestrates workflow with process logging
4. **Species Resolution**: Convert scientific names to AphiaIDs using WoRMS search API
5. **Data Retrieval**: Execute targeted WoRMS API calls with cloudscraper session
6. **Response Processing**: Parse and validate API responses with error handling
7. **Artifact Creation**: Generate structured JSON artifacts with comprehensive metadata
8. **User Communication**: Provide detailed summaries and artifact references via context.reply()

## iChatBio Integration Features

### Artifact Management
The agent creates rich artifacts for each successful query:
```python
await process.create_artifact(
    mimetype="application/json",
    description=f"Marine species synonyms for {species_name} (AphiaID: {aphia_id})",
    uris=[api_url],
    metadata={
        "data_source": "WoRMS Synonyms",
        "aphia_id": aphia_id,
        "scientific_name": species_name,
        "synonym_count": len(synonyms)
    }
)
```

### Process Logging
Comprehensive workflow tracking enables debugging and transparency:
```python
await process.log("Looking up AphiaID for species...")
await process.log(f"Found AphiaID: {aphia_id}")
await process.log("Constructed API URL", data={"url": api_url})
```

### Error Handling & Context
iChatBio's context management provides graceful error handling:
```python
try:
    # API operations
except Exception as e:
    await process.log("Error during API request", data={"error": str(e)})
    await context.reply(f"I encountered an error: {e}")
```

## Error Handling

The agent implements comprehensive error handling:
* **Network Errors**: Connection timeouts and HTTP errors
* **API Errors**: Invalid species names and missing AphiaIDs
* **Data Validation**: Pydantic model validation for all parameters
* **Response Processing**: Graceful handling of unexpected API response formats

## Configuration

The agent supports flexible configuration through:
* **Environment Variables**: `WORMS_API_URL`
* **YAML Configuration**: `env.yaml` file for local development
* **Default Fallbacks**: Built-in defaults for production deployment

## Dependencies

Core dependencies from `requirements.txt`:
* `ichatbio-sdk` - iChatBio agent framework and A2A protocol support
* `pydantic` - Data validation and modeling with type safety
* `requests` - HTTP client library for API communication
* `cloudscraper` - Anti-bot protection for reliable web scraping
* `PyYAML` - YAML configuration parsing and management
* `typing-extensions` - Enhanced type hints for Python compatibility

## Agent Discovery & Integration

The agent is automatically discoverable through the A2A protocol. Once running, the agent card is available at:
```
http://localhost:9999/.well-known/agent.json
```

This enables iChatBio and other A2A agents to discover capabilities, parameters, and integration endpoints automatically.
