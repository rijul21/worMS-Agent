AI agent to query marine species using natural language. 
It extracts scientific names from user input and fetches detailed taxonomic and ecological data via the World Register of Marine Species (WoRMS) API.

---

Tech Stack

* **Python 3.10+**
* **OpenAI SDK** (LLMs via GROQ)
* **Instructor** for structured output
* **iChatBio Agent SDK**
* **Pydantic** for data modeling
* **httpx** for async API calls
* **dotenv** for config

---

Key Components

* `MarineAgent` – Main agent class
* `WoRMSClient` – REST client for WoRMS API
* `MarineDataProcessor` – Processes and validates data into structured models
* `models.py` – Defines species, synonyms, classification, etc.

---

Setup & Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/marine_agent.git
cd marine_agent_sdk
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file:

```env
GROQ_API_KEY=your-groq-api-key
```

### 4. Run the agent (CLI demo)

```bash
python bot.py
```

You’ll see prompts like:

```bash
Ask a marine question (or 'exit'): Tell me about Orcinus orca
```

---

API Endpoints Used

* `GET /AphiaRecordsByName/{name}`
* `GET /AphiaSynonymsByAphiaID/{id}`
* `GET /AphiaDistributionsByAphiaID/{id}`
* `GET /AphiaVernacularsByAphiaID/{id}`
* `GET /AphiaClassificationByAphiaID/{id}`
* `GET /AphiaChildrenByAphiaID/{id}`

