import logging
import os
from ichatbio.server import run_agent_server
from marine_agent import MarineAgent

# Add logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    agent = MarineAgent()
    port = int(os.getenv("PORT", 9999))
    print(f"Starting agent on port {port}")
    run_agent_server(agent, host="0.0.0.0", port=port)