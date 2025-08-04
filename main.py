import os
from ichatbio.server import run_agent_server
from worms_api import WoRMSAgent

if __name__ == "__main__":
    agent = WoRMSAgent()
    port = int(os.getenv("PORT", 9999))
    print(f"Starting agent on port {port}")
    run_agent_server(agent, host="0.0.0.0", port=port)