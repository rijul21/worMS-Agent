import os
from ichatbio.server import run_agent_server
from agent import WoRMSReActAgent 

if __name__ == "__main__":
    agent = WoRMSReActAgent()  
    port = int(os.getenv("PORT", 9999))
    print(f"Starting WoRMS ReAct Agent on port {port}")
    run_agent_server(agent, host="0.0.0.0", port=port)