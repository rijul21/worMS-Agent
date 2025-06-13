from ichatbio.server import run_agent_server
from marine_agent import MarineAgent

if __name__ == "__main__":
    agent = MarineAgent()
    run_agent_server(agent, host="0.0.0.0", port=9999)


    