import asyncio
from marine_agent import MarineAgent

async def main():
    agent = MarineAgent()
    while True:
        query = input("\nAsk a marine question (or 'exit'): ")
        if query.strip().lower() in ("exit", "quit"):
            break
        print("\n[Agent steps:]")
        async for msg in agent.run(query, "get_marine_info", None):
            # Print streaming/status messages as they arrive
            if hasattr(msg, "text") and msg.text:
                print(f"\n[Final Answer]:\n{msg.text}")
            elif hasattr(msg, "summary") and hasattr(msg, "description"):
                print(f"- {msg.summary}: {msg.description}")

if __name__ == "__main__":
    asyncio.run(main())