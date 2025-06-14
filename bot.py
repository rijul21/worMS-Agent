import asyncio
from marine_agent import MarineAgent
from ichatbio.types import ArtifactMessage, TextMessage, ProcessMessage  # keep for future use

async def main():
    agent = MarineAgent()
    while True:
        query = input("\nTry asking this agent anything about the marine world! (or 'exit'): ")
        if query.strip().lower() in ("exit", "quit"):
            break
        print("\n[Agent steps:]")
        async for msg in agent.run(query, "get_marine_info", None):
            # Handle ProcessMessage (step updates)
            if isinstance(msg, ProcessMessage):
                print(f"- {msg.summary}: {msg.description}")

            # Handle TextMessage (final answer as text fallback)
            elif isinstance(msg, TextMessage):
                print(f"\n[Final Answer]:\n{msg.text}")

            # Handle ArtifactMessage (disabled for now)
            # elif isinstance(msg, ArtifactMessage):
            #     print("\n[Artifact Output]")
            #     print("Content:\n", msg.content)
            #     print("Metadata:\n", msg.metadata)

            # Catch-all (optional)
            else:
                print("\n[Other Message]:", msg)

if __name__ == "__main__":
    asyncio.run(main())
