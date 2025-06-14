import asyncio
import time
from marine_agent import MarineAgent
from ichatbio.types import ArtifactMessage, TextMessage, ProcessMessage

async def main():
    agent = MarineAgent()
    while True:
        query = input("\nAsk anything about the marine world (or type 'exit'): ")
        if query.strip().lower() in ("exit", "quit"):
            break

        try:
            start_time = time.time()

            async for msg in agent.run(query, "get_marine_info", None):

                # step messages 
                if isinstance(msg, ProcessMessage):
                    print(f"- {msg.summary}: {msg.description}")

                # final answer
                elif isinstance(msg, TextMessage):
                    print(f"\n[Final Answer]:\n{msg.text}")

                # artifacts commented for now
                # elif isinstance(msg, ArtifactMessage):
                #     print("\n[Artifact Output]")
                #     print("Content:\n", msg.content)
                #     print("Metadata:\n", msg.metadata)

                # else:
                #     print("\n[Other Message]:", msg)

            end_time = time.time()
            print(f"\n[Time Taken]: {end_time - start_time:.2f} seconds")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
