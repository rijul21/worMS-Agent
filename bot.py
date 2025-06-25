import asyncio
import time
from marine_agent import MarineAgent
from ichatbio.types import ArtifactMessage, TextMessage, ProcessMessage

async def main():
    agent = MarineAgent()
    print("Marine Species Bot- Type 'exit'stop\n")
    
    while True:
        query = input("Enter marine species query: ").strip()
        
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
            
        if not query:
            continue

        try:
            start_time = time.time()

            async for msg in agent.run(query, "get_marine_info", None):
                if isinstance(msg, ProcessMessage):
                    print(f"- {msg.summary}: {msg.description}")
                
                elif isinstance(msg, TextMessage):
                    print(f"\n{msg.text}")
                
                elif isinstance(msg, ArtifactMessage):
                    if msg.uris:
                        print(f"\nReference: {msg.uris[0]}")

            print(f"\nCompleted in {time.time() - start_time:.2f}s")
            

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())