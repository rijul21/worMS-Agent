
import asyncio
import time
from marine_agent import MarineAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess, TextPart


class ConsoleContext(ResponseContext):
    def __init__(self):
        self.logs = []
        self.artifacts = []
        self.messages = []

    def begin_process(self, summary: str):
        print(f"\n[Step 1]: {summary}")
        
        class DummyProcess:
            def __init__(self):
                pass
                
            async def __aenter__(self):
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                print("Thank you!!\n")

            async def log(self, message, data=None):
                print(f"[Log]: {message} | {data or {}}")

            async def send(self, message):
                if isinstance(message, TextPart):
                    print(f"\n[Response]: {message.text}")
                else:
                    print(f"\n[Message]: {message}")

            async def create_artifact(self, **kwargs):
                print(f"\n[Artifact created]: {kwargs.get('description')}")
                if kwargs.get("uris"):
                    print(f"[Referance]: {kwargs['uris'][0]}")
                
        return DummyProcess()

    async def reply(self, message: str):
        print(f"\n[Marine Agent's Reply]: {message}")

async def main():
    agent = MarineAgent()
    print("Marine Species Bot - Type 'exit' to stop\n")

    while True:
        query = input("Enter marine species query: ").strip()

        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            start_time = time.time()
            context = ConsoleContext()
            await agent.run(context, query, "get_marine_info", None)
            print(f"\n[Completed in {time.time() - start_time:.2f}s]")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[ERROR]: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())


# import asyncio
# import time
# from marine_agent import MarineAgent
# from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess, TextPart

# # Simulating the iChatBio message flow
# class ConsoleContext(ResponseContext):
#     def __init__(self):
#         self.logs = []
#         self.artifacts = []
#         self.messages = []

#     def begin_process(self, summary: str):
#         print(f"\n[PROCESS STARTED]: {summary}")

#         class DummyProcess:
#             def __init__(self):
#                 pass

#             async def __aenter__(self):
#                 return self

#             async def __aexit__(self, exc_type, exc_val, exc_tb):
#                 print("[PROCESS ENDED]\n")

#             async def log(self, message, data=None):
#                 print(f"[LOG]: {message} | {data or {}}")

#             async def send(self, message):
#                 if isinstance(message, TextPart):
#                     print(f"\n[REPLY]: {message.text}")
#                 else:
#                     print(f"\n[MSG]: {message}")

#             async def create_artifact(self, **kwargs):
#                 print(f"\n[ARTIFACT CREATED]: {kwargs.get('description')}")
#                 if kwargs.get("uris"):
#                     print(f"[REFERENCE]: {kwargs['uris'][0]}")

#         return DummyProcess()

#     async def reply(self, message: str):
#         print(f"\n[AGENT REPLY]: {message}")

# async def main():
#     agent = MarineAgent()
#     print("Marine Species Bot .\n")

#     # 🔧 Hardcoded test query (replace with any species name)
#     query = "Salmo salar"

#     try:
#         start_time = time.time()
#         context = ConsoleContext()
#         await agent.run(context, query, "get_marine_info", None)
#         print(f"\n[Completed in {time.time() - start_time:.2f}s]")

#     except Exception as e:
#         print(f"[ERROR]: {str(e)}")

# if __name__ == "__main__":
#     asyncio.run(main())
