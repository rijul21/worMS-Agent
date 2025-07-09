import asyncio
import json
import httpx
from marine_agent import MarineAgent, Config, WoRMSClient
from openai import AsyncOpenAI
import instructor
from marine_agent import MarineQueryModel

async def debug_llm_extraction(query: str):
    """Debug what the LLM extracts from the query"""
    print(f"\n=== DEBUGGING LLM EXTRACTION ===")
    print(f"Input query: '{query}'")
    
    openai_client = AsyncOpenAI(
        api_key=Config.GROQ_API_KEY,
        base_url=Config.GROQ_BASE_URL,
    )
    instructor_client = instructor.patch(openai_client)

    marine_query = await instructor_client.chat.completions.create(
        model=Config.MODEL_NAME,
        response_model=MarineQueryModel,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract marine animal names from user questions. Look for both scientific names "
                    "(like 'Orcinus orca') and common names (like 'killer whale', 'great white shark'). "
                    "If you find either, extract it. Handle conversational queries naturally."
                )
            },
            {"role": "user", "content": query}
        ],
        max_retries=3
    )
    
    print(f"LLM extracted:")
    print(f"  - Scientific name: {marine_query.scientificname}")
    print(f"  - Common name: {marine_query.common_name}")
    
    return marine_query

async def debug_worms_api(search_term: str, search_type: str):
    """Debug direct WoRMS API calls"""
    print(f"\n=== DEBUGGING WoRMS API ({search_type}) ===")
    print(f"Search term: '{search_term}'")
    
    worms_client = WoRMSClient()
    
    async with httpx.AsyncClient() as client:
        if search_type == "scientific":
            endpoint = f"/AphiaRecordsByName/{search_term}?like=false&marine_only=true"
            url = f"{worms_client.base_url}{endpoint}"
            print(f"API URL: {url}")
            
            try:
                response = await client.get(url)
                print(f"Status Code: {response.status_code}")
                print(f"Response Headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Response data type: {type(data)}")
                    print(f"Response data: {json.dumps(data, indent=2)}")
                    return data
                else:
                    print(f"Error response: {response.text}")
                    return None
                    
            except Exception as e:
                print(f"Exception occurred: {e}")
                return None
                
        elif search_type == "common":
            endpoint = f"/AphiaRecordsByVernacular/{search_term}?like=true&offset=1"
            url = f"{worms_client.base_url}{endpoint}"
            print(f"API URL: {url}")
            
            try:
                response = await client.get(url)
                print(f"Status Code: {response.status_code}")
                print(f"Response Headers: {dict(response.headers)}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"Response data type: {type(data)}")
                    print(f"Response data: {json.dumps(data, indent=2)}")
                    return data
                else:
                    print(f"Error response: {response.text}")
                    return None
                    
            except Exception as e:
                print(f"Exception occurred: {e}")
                return None

async def test_various_queries():
    """Test different types of queries"""
    test_queries = [
        "tell me about whales",
        "tell me about orcas", 
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"TESTING QUERY: '{query}'")
        print(f"{'='*60}")
        
        # Step 1: Debug LLM extraction
        marine_query = await debug_llm_extraction(query)
        
        # Step 2: Test WoRMS API calls
        if marine_query.scientificname:
            await debug_worms_api(marine_query.scientificname, "scientific")
        
        if marine_query.common_name:
            await debug_worms_api(marine_query.common_name, "common")
        
        print(f"\n{'='*60}")
        input("Press Enter to continue to next query...")

async def quick_test_specific_terms():
    """Quick test of specific terms that should work"""
    print("\n=== QUICK TEST OF KNOWN WORKING TERMS ===")
    
    # These should definitely work
    working_terms = [
        ("Orcinus orca", "scientific"),
        ("killer whale", "common"),
        ("Balaenoptera musculus", "scientific"),
        ("blue whale", "common")
    ]
    
    for term, search_type in working_terms:
        print(f"\nTesting: {term} ({search_type})")
        result = await debug_worms_api(term, search_type)
        if result:
            print(f"✅ SUCCESS: Found {len(result) if isinstance(result, list) else 1} record(s)")
        else:
            print("❌ FAILED: No results")

if __name__ == "__main__":
    print("Marine Agent Debug Tool")
    print("1. Quick test of known working terms")
    print("2. Full test of various queries")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(quick_test_specific_terms())
    elif choice == "2":
        asyncio.run(test_various_queries())
    else:
        print("Invalid choice")