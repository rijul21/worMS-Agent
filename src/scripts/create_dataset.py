"""
Create evaluation dataset for WoRMS Agent
Ground truth test cases with known correct answers
"""

from langsmith import Client
from datetime import datetime
import dotenv

# Load environment variables
dotenv.load_dotenv()

client = Client()

# Dataset name
DATASET_NAME = "worms-agent-eval"

def create_evaluation_dataset():
    """Create dataset with test cases"""
    
    # Check if dataset already exists
    try:
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
        print(f"Dataset '{DATASET_NAME}' already exists. Adding examples to it.")
    except:
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Evaluation dataset for WoRMS agent with ground truth answers"
        )
        print(f"Created new dataset: {DATASET_NAME}")
    
    # Test cases with ground truth
    test_cases = [
        {
            "name": "Single species - conservation status",
            "inputs": {
                "request": "What is the conservation status of Orcinus orca?",
                "species_names": ["Orcinus orca"]
            },
            "outputs": {
                "expected_query_type": "conservation",
                "expected_species_count": 1,
                "expected_tools": ["get_species_attributes"],
                "should_mention": ["IUCN", "Data Deficient"],
                "aphia_id": 137102
            }
        },
        {
            "name": "Single species - taxonomy",
            "inputs": {
                "request": "What family does Tursiops truncatus belong to?",
                "species_names": ["Tursiops truncatus"]
            },
            "outputs": {
                "expected_query_type": "taxonomy",
                "expected_species_count": 1,
                "expected_tools": ["get_taxonomic_classification"],
                "should_mention": ["Delphinidae", "family"],
                "aphia_id": 137111
            }
        },
        {
            "name": "Comparison - two species",
            "inputs": {
                "request": "Compare the conservation status of Orcinus orca and Delphinus delphis",
                "species_names": ["Orcinus orca", "Delphinus delphis"]
            },
            "outputs": {
                "expected_query_type": "comparison",
                "expected_species_count": 2,
                "expected_tools": ["get_species_attributes"],
                "should_mention": ["Orcinus orca", "Delphinus delphis"],
                "aphia_ids": [137102, 137094]
            }
        },
        {
            "name": "Distribution query",
            "inputs": {
                "request": "Where is Megaptera novaeangliae found?",
                "species_names": ["Megaptera novaeangliae"]
            },
            "outputs": {
                "expected_query_type": "distribution",
                "expected_species_count": 1,
                "expected_tools": ["get_species_distribution"],
                "should_mention": ["distribution", "ocean"],
                "aphia_id": 137092
            }
        },
        {
            "name": "Common name resolution",
            "inputs": {
                "request": "Tell me about the killer whale",
                "species_names": ["killer whale"]
            },
            "outputs": {
                "expected_query_type": "single_species",
                "expected_species_count": 1,
                "should_resolve_to": "Orcinus orca",
                "expected_tools": ["get_species_attributes"],
                "aphia_id": 137102
            }
        },
        {
            "name": "Nonexistent species",
            "inputs": {
                "request": "What is the conservation status of Fakeus speciesname?",
                "species_names": ["Fakeus speciesname"]
            },
            "outputs": {
                "expected_query_type": "conservation",
                "expected_species_count": 1,
                "should_fail": True,
                "expected_error": "not found in WoRMS"
            }
        }
    ]
    
    # Add examples to dataset
    for i, test_case in enumerate(test_cases):
        try:
            example = client.create_example(
                dataset_id=dataset.id,
                inputs=test_case["inputs"],
                outputs=test_case["outputs"],
                metadata={"test_name": test_case["name"]}
            )
            print(f"Added: {test_case['name']}")
        except Exception as e:
            print(f"Failed to add {test_case['name']}: {e}")
    
    print(f"\nDataset ready with {len(test_cases)} test cases")
    print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")
    
    return dataset


if __name__ == "__main__":
    print("=" * 60)
    print("Creating WoRMS Agent Evaluation Dataset")
    print("=" * 60)
    
    dataset = create_evaluation_dataset()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Run evaluators on this dataset")
    print("2. View results in LangSmith UI")
    print("=" * 60)