from data_simulator import DataSimulator
from dotenv import load_dotenv
import os

load_dotenv()
generator = DataSimulator(api_key=os.getenv("OPENAI_API_KEY"))

# Option 1: Generate from pre-chunked documents in a JSON file
print("=== Option 1: Generate from pre-chunked documents (JSON) ===")
results_from_json = generator.generate_from_json(
    json_file_path="test_data/chroma_docs.json",
    context="This is a technical support bot for Chroma, a vector database company often used by developers for building AI applications.",
    example_queries="how to add to a collection\nfilter by metadata\nretrieve embeddings when querying"
)

print(f"Generated {len(results_from_json)} results from JSON file")
for i, item in enumerate(results_from_json[:2]):  # Print first 2 results
    print(f"Result {i+1}:")
    print(f"Chunk ID: {item['id']}")
    print(f"Query: {item['query']}")
    print(f"Document Chunk: {item['document'][:100]}...")
    print("-" * 50)

# Option 2: Generate from document files
print("\n=== Option 2: Generate from document files ===")
results_from_files = generator.generate_from_docs(
    file_paths=["test_data/nike_10k.pdf"],
    context="You're a financial support assistant for Nike, helping a financial analyst decide whether to invest in the stock.",
    example_queries="how much revenue did nike make last year\nwhat risks does nike face\nwhat are nike's top 3 priorities"
)

print(f"Generated {len(results_from_files)} results from files")
for i, item in enumerate(results_from_files[:2]):  # Print first 2 results
    print(f"Result {i+1}:")
    print(f"Chunk ID: {item['id']}")
    print(f"Query: {item['query']}")
    print(f"Document Chunk: {item['document'][:100]}...")
    print("-" * 50)