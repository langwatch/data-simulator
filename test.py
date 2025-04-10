from data_simulator import DataSimulator
from dotenv import load_dotenv
import os
import json

load_dotenv()
generator = DataSimulator(api_key=os.getenv("OPENAI_API_KEY"))

# Generate from a single file
results = generator.generate_from_directory(
    directory_path="sample_dir",
    context="You're a financial support assistant for Nike, helping a financial analyst decide whether to invest in the stock.",
    example_queries="how much revenue did nike make last year\nwhat risks does nike face\nwhat are nike's top 3 priorities",
    chunk_size=1000,
    chunk_overlap=200
)

for item in results:
    print(f"Chunk ID: {item['id']}")
    print(f"Query: {item['query']}")
    print(f"Document Chunk: {item['document'][:100]}...")
    print("-" * 50)

"""
# Example documents
documents = json.load(open("test_data/chroma_docs.json"))

results = generator.generate(
    documents=documents,
    context = "This is a technical support bot for Chroma, a vector database company often used by developers for building AI applications.",
    example_queries="how to add to a collection\nfilter by metadata\nretrieve embeddings when querying"
)

for item in results:
    print(item)
"""