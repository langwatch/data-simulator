from data_simulator import DataSimulator
from dotenv import load_dotenv
import os
import json

load_dotenv()
generator = DataSimulator(api_key=os.getenv("OPENAI_API_KEY"))

# Example documents
documents = json.load(open("test_data/chroma_docs.json"))

results = generator.generate(
    documents=documents,
    context = "This is a technical support bot for Chroma, a vector database company often used by developers for building AI applications.",
    example_queries="how to add to a collection\nfilter by metadata\nretrieve embeddings when querying"
)

for item in results:
    print(item)