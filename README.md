# Data Simulator

`data-simulator` is a lightweight Python library for generating synthetic datasets of queries and documents from your own corpus — perfect for testing, evaluating, or fine-tuning RAG and retrieval systems.

Just pass in your documents, and it will:
- Filter out noisy or irrelevant content
- Generate realistic queries grounded in your use case
- Return high-quality, ready-to-use query-doc pairs

No manual tagging. No public benchmark assumptions. Just data that looks and feels like your production traffic.

---

## Getting Started

Install it locally:

```bash
git clone https://github.com/langwatch/data-simulator.git
cd data-simulator
pip install -e .
```

Run the built-in test script:

```bash
python test.py
```

## Example test.py

```python
from data_simulator import DataSimulator
import os
from dotenv import load_dotenv

load_dotenv()

generator = DataSimulator(api_key=os.getenv("OPENAI_API_KEY"))

documents = {
    "doc1": "To reset your password, visit the settings page.",
    "doc2": "Refunds are available up to 14 days after purchase."
}

example_queries = '''
reset password
refund policy
cancel subscription
'''

context = "This is a knowledge base for customer support."

query_doc_pairs = generator.generate(
    documents=documents,
    context=context,
    example_queries=example_queries
)

for pair in query_doc_pairs:
    print(pair)
```

## Output Format

```python
{
  "id": "doc1",
  "document": "To reset your password, visit the settings page.",
  "query": "how do I change my password"
}
```

## Why Filtering First?

Most synthetic benchmarks generate queries directly from documents — assuming the documents are representative. But if your dataset includes intro pages, legal disclaimers, or incomplete content, your queries won’t reflect real user behavior.

By filtering first, data-simulator ensures that:

- Queries are only generated from high-signal content
- The resulting dataset better mirrors real-world use
- Your evaluation metrics actually mean something

## License

MIT License