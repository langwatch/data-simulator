# Data Simulator

`data-simulator` is a lightweight Python library for generating synthetic datasets from your own corpus — perfect for testing, evaluating, or fine-tuning LLM Applications.

## Motivation

**The Problem**: Building high-quality synthetic datasets for LLM applications is challenging. Most approaches treat all documents equally, but real-world document collections are messy:

- Corporate documents filled with boilerplate legal text
- Technical manuals with irrelevant metadata sections
- Knowledge bases containing outdated or placeholder content

When you generate synthetic queries from these unfiltered documents, you end up with:

1. **Unrealistic queries** that don't reflect what users actually ask
2. **Distorted benchmarks** that don't measure what matters in production
3. **Development time wasted** chasing improvements that won't impact real users

**The Solution**: Data Simulator takes a different approach:

- **Smart filtering** removes low-value content before query generation
- **Context-aware prompting** creates queries aligned with your specific use case
- **Complete triplets** provide document-query-answer combinations for end-to-end testing

The result? A synthetic dataset that actually represents how your system will be used in the real world, leading to meaningful improvements in your LLM applications.

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
from dotenv import load_dotenv
import os
from data_simulator.utils import display_results

load_dotenv()

generator = DataSimulator(api_key=os.getenv("OPENAI_API_KEY"))

results = generator.generate_from_docs(
    file_paths=["test_data/nike_10k.pdf"],
    context="You're a financial support assistant for Nike, helping a financial analyst decide whether to invest in the stock.",
    example_queries="how much revenue did nike make last year\nwhat risks does nike face\nwhat are nike's top 3 priorities"
)

display_results(results)
```

## Output Format

```python
{
  "id": "doc1",
  "document": "To reset your password, visit the settings page.",
  "query": "how do I change my password",
  "answer": "To reset your password, visit the settings page."
}
```

## Project Structure

The project follows a modular, object-oriented design:

- `simulator.py`: Contains the main `DataSimulator` class that orchestrates the data generation process
- `llm.py`: Houses the `LLMProcessor` class that handles all LLM-related operations
- `document_processor.py`: Provides the `DocumentProcessor` class for loading and chunking documents
- `prompts.py`: Stores all prompt templates used for LLM interactions
- `utils.py`: Contains utility functions like `display_results` for formatting output

## License

MIT License