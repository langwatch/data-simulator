# Data Simulator

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langwatch/data-simulator/blob/main/data_simulator.ipynb)

`data-simulator` is a lightweight Python library for generating synthetic datasets from your own corpus — perfect for testing, evaluating, or fine-tuning LLM Applications.

## Motivation

Real documents contain a mix of useful and irrelevant content. When generating synthetic data, this leads to:

- Queries that real users would never ask
- Test sets that don't reflect actual usage
- Wasted effort optimizing for the wrong things

Data Simulator filters out low-quality content first, then generates realistic queries and answers that match how your system will actually be used.

---

## Getting Started

Install from PyPI:

```bash
pip install llm-data-simulator
```

Or install it locally:

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
  "id": "chunk_42",
  "document": "Nike reported annual revenue of $44.5 billion for fiscal year 2022, an increase of 5% compared to the previous year.",
  "query": "What was Nike's revenue growth in 2022?",
  "answer": "Nike's revenue grew by 5% in fiscal year 2022, reaching $44.5 billion."
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