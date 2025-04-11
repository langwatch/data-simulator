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