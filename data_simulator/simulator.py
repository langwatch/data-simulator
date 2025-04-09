# query_doc_generator/generator.py
from typing import List, Dict
from .llm import create_golden_dataset, filter_documents
from openai import OpenAI

class DataSimulator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        documents: Dict[str, str],
        context: str,
        example_queries: str,
        model_filter: str = "gpt-4o-mini",
        model_query: str = "gpt-4o-mini"
    ) -> List[Dict[str, str]]:

        corpus_ids = list(documents.keys())
        corpus_documents = [documents[key] for key in corpus_ids]

        relevance = f"The document is relevant to the following context: {context}"
        completeness = "The document is complete, containing useful information."

        criteria = [relevance, completeness]
        criteria_labels = ["relevance", "completeness"]

        filtered_document_ids = filter_documents(
            client=self.client,
            model=model_filter,
            documents=corpus_documents,
            ids=corpus_ids,
            criteria=criteria,
            criteria_labels=criteria_labels
        )

        corpus_documents = [documents[id] for id in filtered_document_ids]
        corpus_ids = filtered_document_ids

        golden_dataset = create_golden_dataset(
            client=self.client,
            model=model_query,
            documents=corpus_documents,
            ids=corpus_ids,
            context=context,
            example_queries=example_queries
        )

        output = []
        for row in golden_dataset.itertuples(index=False):
            output.append({
                "id": row.id,
                "document": documents[row.id],
                "query": row.query
            })

        return output