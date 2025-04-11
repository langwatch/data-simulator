from openai import OpenAI
from typing import List, Dict
from tqdm import tqdm
import pandas as pd
from .prompts import *

class LLMProcessor:
    """
    Handles all LLM-related operations for data simulation.
    """
    
    def __init__(self, client: OpenAI):
        """
        Initialize the LLM processor.
        
        Args:
            client: OpenAI client instance
        """
        self.client = client
    
    def filter_documents(
        self,
        model: str,
        documents: List[str],
        ids: List[str],
        criteria: List[str],
        criteria_labels: List[str]
    ) -> List[str]:
        """
        Filter documents based on specified criteria.
        
        Args:
            model: Model name to use for filtering
            documents: List of documents to filter
            ids: List of document IDs
            criteria: List of criteria to filter by
            criteria_labels: Labels for the criteria
            
        Returns:
            List of IDs for documents that passed all criteria
        """
        
        labels = {}
        filtered_document_ids = []

        for document, id in tqdm(zip(documents, ids), total=len(documents), desc="Filtering documents"):
            labels[id] = {}

            for criterion, criterion_label in zip(criteria, criteria_labels):
                PROMPT = FILTER_USER_PROMPT.format(
                    criterion=criterion,
                    document=document
                )
                
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                        {"role": "user", "content": PROMPT}
                    ]
                )

                if completion.choices[0].message.content == "yes":
                    labels[id][criterion_label] = True
                else:
                    labels[id][criterion_label] = False
            
            passed_all = True
            
            for criterion_label in criteria_labels:
                if not labels[id][criterion_label]:
                    passed_all = False
                    break

            if passed_all:
                filtered_document_ids.append(id)

        return filtered_document_ids

    def generate_queries(
        self,
        model: str,
        documents: List[str],
        ids: List[str],
        context: str,
        example_queries: str
    ) -> pd.DataFrame:
        """
        Generate queries for documents based on context and examples.
        
        Args:
            model: Model name to use for generation
            documents: List of documents
            ids: List of document IDs
            context: Context for query generation
            example_queries: Example queries to guide generation
            
        Returns:
            DataFrame with document IDs and generated queries
        """
        if len(ids) != len(documents):
            raise ValueError("Length of ids must match length of documents")

        queries = []

        for id, document in tqdm(zip(ids, documents), total=len(ids), desc="Generating queries"):
            PROMPT = QUERY_USER_PROMPT.format(
                context=context,
                document=document,
                example_queries=example_queries
            )

            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT}
                ]
            )

            queries.append(completion.choices[0].message.content)

        queries_df = pd.DataFrame({"id": ids, "query": queries})

        return queries_df

    def generate_answers(
        self,
        model: str,
        queries: List[str],
        documents: List[str],
        ids: List[str]
    ) -> Dict[str, str]:
        """
        Generate answers for query-document pairs.
        
        Args:
            model: Model name to use for generation
            queries: List of queries
            documents: List of documents
            ids: List of document IDs
            
        Returns:
            Dictionary mapping document IDs to generated answers
        """
        if len(ids) != len(documents) or len(ids) != len(queries):
            raise ValueError("Length of ids, documents, and queries must match")
        
        answers = {}

        for id, document, query in tqdm(zip(ids, documents, queries), total=len(ids), desc="Generating answers"):
            PROMPT = ANSWER_USER_PROMPT.format(
                query=query,
                document=document
            )
            
            completion = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                    {"role": "user", "content": PROMPT}
                ]
            )
            
            answers[id] = completion.choices[0].message.content
        
        return answers