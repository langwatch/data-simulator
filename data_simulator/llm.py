from openai import OpenAI
from typing import List
from tqdm import tqdm
import pandas as pd

def filter_documents(
    client: OpenAI,
    model: str,
    documents: List[str],
    ids: List[str],
    criteria: List[str],
    criteria_labels: List[str]
) -> List[str]:
        
    SYSTEM_INSTRUCTION = """
        You are an assistant specialized in filtering documents based on specific criteria.

        Given a document and a criterion, evaluate whether the document meets the criterion and output a single word: "yes" if the document meets the criterion, or "no" if it does not. Do not include any extra text or formatting, simply "yes" or "no".
        """
    
    labels = {}
    filtered_document_ids = []

    for document, id in tqdm(zip(documents, ids), total=len(documents), desc="Filtering documents"):
        labels[id] = {}

        for criterion, criterion_label in zip(criteria, criteria_labels):
            PROMPT = f"""
                Evaluate the following document with the criterion below.

                Criterion: {criterion}

                Document: {document}

                Output a single word: "yes" if the document meets the criterion, or "no" if it does not. Do not include any extra text or formatting, simply "yes" or "no".
                """
            
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
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

def create_golden_dataset(
    client: OpenAI,
    model: str,
    documents: List[str],
    ids: List[str],
    context: str,
    example_queries: str
) -> pd.DataFrame:

    if len(ids) != len(documents):
        raise ValueError("Length of ids must match length of documents")

    queries = []

    SYSTEM_INSTRUCTION = f"""
        You are an assistant specialized in generating queries to curate a high-quality synthetic dataset.

        Simply output the query without any additional words or formatting.
    """

    for id, document in tqdm(zip(ids, documents), total=len(ids), desc="Generating queries"):
        PROMPT = f"""
            Consider the context: 
            {context}

            Based on the following piece of text:
            <text>
            {document}
            <text>

            Please generate a realistic query that a user may ask relevant to the information provided above.

            Here are some example queries that users have asked which you should consider when generating your query:
            <example-queries>
            {example_queries}
            <example-queries>

            Do not repeat the example queries, they are only provided to give you an idea of the type of queries that users ask. 
            Make your query relevant to the information provided above and keep it in a similar style to the example queries, which may not always be in a complete question format.

            Simply output the query without any additional words.
        """

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": PROMPT}
            ]
        )

        queries.append(completion.choices[0].message.content)

    queries_df = pd.DataFrame({"id": ids, "query": queries})

    return queries_df