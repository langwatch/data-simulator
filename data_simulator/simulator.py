from typing import List, Dict
from .llm import LLMProcessor
from openai import OpenAI
from .document_processor import DocumentProcessor
import json

class DataSimulator:
    """
    Generates synthetic data for RAG evaluation using LLMs.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the data simulator.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.llm_processor = LLMProcessor(self.client)

    def _generate(
        self,
        documents: Dict[str, str],
        context: str,
        example_queries: str,
        model_filter: str = "gpt-4o-mini",
        model_query: str = "gpt-4o-mini",
        model_answer: str = "gpt-4o-mini"
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic data from documents.
        
        Args:
            documents: Dictionary mapping document IDs to document content
            context: Context for query generation
            example_queries: Example queries to guide generation
            model_filter: Model to use for document filtering
            model_query: Model to use for query generation
            model_answer: Model to use for answer generation
            
        Returns:
            List of dictionaries containing document ID, document content, query, and answer
        """
        # Extract document IDs and contents
        corpus_ids = list(documents.keys())
        corpus_documents = [documents[key] for key in corpus_ids]

        # Define filtering criteria
        relevance = f"The document is relevant to the following context: {context}"
        completeness = "The document is complete, containing useful information."
        criteria = [relevance, completeness]
        criteria_labels = ["relevance", "completeness"]

        # Filter documents
        filtered_document_ids = self.llm_processor.filter_documents(
            model=model_filter,
            documents=corpus_documents,
            ids=corpus_ids,
            criteria=criteria,
            criteria_labels=criteria_labels
        )

        # Get filtered documents
        filtered_documents = [documents[id] for id in filtered_document_ids]

        # Generate queries
        query_dataset = self.llm_processor.generate_queries(
            model=model_query,
            documents=filtered_documents,
            ids=filtered_document_ids,
            context=context,
            example_queries=example_queries
        )
        
        # Extract queries
        queries = query_dataset['query'].tolist()
        
        # Generate answers
        answers_dict = self.llm_processor.generate_answers(
            model=model_answer,
            queries=queries,
            documents=filtered_documents,
            ids=filtered_document_ids
        )

        # Combine results
        output = []
        for i, row in enumerate(query_dataset.itertuples(index=False)):
            doc_id = row.id
            output.append({
                "id": doc_id,
                "document": documents[doc_id],
                "query": row.query,
                "answer": answers_dict[doc_id]
            })

        return output
    
    def generate_from_json(
        self,
        json_file_path: str,
        context: str,
        example_queries: str,
        model_filter: str = "gpt-4o-mini",
        model_query: str = "gpt-4o-mini",
        model_answer: str = "gpt-4o-mini"
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic data from pre-chunked documents stored in a JSON file.
        
        Args:
            json_file_path: Path to JSON file containing document chunks
            context: Context for query generation
            example_queries: Example queries to guide generation
            model_filter: Model to use for document filtering
            model_query: Model to use for query generation
            model_answer: Model to use for answer generation
            
        Returns:
            List of dictionaries containing document ID, document content, query, and answer
        """
        # Load the pre-chunked documents from the JSON file
        with open(json_file_path, 'r') as f:
            documents = json.load(f)
            
        # Use the existing generate method with the loaded documents
        return self._generate(
            documents=documents,
            context=context,
            example_queries=example_queries,
            model_filter=model_filter,
            model_query=model_query,
            model_answer=model_answer
        )

    def generate_from_docs(
        self,
        file_paths: List[str],
        context: str,
        example_queries: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_filter: str = "gpt-4o-mini",
        model_query: str = "gpt-4o-mini",
        model_answer: str = "gpt-4o-mini"
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic data from document files.
        
        Args:
            file_paths: List of paths to document files
            context: Context for query generation
            example_queries: Example queries to guide generation
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between document chunks
            model_filter: Model to use for document filtering
            model_query: Model to use for query generation
            model_answer: Model to use for answer generation
            
        Returns:
            List of dictionaries containing document ID, document content, query, and answer
        """
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Load and chunk all documents
        documents = {}
        for file_path in file_paths:
            file_chunks = processor.load_document(file_path)
            documents.update(file_chunks)
            
        # Use the existing generate method with the chunked documents
        return self._generate(
            documents=documents,
            context=context,
            example_queries=example_queries,
            model_filter=model_filter,
            model_query=model_query,
            model_answer=model_answer
        )
    
    def generate_from_directory(
        self,
        directory_path: str,
        context: str,
        example_queries: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_filter: str = "gpt-4o-mini",
        model_query: str = "gpt-4o-mini",
        model_answer: str = "gpt-4o-mini"
    ) -> List[Dict[str, str]]:
        """
        Generate synthetic data from all documents in a directory.
        
        Args:
            directory_path: Path to directory containing document files
            context: Context for query generation
            example_queries: Example queries to guide generation
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between document chunks
            model_filter: Model to use for document filtering
            model_query: Model to use for query generation
            model_answer: Model to use for answer generation
            
        Returns:
            List of dictionaries containing document ID, document content, query, and answer
        """
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Load and chunk all documents in the directory
        documents = processor.load_directory(directory_path)
        
        # Use the existing generate method with the chunked documents
        return self._generate(
            documents=documents,
            context=context,
            example_queries=example_queries,
            model_filter=model_filter,
            model_query=model_query,
            model_answer=model_answer
        )