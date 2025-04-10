from typing import List, Dict, Optional
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

class DocumentProcessor:
    """Handles loading and chunking of document files for data simulation."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
    def load_document(self, file_path: str) -> List[Dict[str, str]]:
        """Load a document file and split it into chunks."""
        # Select the appropriate loader based on file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext in ['.docx', '.doc']:
            loader = Docx2txtLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        elif ext in ['.md', '.markdown']:
            loader = UnstructuredMarkdownLoader(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        # Load the document
        documents = loader.load()
        
        # Split the document into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Convert to the format expected by DataSimulator
        result = {}
        for i, chunk in enumerate(chunks):
            chunk_id = f"{os.path.basename(file_path)}_chunk_{i}"
            result[chunk_id] = chunk.page_content
            
        return result
    
    def load_directory(self, directory_path: str) -> Dict[str, str]:
        """Load all documents from a directory and split them into chunks."""
        result = {}
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_chunks = self.load_document(file_path)
                result.update(file_chunks)
                    
        return result