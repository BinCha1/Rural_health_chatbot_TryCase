from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.conf import settings

def split_documents(documents: List) -> List:
    """
    Split documents into chunks for better retrieval.
    """
    # Get chunk size and overlap from settings
    chunk_size = getattr(settings, 'RAG_CONFIG', {}).get('CHUNK_SIZE', 1000)
    chunk_overlap = getattr(settings, 'RAG_CONFIG', {}).get('CHUNK_OVERLAP', 200)
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split documents
    chunks = text_splitter.split_documents(documents)
    
    return chunks