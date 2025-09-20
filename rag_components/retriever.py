from langchain.vectorstores import Chroma
from django.conf import settings
from .embeddings import get_embeddings
import os

def get_vector_store():
    """
    Returns the Chroma vector store. If it doesn't exist, creates an empty one.
    """
    vector_db_path = getattr(settings, 'RAG_CONFIG', {}).get('VECTOR_DB_PATH', './vector_db')
    os.makedirs(vector_db_path, exist_ok=True)

    embeddings = get_embeddings()

    vector_store = Chroma(
        persist_directory=vector_db_path,
        embedding_function=embeddings
    )
    return vector_store
