from langchain_huggingface import HuggingFaceEmbeddings
from django.conf import settings

def get_embeddings():
    """
    Returns the HuggingFace embeddings object.
    """
    embedding_model = getattr(settings, 'RAG_CONFIG', {}).get(
        'EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'
    )
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    return embeddings
