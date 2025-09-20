import os
from django.conf import settings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_documents():
    """
    Load all uploaded documents from media/documents.
    Supports PDF and TXT files.
    """
    documents = []
    upload_dir = os.path.join(settings.MEDIA_ROOT, 'documents')

    if not os.path.exists(upload_dir):
        return documents

    for filename in os.listdir(upload_dir):
        file_path = os.path.join(upload_dir, filename)
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            continue
        docs = loader.load()
        documents.extend(docs)

    return documents
