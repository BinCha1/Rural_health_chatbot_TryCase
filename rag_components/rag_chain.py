import os
from django.conf import settings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .document_loader import load_documents
from .text_splitter import split_documents
from .retriever import get_vector_store
from .prompt_template import get_prompt_template

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    rag_config = getattr(settings, 'RAG_CONFIG', {})
    model_name = rag_config.get("LLM_MODEL", 'llama-3.1-8b-instant')
    temperature = rag_config.get('TEMPERATURE', 0.1)
    max_tokens = rag_config.get('MAX_TOKENS', 2048)

    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        groq_api_key=getattr(settings, 'GROQ_API_KEY', None)
    )
    return llm

def update_vector_db():
    vector_db_path = getattr(settings, 'RAG_CONFIG', {}).get('VECTOR_DB_PATH', './vector_db')
    os.makedirs(vector_db_path, exist_ok=True)

    embeddings = get_embeddings()
    documents = load_documents()
    chunks = split_documents(documents)

    if chunks:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=vector_db_path
        )
    else:
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=vector_db_path
        )

    return vector_store

def get_rag_response(question):
    try:
        vector_store = get_vector_store()
        llm = get_llm()
        prompt = get_prompt_template()

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        result = qa_chain({"query": question})
        answer = result['result']
        source_documents = result.get('source_documents', [])

        sources = []
        for doc in source_documents:
            source = {
                'title': doc.metadata.get('source', 'Unknown'),
                'content': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
            }
            sources.append(source)

        return {
            'answer': answer,
            'sources': sources
        }

    except Exception as e:
        print(f"Error in RAG chain: {str(e)}")
        return {
            'answer': "I don't have enough information to answer this question.",
            'sources': []
        }
