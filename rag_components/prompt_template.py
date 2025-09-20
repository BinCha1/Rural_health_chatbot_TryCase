from langchain.prompts import PromptTemplate

def get_prompt_template():
    template = """
Instructions:
1. Answer the question based only on the provided context.
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Keep your answers concise and easy to understand for rural communities.
4. If appropriate, suggest consulting a healthcare professional for personalized advice.
5. Do not make up information.

Context:
{context}

Question:
{question}

Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)
