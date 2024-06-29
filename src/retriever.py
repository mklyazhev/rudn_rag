from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever


def get_history_aware_retriever(llm, retriever, prompt):
    return create_history_aware_retriever(llm, retriever, prompt)


def get_retriever(documents, embeddings_model):
    return Chroma.from_documents(documents=documents, embedding=embeddings_model)\
        .as_retriever(search_type="similarity", search_kwargs={"k": 3})
