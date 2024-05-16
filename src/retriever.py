from langchain_chroma import Chroma


def get_retriever(documents, embeddings_model):
    return Chroma.from_documents(documents=documents, embedding=embeddings_model)
