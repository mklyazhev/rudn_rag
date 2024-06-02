from langchain_chroma import Chroma


def get_retriever(documents, embeddings_model):
    return Chroma.from_documents(documents=documents, embedding=embeddings_model)\
        .as_retriever(search_type="similarity", search_kwargs={"k": 3})
