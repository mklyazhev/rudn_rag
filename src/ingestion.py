from src.utils import get_chunks, get_embeddings, get_retriever
from src.config import config


chunks = get_chunks(config.documents, "*.md")
embeddings = get_embeddings(config.embeddings)
retriever = get_retriever(chunks, embeddings)
