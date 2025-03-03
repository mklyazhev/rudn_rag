from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_chunks(docs_path, file_ext):
    loader = DirectoryLoader(docs_path, glob=file_ext, show_progress=True)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, add_start_index=True
    )

    return text_splitter.split_documents(docs)
