import torch
from torch import bfloat16
import transformers
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.chat_models.gigachat import GigaChat
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import JsonOutputParser


def get_chunks(docs_path, file_ext):
    loader = DirectoryLoader(docs_path, glob=file_ext, show_progress=True)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=100, add_start_index=True
    )

    return text_splitter.split_documents(docs)


def get_embeddings(model_name):
    # model_kwargs = {'device': 'cuda'}
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )


def get_retriever(documents, embeddings_model):
    return Chroma.from_documents(documents=documents, embedding=embeddings_model)\
        .as_retriever(search_type="similarity", search_kwargs={"k": 3})


def get_llm(model_name, use_api=False, api_key=None):
    if use_api:
        return __get_api_llm(model_name, api_key)
    else:
        return __get_hf_llm(model_name)


def __get_api_llm(model_name, api_key):
    if "gpt" in model_name:
        return ChatOpenAI(model=model_name, api_key=api_key)
    elif "Giga" in model_name:
        return GigaChat(model=model_name, credentials=api_key, verify_ssl_certs=False)


def __get_hf_llm(model_name):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_name
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return HuggingFacePipeline(pipeline=query_pipeline)


def get_llm_with_chain(model_name, prompt, use_api=False, api_key=None):
    llm = get_llm(model_name, use_api, api_key)

    return prompt | llm | JsonOutputParser()
