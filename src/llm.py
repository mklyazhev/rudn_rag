import torch
from torch import bfloat16
import transformers
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline
from langchain_openai import ChatOpenAI
from langchain.chat_models.gigachat import GigaChat


def get_llm(model_name, use_api=False, api_key=None):
    if use_api:
        return get_api_llm(model_name, api_key)
    else:
        return get_hf_llm(model_name)


def get_api_llm(model_name, api_key):
    if "gpt" in model_name:
        return ChatOpenAI(model=model_name, api_key=api_key)
    elif "Giga" in model_name:
        return GigaChat(model=model_name, credentials=api_key, verify_ssl_certs=False)


def get_hf_llm(model_name):
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
