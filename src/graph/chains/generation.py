from langchain_core.output_parsers import StrOutputParser

from src.utils import get_llm
from src.graph.consts import GENERATION_PROMPT
from src.config import config


llm = get_llm(config.llm, True, config.gigachat_api_key.get_secret_value())
prompt = GENERATION_PROMPT

generation_chain = prompt | llm | StrOutputParser()
