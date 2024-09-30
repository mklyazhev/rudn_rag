from src.utils import get_llm_with_chain
from src.prompts import ROUTER_PROMPT
from src.config import config


question_router = get_llm_with_chain(config.llm, ROUTER_PROMPT, True, config.gigachat_api_key.get_secret_value())
