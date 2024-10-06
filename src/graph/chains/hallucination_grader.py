from src.utils import get_llm_with_chain
from src.graph.consts import HALLUCINATION_GRADER_PROMPT
from src.config import config


hallucination_grader = get_llm_with_chain(config.llm, HALLUCINATION_GRADER_PROMPT, True, config.gigachat_api_key.get_secret_value())
