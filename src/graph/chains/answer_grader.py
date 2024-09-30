from src.utils import get_llm_with_chain
from src.prompts import ANSWER_GRADER_PROMPT
from src.config import config


answer_grader = get_llm_with_chain(config.llm, ANSWER_GRADER_PROMPT, True, config.gigachat_api_key.get_secret_value())