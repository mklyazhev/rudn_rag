from typing import Any, Dict

from src.graph.state import GraphState
from src.graph.chains.generation import generation_chain
from loguru import logger


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response to the user question.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the generated response and the question
    """
    logger.info("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "documents": documents, "question": question}
