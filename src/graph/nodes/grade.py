from typing import Any, Dict

from loguru import logger

from src.graph.chains.retrieval_grader import retrieval_grader
from src.graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the user question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): Filtered out irrelevant documents and updated use_web_search state.
    """
    logger.info("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    use_web_search = False
    for doc in documents:
        result = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        if "yes" in result.lower():
            logger.info("---DOCUMENT IS RELEVANT---")
            filtered_documents.append(doc)
        else:
            logger.info("---DOCUMENT IS NOT RELEVANT---")
            continue
    if [] == filtered_documents:
      use_web_search = True
    return {
        "documents": filtered_documents,
        "use_web_search": use_web_search,
        "question": question,
    }
