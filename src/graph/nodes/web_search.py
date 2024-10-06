import os
from typing import Any, Dict

from loguru import logger

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from src.graph.state import GraphState
from src.config import config

os.environ["TAVILY_API_KEY"] = config.tavily_api_key.get_secret_value()
search = TavilySearchAPIWrapper()
web_search_tool = TavilySearchResults(max_results=3, api_wrapper=search)


def web_search(state: GraphState) -> Dict[str, Any]:
    """
    Search the web for documents.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the retrieved documents and the question
    """
    logger.info("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]  # only relevant documents

    tavily_results = web_search_tool.invoke({"query": question})

    # get one huge string with all the results
    tavily_results_joined = "\n".join([res["content"] for res in tavily_results])

    # create a document object
    web_search_result = Document(page_content=tavily_results_joined)

    # append web search to the list of documents
    if documents is not None:
        documents.append(web_search_result)
    else:
        documents = [web_search_result]

    return {"documents": documents, "question": question}
