from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents a state of a graph.

    Attributes:
        question: Question being asked.
        generation: LLM Generation.
        use_web_search: Whether to use web search.
        documents: List of documents.
        user_id: The ID of the user interacting with the bot.
        chat_history: Chat history containing message histories.
    """

    question: str
    generation: str
    use_web_search: bool
    documents: List[str]
    user_id: str
    chat_history: List[str]
