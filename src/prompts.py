from langchain_core.prompts import PromptTemplate


RETRIEVAL_RERANKER = PromptTemplate(
        template="",
        input_variables=["question", "document"],
    )

QUESTION_ROUTER = PromptTemplate(
        template="",
        input_variables=["question"],
    )

HALLUCINATION_GRADER = PromptTemplate(
        template="",
        input_variables=["generation", "documents"],
    )
