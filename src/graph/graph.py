from langgraph.graph import END, StateGraph

from src.graph.state import GraphState
from src.graph.consts import RETRIEVE, GENERATE, GRADE_DOCUMENTS, WEBSEARCH
from src.graph.chains import hallucination_grader, answer_grader, question_router
from src.graph.nodes import generate, grade_documents, retrieve, web_search
from loguru import logger


def decide_to_generate(state):
    logger.info("---ASSESS GRADED DOCUMENTS---")

    if state["use_web_search"]:
        logger.info("---DECISION: NOT ALL DOCUMENTS ARE RELEVANT, GO TO WEB---")
        return WEBSEARCH
    else:
        logger.info("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState):
    logger.info("---CHECK HALLUCINATIONS---")

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    if "yes" in score.lower():
        logger.info("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        logger.info("---CHECK ANSWER---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if "yes" in score.lower():
            logger.info("---DECISION: ANSWER ADDRESSES THE USER QUESTION---")
            return "useful"
        else:
            logger.info("---DECISION: ANSWER DOES NOT ADDRESS THE USER QUESTION---")
            return "not_useful"
    else:
        logger.info("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not_supported"


def route_question(state: GraphState):
    logger.info("---ROUTE QUESTION---")
    question = state["question"]

    source = question_router.invoke({"question": question})
    if GENERATE in source.lower():
        logger.info("---DECISION: ROUTE QUESTION TO GENERATE---")
        return GENERATE
    elif "vectorstore" in source:
        logger.info("---DECISION: ROUTE QUESTION TO RAG---")
        return RETRIEVE


# Создаем граф состояния
flow = StateGraph(state_schema=GraphState)

# Добавляем узлы графа
flow.add_node(RETRIEVE, retrieve)
flow.add_node(GRADE_DOCUMENTS, grade_documents)
flow.add_node(GENERATE, generate)
flow.add_node(WEBSEARCH, web_search)

# Устанавливаем точку входа в граф
flow.set_conditional_entry_point(
    route_question,
    path_map={RETRIEVE: RETRIEVE, GENERATE: GENERATE}
)

# Добавляем переход без условий RETRIEVE -> GRADE_DOCUMENTS
flow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

# Добавляем условные переходы из GRADE_DOCUMENTS
flow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={WEBSEARCH: WEBSEARCH, GENERATE: GENERATE},
)

# Добавляем условные переходы из GENERATE
flow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    path_map={"useful": END, "not_useful": WEBSEARCH, "not_supported": GENERATE},
)

# Добавляем переходы между WEBSEARCH и GENERATE
flow.add_edge(WEBSEARCH, GENERATE)
flow.add_edge(GENERATE, END)

# Компилируем граф
app = flow.compile()
