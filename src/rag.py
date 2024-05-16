from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


class RAG:
    def __init__(self, llm, retriever, prompt):
        self.qa_chain = (
            {"context": retriever | self.__format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def generate(self, question, stream=False):
        if stream:
            self.qa_chain.stream(question)

        return self.qa_chain.invoke(question)

    def __format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
