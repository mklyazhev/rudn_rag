import re

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough


class LlamaOutputParser:
    def __init__(self, start_substring, end_substring):
        self.start_substring = re.escape(start_substring)
        self.end_substring = re.escape(end_substring)
        self.pattern = re.compile(f"{self.start_substring}.*{self.end_substring}", re.DOTALL)

    def __call__(self, output):
        cleaned_output = re.sub(self.pattern, '', output).strip()
        return cleaned_output


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class RAG:
    def __init__(self, llm, retriever, prompt, parser="default"):
        # parser = StrOutputParser() if parser == "default" else LlamaOutputParser("<|begin_of_text|>", "<|end_header_id|>")
        # self.qa_chain = (
        #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
        #     | prompt
        #     | llm
        #     | parser
        # )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        chain_with_trimming = (
                RunnablePassthrough.assign(messages_trimmed=self.trim_messages)
                | conversational_rag_chain
        )
        self.chain = chain_with_trimming
        self.history = {}

    async def generate(self, question, user_id, stream=False):
        if stream:
            self.chain.stream(question)

        # return self.chain.invoke(question)
        return self.chain.invoke(
            {"input": question},
            config={
                "configurable": {"user_id": user_id}
            },
        )["answer"]

    def get_chat_history(self, user_id):
        if user_id not in self.history:
            self.history[user_id] = ChatMessageHistory()
        return self.history[user_id]

    def trim_messages(self, chain_input):
        stored_messages = chain_input.get_session_history().messages
        if len(stored_messages) <= 10:
            return False

        chain_input.get_session_history().clear()

        for message in stored_messages[-10:]:
            chain_input.get_session_history().add_message(message)

        return True
