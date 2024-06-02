import re

from langchain_core.output_parsers import StrOutputParser
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
    def __init__(self, llm, retriever, prompt, use_chatgpt=False):
        parser = StrOutputParser() if use_chatgpt else LlamaOutputParser("<|begin_of_text|>", "<|end_header_id|>")
        self.qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | parser
        )

    def generate(self, question, stream=False):
        if stream:
            self.qa_chain.stream(question)

        return self.qa_chain.invoke(question)
