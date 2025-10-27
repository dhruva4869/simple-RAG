from abc import ABC, abstractmethod
from .llm import BaseLLM
from .prompt import ANSWER_PROMPT
from .rerank import BaseRerank
from .retrieval import BaseRetrieval


class Answer:
    def __init__(self, answer: str, contexts: list[str]):
        self.answer, self.contexts = answer, contexts


class Pipeline(ABC):
    @abstractmethod
    def run(self, query: str) -> Answer:
        ...


class SimpleRAGPipeline(Pipeline):
    def __init__(self, retrieval: BaseRetrieval, llm: BaseLLM,
                 rerank: BaseRerank | None = None,
                 retrieval_top_k: int = 100, rerank_top_k: int = 3):
        for name, obj, base in [("retrieval", retrieval, BaseRetrieval),
                                ("llm", llm, BaseLLM),
                                ("rerank", rerank, BaseRerank)]:
            if obj and not isinstance(obj, base):
                raise TypeError(f"{name} must be instance of {base.__name__}")

        self.retrieval, self.llm, self.rerank = retrieval, llm, rerank
        self.retrieval_top_k, self.rerank_top_k = retrieval_top_k, rerank_top_k

    def run(self, query: str) -> Answer:
        docs, metadata = self.retrieval.retrieve(query, top_k=self.retrieval_top_k)
        if self.rerank:
            docs, _ = self.rerank.rerank(query, docs, top_k=self.rerank_top_k)
        prompt = ANSWER_PROMPT.format(query=query, context="\n".join(docs))
        return Answer(self.llm.generate(prompt), docs)
