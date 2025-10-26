# this is needed because after tokenization we will be having a bunch of documents and from those we need to find the top k
# that best suite the query. This finds those top documents / blocks of information that are actually useful

from abc import ABC
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

class BaseRetrieval(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def ingest(self, documents: list[str], metadata: Optional[list[dict]] = None) -> bool:
        raise NotImplementedError()
    
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        raise NotImplementedError()

    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        raise NotImplementedError

class BM25Retrieval(BaseRetrieval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        documents = kwargs.get("documents")
        metadata = kwargs.get("metadata")
        self.bm25 = None
        self.documents = None
        self.metadata = None
        self.__ingest__(documents, metadata)
    
    def __ingest__(self, documents: list[str], metadata: Optional[list[dict]] = None) -> bool:
        assert len(documents) > 0, "Documents is empty."
        tokens = [a.split() for a in documents]
        self.bm25 = BM25Okapi(tokens)
        self.metadata = metadata
        self.documents = documents
        return True
    
    def retrieve(self, query: str, top_k: int = 10):
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:top_k]
        metadata = None
        if self.metadata:
            metadata = [self.metadata[i] for i in top_n]
        docs = [self.documents[i] for i in top_n]
        return docs, metadata