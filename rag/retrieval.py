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


class VectorRetrieval(BaseRetrieval):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .vector_db import VectorDatabase
        from .embeddings import SentenceTransformerEmbedder
        
        persist_directory = kwargs.get("persist_directory", "./vector_db")
        collection_name = kwargs.get("collection_name", "documents")
        embedder = kwargs.get("embedder", SentenceTransformerEmbedder())
        
        self.vector_db = VectorDatabase(
            persist_directory=persist_directory,
            collection_name=collection_name,
            embedder=embedder
        )
        
        documents = kwargs.get("documents")
        metadata = kwargs.get("metadata")
        if documents:
            self.ingest(documents, metadata)
    
    def ingest(self, documents: list[str], metadata: Optional[list[dict]] = None) -> bool:
        if not documents:
            return False
        
        try:
            self.vector_db.add_documents(documents, metadata)
            return True
        except Exception as e:
            print(f"Error ingesting documents: {e}")
            return False
    
    def retrieve(self, query: str, top_k: int = 10):
        try:
            results = self.vector_db.search(query, top_k=top_k)
            docs = results["documents"]
            metadata = results["metadatas"]
            return docs, metadata
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return [], []
    
    def rerank(self, query: str, documents: list[dict], top_k: int = 10) -> list[dict]:
        return documents[:top_k]