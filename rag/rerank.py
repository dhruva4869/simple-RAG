# time to rerank our documents, metadata pairs that we generated
# this is to get the better results on top of the bm25 that we just did

import re
from abc import ABC

class BaseRerank(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> tuple[list, list]:
        raise NotImplementedError


class SimpleRerank(BaseRerank):
    """
    Simple reranking based on keyword overlap and document length.
    This avoids dependency issues with CrossEncoder.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def rerank(self, query: str, documents: list[str], top_k: int = 10) -> tuple[list, list]:
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored_docs = []
        for i, doc in enumerate(documents):
            doc_words = set(re.findall(r'\b\w+\b', doc.lower()))
            overlap = len(query_words.intersection(doc_words))
            overlap_score = overlap / len(query_words) if query_words else 0
            length_penalty = 1.0 / (1.0 + len(doc.split()) / 100.0)
            score = overlap_score * length_penalty
            
            scored_docs.append((i, score, doc))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        relevants = []
        scores = []
        for i, score, doc in scored_docs[:top_k]:
            if score > 0:
                relevants.append(doc)
                scores.append(score)
        
        return relevants, scores