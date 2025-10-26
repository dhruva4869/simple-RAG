from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray: ...
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts)


class EmbeddingCache:
    def __init__(self): self.cache = {}
    def get(self, text: str) -> Optional[np.ndarray]: return self.cache.get(text)
    def set(self, text: str, emb: np.ndarray): self.cache[text] = emb
    def clear(self): self.cache.clear()
    def __len__(self): return len(self.cache)


class CachedEmbedder(BaseEmbedder):
    def __init__(self, embedder: BaseEmbedder):
        self.embedder, self.cache = embedder, EmbeddingCache()

    def embed_text(self, text: str) -> np.ndarray:
        return self.cache.get(text) or self.cache.set(text, self.embedder.embed_text(text)) or self.cache.get(text)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings, to_embed = [], [t for t in texts if t not in self.cache.cache]
        if to_embed:
            new_embs = self.embedder.embed_texts(to_embed)
            for t, e in zip(to_embed, new_embs): self.cache.set(t, e)
        return np.array([self.cache.get(t) for t in texts])
