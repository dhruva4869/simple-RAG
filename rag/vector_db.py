import os
import uuid
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings

from .embeddings import BaseEmbedder, SentenceTransformerEmbedder


class VectorDatabase:
    def __init__(self, 
                 persist_directory: str = "./vector_db",
                 collection_name: str = "documents",
                 embedder: Optional[BaseEmbedder] = None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = embedder or SentenceTransformerEmbedder()
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        # collection = table storing embeddings, documents, and metadata
        try:
            self.collection = self.client.get_collection(collection_name)
        except ValueError:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Document embeddings for RAG system"}
            )
    
    def add_documents(self, 
                     documents: List[str], 
                     metadata: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        if not documents:
            return []
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        embeddings = self.embedder.embed_texts(documents)
        
        if metadata is None:
            metadata = [{"text": doc} for doc in documents]
        else:
            for i, meta in enumerate(metadata):
                if "text" not in meta:
                    meta["text"] = documents[i]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        
        return ids
    
    def search(self, 
               query: str, 
               top_k: int = 10,
               where: Optional[Dict[str, Any]] = None) -> Dict[str, List]:
       
        query_embedding = self.embedder.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where
        )
        
        return {
            "documents": results["documents"][0] if results["documents"] else [],
            "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            "distances": results["distances"][0] if results["distances"] else [],
            "ids": results["ids"][0] if results["ids"] else []
        }