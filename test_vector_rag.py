import os
from rag.data_parser import PDFReader
from rag.llm import GeminiLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import SimpleRerank
from rag.retrieval import VectorRetrieval
from rag.text_utils import text2chunk
from rag.embeddings import SentenceTransformerEmbedder, CachedEmbedder
from dotenv import load_dotenv

load_dotenv()


def test_vector_rag():
    sample_pdf = "./documents/sample.pdf"
    contents = PDFReader(pdf_paths=[sample_pdf]).read()
    text = " ".join(contents)
    chunks = text2chunk(text, chunk_size=200, overlap=50)
    
    embedder = CachedEmbedder(SentenceTransformerEmbedder())
    
    vector_retrieval = VectorRetrieval(
        documents=chunks,
        persist_directory="./vector_db_test",
        collection_name="test_documents",
        embedder=embedder
    )
    
    print("Documents ingested into vector database")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = GeminiLLM(api_key=api_key)
    rerank = SimpleRerank()
    
    pipeline = SimpleRAGPipeline(
        retrieval=vector_retrieval, 
        llm=llm, 
        rerank=rerank,
        retrieval_top_k=10,
        rerank_top_k=3
    )

    test_queries = [
        "What can Ollama do?",
        "What is Clash Royale?"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        response: Answer = pipeline.run(query)
        print(f"Answer: {response.answer}")
    
    return pipeline

test_vector_rag()