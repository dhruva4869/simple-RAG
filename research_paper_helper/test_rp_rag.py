import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.data_parser import PDFReader
from rag.llm import GeminiLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import SimpleRerank
from rag.retrieval import VectorRetrieval
from rag.text_utils import text2chunk
from rag.embeddings import SentenceTransformerEmbedder, CachedEmbedder
from dotenv import load_dotenv
from test_rp_utils import download_research_paper_for_topic, get_file_names

load_dotenv()


def test_vector_rag():
    # topic = input("Enter your own research topic : ")
    topic = "RLHF"
    download_research_paper_for_topic(topic)
    sample_pdfs = get_file_names(topic)
    contents = PDFReader(pdf_paths=sample_pdfs).read()
    text = " ".join(contents)
    chunks = text2chunk(text, chunk_size=200, overlap=50)
    
    embedder = CachedEmbedder(SentenceTransformerEmbedder())
    
    vector_retrieval = VectorRetrieval(
        documents=chunks,
        persist_directory="../vector_db_test",
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
    
    # test query first one, it should not respond anything.
    test_queries = [
        "What is RLHF?",
        "What can RLHF be used for?"
    ]
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        response: Answer = pipeline.run(query)
        print(f"Answer: {response.answer}")

    while True:
        carry = input("Do you want to ask another question? (y/n): ")
        if carry == "y":
            query = input("Your question: ")
            response: Answer = pipeline.run(query)
            print(f"Answer: {response.answer}")
        else:
            break
    
test_vector_rag()