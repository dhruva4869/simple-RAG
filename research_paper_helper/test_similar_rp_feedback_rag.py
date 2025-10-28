import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.data_parser import PDFReader
from rag.llm import GeminiLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import SimpleRerank
from rag.retrieval import VectorRetrieval
from rag.text_utils import text2chunk
from rag.embeddings import SentenceTransformerEmbedder, CachedEmbedder
from dotenv import load_dotenv
from test_rp_utils import download_research_paper_for_topic, get_file_names, get_important_keywords

load_dotenv()


def test_vector_rag():
    draft_pdf_path = "./documents/Reinforcement_Learning_from_Human_Feedback_draft.pdf"
    
    print(f"Extracting keywords from draft PDF: {draft_pdf_path}")
    important_keywords = get_important_keywords(draft_pdf_path)
    print(f"Important keywords found: {important_keywords[:10]}")
    
    search_topics = []
    # random.shuffle(important_keywords)
    
    for keyword in important_keywords[:6]:
        clean_keyword = keyword.replace("_", " ").strip()
        search_topics.append(clean_keyword)
    
    top_keywords = important_keywords[:4]
    for i in range(len(top_keywords)):
        for j in range(i+1, len(top_keywords)):
            clean_kw1 = top_keywords[i].replace("_", " ").strip()
            clean_kw2 = top_keywords[j].replace("_", " ").strip()
            combination = f"{clean_kw1} AND {clean_kw2}"
            search_topics.append(combination)

    
    print(f"Search topics created: {search_topics[:10]}")
    
    all_pdfs = []
    for i, topic in enumerate(search_topics[:12]):
        folder_path = f"./research_papers/{topic}"
        if os.path.exists(folder_path):
            print(f"Skipping topic {i+1}/{min(12, len(search_topics))}: '{topic}' (folder already exists)")
            try:
                topic_pdfs = get_file_names(topic)
                all_pdfs.extend(topic_pdfs)
                print(f"  Found {len(topic_pdfs)} existing papers for '{topic}'")
            except Exception as e:
                print(f"Failed to read existing papers for topic '{topic}': {e}")
            continue
        
        print(f"Downloading papers for topic {i+1}/{min(12, len(search_topics))}: {topic}")
        try:
            download_research_paper_for_topic(topic)
            topic_pdfs = get_file_names(topic)
            all_pdfs.extend(topic_pdfs)
            print(f"  Found {len(topic_pdfs)} papers for '{topic}'")
        except Exception as e:
            print(f"Failed to download papers for topic '{topic}': {e}")
            continue
    
    all_pdfs.append(draft_pdf_path)
    
    print(f"Total papers collected: {len(all_pdfs)}")
    
    if not all_pdfs:
        print("No papers found. Exiting.")
        return
    
    contents = PDFReader(pdf_paths=all_pdfs).read()
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
        "What are some neural policy gradient methods?",
        "What is RLHF?",
        "How do Lifelong RL systems work?"
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