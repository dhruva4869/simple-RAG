import os
from dotenv import load_dotenv
from rag.data_parser import PDFReader
from rag.llm import GeminiLLM
from rag.pipeline import Answer, SimpleRAGPipeline
from rag.rerank import SimpleRerank
from rag.retrieval import BM25Retrieval
from rag.text_utils import text2chunk
from rag.web_scraper import scrape_to_text

load_dotenv()

contents = scrape_to_text("https://en.wikipedia.org/wiki/Clash_Royale")
text = contents
chunks = text2chunk(text, chunk_size=200, overlap=50)
retrieval = BM25Retrieval(documents=chunks)
api_key = os.getenv("GOOGLE_API_KEY")
llm = GeminiLLM(api_key=api_key)
rerank = SimpleRerank()
pipeline = SimpleRAGPipeline(retrieval=retrieval, llm=llm, rerank=rerank)


def run(query: str) -> Answer:
    return pipeline.run(query)


def main():
    query = "What is Clash Royale?"
    print("Sample query:", query)
    response: Answer = pipeline.run(query)
    print(response.answer)
    while True:
        answer = input("Do you want to ask another question? (y/n): ")
        if answer == "y":
            query = input("Your question: ")
            response: Answer = run(query)
            print(response.answer)
            print()
        else:
            break

if __name__ == "__main__":
    main()