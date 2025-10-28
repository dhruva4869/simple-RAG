import arxiv
import os
from os.path import isfile, join
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import yake
from sentence_transformers import SentenceTransformer, util


def download_research_paper_for_topic(topic_name: str):
    research_paper = topic_name
    if not os.path.exists("research_papers"):
        os.mkdir("research_papers")
    folder_name = f"./research_papers/{topic_name}"
    client = arxiv.Client()
    search = arxiv.Search(
        query = research_paper,
        max_results = 2,
    )
    results = client.results(search)
    entry_ids = [r.entry_id.split("/")[-1] for r in results]
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for entry_id in entry_ids:
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[entry_id])))
        paper.download_pdf(dirpath=folder_name)

# download_research_paper_for_topic("RLHF")

def get_file_names(topic_name):
    path = f"./research_papers/{topic_name}"
    files = [f"{path}/{f}" for f in os.listdir(path) if isfile(join(path, f))]
    return files

# print(get_file_names("RLHF"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')
def get_important_keywords(research_paper):
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rag.data_parser import PDFReader

    text = PDFReader(pdf_paths=[research_paper]).read()
    text = " ".join(text)
    
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=1,
        top=100,
        features=None
    )
    yake_keywords = [kw for kw, antiscore in kw_extractor.extract_keywords(text)]
    title = " ".join(research_paper.split("/")[-1].replace("_draft.pdf", "").split("_")).lower()
    # print(title)
    title_emb = embedder.encode(title, convert_to_tensor=True)
    kw_embs = embedder.encode(yake_keywords, convert_to_tensor=True)
    sims = util.cos_sim(title_emb, kw_embs)[0]
    ranked = sorted(zip(yake_keywords, sims.tolist()), key=lambda x: x[1], reverse=True)
    ranked = [k for k, score in ranked if score > 0.2]
    return ranked

# TODO - the title always has to be perfect otherwise it just won't work
# because it is a draft research paper, we can assume the person uploading it will give the perfect title
# print(get_important_keywords("./documents/Reinforcement_Learning_from_Human_Feedback_draft.pdf"))