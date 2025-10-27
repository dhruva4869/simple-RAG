import arxiv
import os
from os.path import isfile, join

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