import requests
from bs4 import BeautifulSoup
import re

def scrape_to_text(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "header", "footer", "nav", "aside"]):
            element.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n+", "\n", text).strip()
        return text
    except Exception as e:
        return f"Error scraping {url}: {e}"

# print(scrape_to_text("https://en.wikipedia.org/wiki/Clash_Royale"))
