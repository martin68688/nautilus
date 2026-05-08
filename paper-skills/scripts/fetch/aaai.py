"""AAAI fetcher - Semantic Scholar API"""
import time
from tqdm import tqdm
from .base import ConferenceFetcher

API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,abstract,url,openAccessPdf"

class AAAIFetcher(ConferenceFetcher):
    venue = "AAAI"

    def __init__(self, year=2024):
        super().__init__()
        self.year = year

    def fetch(self):
        papers = []
        token = None
        pbar = tqdm(desc=f"AAAI {self.year}")
        while True:
            params = {"venue": "AAAI", "year": self.year,
                      "fields": FIELDS, "limit": 500}
            if token:
                params["token"] = token
            r = self._get(API, params=params)
            data = r.json()
            for item in data.get("data", []):
                title = item.get("title", "")
                abstract = item.get("abstract", "")
                if not title or not abstract:
                    continue
                pdf = (item.get("openAccessPdf") or {}).get("url", "")
                papers.append(self._make_paper(
                    title=title,
                    abstract=abstract,
                    url=item.get("url", ""),
                    pdf_url=pdf,
                    paper_id=item["paperId"],
                ))
            pbar.update(len(data.get("data", [])))
            token = data.get("token")
            if not token:
                break
            time.sleep(1)
        pbar.close()
        return papers
