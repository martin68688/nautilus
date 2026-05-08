"""ICML fetcher - proceedings.mlr.press"""
from bs4 import BeautifulSoup
from tqdm import tqdm
from .base import ConferenceFetcher

VOLUME_MAP = {2024: "v235", 2023: "v202", 2022: "v162"}
BASE_URL = "https://proceedings.mlr.press"

class ICMLFetcher(ConferenceFetcher):
    venue = "ICML"

    def __init__(self, year=2024):
        super().__init__()
        self.year = year

    def fetch(self):
        vol = VOLUME_MAP.get(self.year, "v235")
        r = self._get(f"{BASE_URL}/{vol}/")
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.select("div.paper a[href]")
                 if a["href"].endswith(".html") and "abs" not in a.text.lower()]
        papers = []
        for link in tqdm(links, desc=f"ICML {self.year}"):
            try:
                r = self._get(link)
                s = BeautifulSoup(r.text, "html.parser")
                title = s.find("h1")
                abstract = s.select_one("div#abstract")
                pdf = s.find("a", string=lambda t: t and "Download PDF" in t)
                if title and abstract:
                    papers.append(self._make_paper(
                        title=title.text,
                        abstract=abstract.text,
                        url=link,
                        pdf_url=pdf["href"] if pdf else "",
                        paper_id=link.split("/")[-1].replace(".html", ""),
                    ))
            except Exception as e:
                print(f"Failed {link}: {e}")
        return papers
