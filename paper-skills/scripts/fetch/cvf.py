"""CVF fetcher - CVPR and ICCV from openaccess.thecvf.com"""
from bs4 import BeautifulSoup
from tqdm import tqdm
from .base import ConferenceFetcher

BASE_URL = "https://openaccess.thecvf.com"

class CVFFetcher(ConferenceFetcher):
    def __init__(self, venue="CVPR", year=2024):
        super().__init__()
        self.venue = venue
        self.year = year

    def fetch(self):
        r = self._get(f"{BASE_URL}/{self.venue}{self.year}?day=all")
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.select("dt.ptitle a")]
        papers = []
        for link in tqdm(links, desc=f"{self.venue} {self.year}"):
            try:
                r = self._get(BASE_URL + link)
                s = BeautifulSoup(r.text, "html.parser")
                title = s.find("div", id="papertitle")
                abstract = s.find("div", id="abstract")
                pdf = s.find("a", string=lambda t: t and "pdf" in t.lower())
                if title and abstract:
                    papers.append(self._make_paper(
                        title=title.text,
                        abstract=abstract.text,
                        url=BASE_URL + link,
                        pdf_url=BASE_URL + pdf["href"] if pdf else "",
                        paper_id=link.split("/")[-1].replace(".html", ""),
                    ))
            except Exception as e:
                print(f"Failed {link}: {e}")
        return papers
