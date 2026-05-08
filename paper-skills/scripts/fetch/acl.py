"""ACL Anthology fetcher - ACL and NAACL"""
from bs4 import BeautifulSoup
from tqdm import tqdm
from .base import ConferenceFetcher

BASE_URL = "https://aclanthology.org"
EVENT_MAP = {
    ("ACL", 2024): "acl-2024",
    ("ACL", 2025): "acl-2025",
    ("NAACL", 2024): "naacl-2024",
    ("NAACL", 2025): "naacl-2025",
}

class ACLFetcher(ConferenceFetcher):
    def __init__(self, venue="ACL", year=2024):
        super().__init__()
        self.venue = venue
        self.year = year

    def fetch(self):
        event = EVENT_MAP.get((self.venue, self.year), f"{self.venue.lower()}-{self.year}")
        r = self._get(f"{BASE_URL}/events/{event}/")
        soup = BeautifulSoup(r.text, "html.parser")
        year_str = str(self.year)
        venue_key = self.venue.lower()
        links = list({a["href"] for a in soup.select("a[href]")
                      if a["href"].startswith("/") and year_str in a["href"]
                      and venue_key in a["href"]
                      and not a["href"].endswith(".bib")
                      and not a["href"].endswith(".pdf")
                      and "/people/" not in a["href"]
                      and "/volumes/" not in a["href"]})
        papers = []
        for link in tqdm(links, desc=f"{self.venue} {self.year}"):
            try:
                r = self._get(BASE_URL + link)
                s = BeautifulSoup(r.text, "html.parser")
                title = s.find("h2", id="title")
                abstract = s.find("div", class_="acl-abstract")
                pdf = s.find("a", class_="acl-button", href=lambda h: h and h.endswith(".pdf"))
                if title and abstract:
                    papers.append(self._make_paper(
                        title=title.text,
                        abstract=abstract.text,
                        url=BASE_URL + link,
                        pdf_url=pdf["href"] if pdf else "",
                        paper_id=link.strip("/").split("/")[-1],
                    ))
            except Exception as e:
                print(f"Failed {link}: {e}")
        return papers
