"""
Base class for all conference fetchers.
All fetchers return a list of dicts with these fields:
  title, abstract, url, pdf_url, id, venue, year
"""
import time
import requests


class ConferenceFetcher:
    venue: str = ""
    year: int = 2024

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "Mozilla/5.0"

    def fetch(self) -> list[dict]:
        raise NotImplementedError

    def _get(self, url, **kwargs):
        r = self.session.get(url, timeout=15, **kwargs)
        r.raise_for_status()
        time.sleep(0.1)
        return r

    def _make_paper(self, title, abstract, url, pdf_url="", paper_id=""):
        return {
            "title": title.strip(),
            "abstract": abstract.strip(),
            "url": url,
            "pdf_url": pdf_url,
            "id": paper_id or url.split("/")[-1].split(".")[0],
            "venue": self.venue,
            "year": self.year,
        }
