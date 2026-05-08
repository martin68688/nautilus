"""ICLR fetcher - OpenReview API (requires login)"""
import os
from tqdm import tqdm
from dotenv import load_dotenv
from .base import ConferenceFetcher

load_dotenv("/Users/haoming/Downloads/paper-skills/.env")

VENUEID_MAP = {
    2024: "ICLR.cc/2024/Conference",
    2025: "ICLR.cc/2025/Conference",
}

class ICLRFetcher(ConferenceFetcher):
    venue = "ICLR"

    def __init__(self, year=2024):
        super().__init__()
        self.year = year

    def fetch(self):
        import openreview
        client = openreview.api.OpenReviewClient(
            baseurl="https://api2.openreview.net",
            username=os.getenv("OPENREVIEW_USERNAME"),
            password=os.getenv("OPENREVIEW_PASSWORD"),
        )
        venueid = VENUEID_MAP.get(self.year, f"ICLR.cc/{self.year}/Conference")
        notes = client.get_all_notes(content={"venueid": venueid})
        papers = []
        for note in tqdm(notes, desc=f"ICLR {self.year}"):
            c = note.content
            title = c.get("title", {}).get("value", "")
            abstract = c.get("abstract", {}).get("value", "")
            if not title or not abstract:
                continue
            papers.append(self._make_paper(
                title=title,
                abstract=abstract,
                url=f"https://openreview.net/forum?id={note.id}",
                pdf_url=f"https://openreview.net/pdf?id={note.id}",
                paper_id=note.id,
            ))
        return papers
