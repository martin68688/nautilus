"""NeurIPS fetcher - papers.nips.cc"""
from bs4 import BeautifulSoup
from tqdm import tqdm
from .base import ConferenceFetcher

BASE_URL = "https://papers.nips.cc"

class NeurIPSFetcher(ConferenceFetcher):
    venue = "NeurIPS"

    def __init__(self, year=2024):
        super().__init__()
        self.year = year

    def fetch(self):
        r = self._get(f"{BASE_URL}/paper_files/paper/{self.year}")
        soup = BeautifulSoup(r.text, "html.parser")
        links = [a["href"] for a in soup.select('a[href*="/paper/{}/hash"]'.format(self.year))]
        papers = []
        for link in tqdm(links, desc=f"NeurIPS {self.year}"):
            try:
                r = self._get(BASE_URL + link)
                s = BeautifulSoup(r.text, "html.parser")
                title = s.find("h1")
                paras = s.find_all("p")
                pdf = s.find("a", string="Paper")
                if title and len(paras) > 2:
                    papers.append(self._make_paper(
                        title=title.text,
                        abstract=paras[2].text,
                        url=BASE_URL + link,
                        pdf_url=BASE_URL + pdf["href"] if pdf else "",
                        paper_id=link.split("/")[-1].replace("-Abstract-Conference.html", ""),
                    ))
            except Exception as e:
                print(f"Failed {link}: {e}")
        return papers
