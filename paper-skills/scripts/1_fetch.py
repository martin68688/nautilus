"""Unified fetcher. Usage: python 1_fetch.py --venue neurips --year 2024"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(__file__))

from fetch.neurips import NeurIPSFetcher
from fetch.icml import ICMLFetcher
from fetch.cvf import CVFFetcher
from fetch.acl import ACLFetcher
from fetch.iclr import ICLRFetcher
from fetch.aaai import AAAIFetcher

FETCHERS = {
    "neurips": lambda y: NeurIPSFetcher(y),
    "icml":    lambda y: ICMLFetcher(y),
    "cvpr":    lambda y: CVFFetcher("CVPR", y),
    "iccv":    lambda y: CVFFetcher("ICCV", y),
    "acl":     lambda y: ACLFetcher("ACL", y),
    "naacl":   lambda y: ACLFetcher("NAACL", y),
    "iclr":    lambda y: ICLRFetcher(y),
    "aaai":    lambda y: AAAIFetcher(y),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="neurips",
                        choices=list(FETCHERS) + ["all"])
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    venues = list(FETCHERS) if args.venue == "all" else [args.venue]
    cache_dir = os.path.join(os.path.dirname(__file__), "../cache")
    os.makedirs(cache_dir, exist_ok=True)

    for venue in venues:
        out = os.path.join(cache_dir, f"{venue}_{args.year}_papers.json")
        if os.path.exists(out):
            print(f"[skip] {out} already exists")
            continue
        print(f"Fetching {venue} {args.year}...")
        papers = FETCHERS[venue](args.year).fetch()
        with open(out, "w") as f:
            json.dump(papers, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(papers)} papers -> {out}")

if __name__ == "__main__":
    main()
