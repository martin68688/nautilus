"""
Embed abstracts, hierarchical cluster, then use LLM to name each cluster.
Saves to cache/clusters.json

Tune: N_CLUSTERS (bigger = finer granularity)
"""
import json
import os
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("/Users/haoming/Downloads/paper-skills/.env")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")

N_CLUSTERS = 80  # <-- adjust for granularity


def embed_abstracts(papers):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    return embeddings


def cluster(embeddings):
    clustering = AgglomerativeClustering(n_clusters=N_CLUSTERS, metric="cosine", linkage="average")
    labels = clustering.fit_predict(embeddings)
    return labels


def name_cluster(client, titles):
    sample = "\n".join(f"- {t}" for t in titles[:20])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=64,
        messages=[{"role": "user", "content": (
            "Given these paper titles from the same research cluster, "
            "give a concise category name (3-7 words, lowercase, hyphen-separated):\n"
            f"{sample}\n\nCategory name:"
        )}]
    )
    return resp.choices[0].message.content.strip().strip('"`').lower().replace(" ", "-")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="neurips")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    papers_path = os.path.join(CACHE_DIR, f"{args.venue}_{args.year}_papers.json")
    clusters_path = os.path.join(CACHE_DIR, f"{args.venue}_{args.year}_clusters.json")

    with open(papers_path) as f:
        papers = json.load(f)

    print(f"Loaded {len(papers)} papers")
    embeddings = embed_abstracts(papers)
    labels = cluster(embeddings)

    clusters = {}
    for i, (paper, label) in enumerate(zip(papers, labels)):
        clusters.setdefault(label, []).append(i)

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )
    named = {}
    for label, indices in tqdm(clusters.items(), desc="Naming clusters"):
        titles = [papers[i]["title"] for i in indices]
        name = name_cluster(client, titles)
        named[name] = indices

    with open(clusters_path, "w") as f:
        json.dump({"clusters": named, "paper_count": len(papers)}, f, indent=2)

    print(f"Saved {len(named)} clusters -> {clusters_path}")


if __name__ == "__main__":
    main()
