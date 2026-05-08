"""
Re-classify each paper using LLM against the auto-generated cluster names.
Saves to cache/classified_papers.json
"""
import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")

BATCH_SIZE = 20

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def llm(prompt, max_tokens=512):
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="deepseek-chat",
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def classify_batch(papers, categories):
    cat_list = "\n".join(f"{i+1}. {c}" for i, c in enumerate(categories))
    entries = "\n\n".join(
        f"[{i+1}] {p['title']}\n{p['abstract'][:300]}"
        for i, p in enumerate(papers)
    )
    text = llm(
        f"Categories:\n{cat_list}\n\n"
        f"For each paper, output exactly one line in this format:\n"
        f"CATS:1,3 | TAGS:tag1,tag2,tag3 | TLDR:one sentence summary\n\n"
        f"{entries}",
        max_tokens=1024,
    )
    results = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        cats, tags, tldr = [], [], ""
        for part in line.split("|"):
            part = part.strip()
            if part.startswith("CATS:"):
                for token in part[5:].split(","):
                    try:
                        idx = int(token.strip()) - 1
                        if 0 <= idx < len(categories):
                            cats.append(categories[idx])
                    except ValueError:
                        pass
            elif part.startswith("TAGS:"):
                tags = [t.strip() for t in part[5:].split(",") if t.strip()]
            elif part.startswith("TLDR:"):
                tldr = part[5:].strip()
        results.append({
            "categories": cats or [categories[0]],
            "tags": tags,
            "tldr": tldr,
        })
    while len(results) < len(papers):
        results.append({"categories": [categories[0]], "tags": [], "tldr": ""})
    return results[:len(papers)]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="neurips")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    papers_path = os.path.join(CACHE_DIR, f"{args.venue}_{args.year}_papers.json")
    clusters_path = os.path.join(CACHE_DIR, f"{args.venue}_{args.year}_clusters.json")
    output_path = os.path.join(CACHE_DIR, f"{args.venue}_{args.year}_classified.json")

    with open(papers_path) as f:
        papers = json.load(f)
    with open(clusters_path) as f:
        clusters_data = json.load(f)

    categories = list(clusters_data["clusters"].keys())

    classified = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            classified = json.load(f)
    done = len(classified)

    for i in tqdm(range(done, len(papers), BATCH_SIZE), desc="Classifying"):
        batch = papers[i:i + BATCH_SIZE]
        label_lists = classify_batch(batch, categories)
        for paper, result in zip(batch, label_lists):
            classified.append({**paper, **result})
        with open(output_path, "w") as f:
            json.dump(classified, f, indent=2)

    print(f"Classified {len(classified)} papers into {len(categories)} categories")


if __name__ == "__main__":
    main()
