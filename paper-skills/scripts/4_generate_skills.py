"""Generate Skill files from classified papers."""
import json
import os
import re
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv("/Users/haoming/Downloads/paper-skills/.env")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "../cache")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def slugify(text):
    return re.sub(r"[^a-z0-9-]", "", text.lower().replace(" ", "-"))


def generate_description(category, papers):
    titles = "\n".join(f"- {p['title']}" for p in papers[:15])
    resp = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=80,
        messages=[{"role": "user", "content": (
            f"Given these paper titles from the '{category}' research area, "
            f"write a 1-2 sentence description of what this skill covers. "
            f"Be specific about methods, tasks, and applications. No fluff:\n{titles}"
        )}]
    )
    return resp.choices[0].message.content.strip()


def make_reference(paper):
    url = paper.get("url") or paper.get("forum", "")
    categories = paper.get("categories") or [paper.get("category", "")]
    tags = paper.get("tags", [])
    tldr = paper.get("tldr", "")
    venue = paper.get("venue", "")
    year = paper.get("year", "")

    return f"""---
title: "{paper['title'].replace('"', "'")}"
source: "{url}"
categories: {categories}
tags: {tags}
venue: "{venue} {year}"
tldr: "{tldr}"
---

# {paper['title']}

**Source**: [{url}]({url})

**TLDR**: {tldr}

## Abstract

{paper['abstract']}"""


def make_skill_md(category, papers, description):
    rows = "\n".join(
        f"| {i+1} | {p['title'][:60]} | {', '.join(p.get('tags', [])[:3])} | {slugify(p['id'])}.md |"
        for i, p in enumerate(papers)
    )
    return f"""---
name: {category}
description: >-
  {description}
---

# {category.replace("-", " ").title()}

{description}

## Entry Index

| # | Title | Tags | File |
|---|-------|------|------|
{rows}
"""


def generate(classified_path, output_root):
    with open(classified_path) as f:
        papers = json.load(f)

    by_category = {}
    for p in papers:
        for cat in p.get("categories", [p.get("category", "uncategorized")]):
            by_category.setdefault(cat, []).append(p)

    for category, cat_papers in tqdm(by_category.items(), desc="Generating skills"):
        cat_slug = slugify(category)
        cat_dir = os.path.join(output_root, cat_slug)
        ref_dir = os.path.join(cat_dir, "references")
        os.makedirs(ref_dir, exist_ok=True)

        description = generate_description(category, cat_papers)
        with open(os.path.join(cat_dir, "SKILL.md"), "w") as f:
            f.write(make_skill_md(category, cat_papers, description))

        for paper in cat_papers:
            fname = slugify(paper["id"]) + ".md"
            with open(os.path.join(ref_dir, fname), "w") as f:
                f.write(make_reference(paper))

    print(f"Generated {len(by_category)} skills in {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", default="neurips")
    parser.add_argument("--year", type=int, default=2024)
    args = parser.parse_args()

    classified_path = os.path.join(CACHE_DIR, f"{args.venue}_{args.year}_classified.json")
    output_root = os.path.join(os.path.dirname(__file__), "../output", f"{args.venue}-{args.year}")
    generate(classified_path, output_root)
