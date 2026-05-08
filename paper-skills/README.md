# paper-skills

Scrapes papers from 8 major ML/AI conferences, clusters them by topic, and generates structured knowledge bases (skills) for use with AI agents.

## What it does

1. Fetches paper titles and abstracts from conference proceedings
2. Embeds abstracts and clusters them into ~80 topic categories
3. Classifies each paper into 1-3 categories with tags and a one-line summary
4. Writes a `SKILL.md` index + individual reference files per category

Output lives in `output/{venue}-{year}/`.

## Supported conferences

| Conference | Source |
|------------|--------|
| NeurIPS | papers.nips.cc |
| ICML | proceedings.mlr.press |
| CVPR / ICCV | openaccess.thecvf.com |
| ACL / NAACL | aclanthology.org |
| ICLR | OpenReview API |
| AAAI | Semantic Scholar API |

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env` and fill in your keys:

```
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=...
OPENREVIEW_USERNAME=...   # only needed for ICLR
OPENREVIEW_PASSWORD=...
```

## Usage

```bash
# Single conference
bash run_all.sh neurips 2024

# All conferences
for venue in neurips icml cvpr acl naacl iclr aaai; do
  bash run_all.sh $venue 2024
done
```

Or run steps individually:

```bash
python scripts/1_fetch.py --venue icml --year 2024
python scripts/2_embed_cluster.py --venue icml --year 2024
python scripts/3_classify.py --venue icml --year 2024
python scripts/4_generate_skills.py --venue icml --year 2024
```

Steps 2-4 resume automatically if interrupted.

To adjust the number of topic clusters, edit `N_CLUSTERS` in `scripts/2_embed_cluster.py` (default: 80). Higher = finer granularity.

## Output structure

```
output/
└── neurips-2024/
    ├── diffusion-models-image-generation/
    │   ├── SKILL.md
    │   └── references/
    │       ├── abc123.md
    │       └── ...
    └── ...
```

Each `SKILL.md` contains a description and an index table (title, tags, file).
Each `references/*.md` contains the abstract, tags, TLDR, and source URL.

## Agents

`agents/` contains 8 subagent definitions (one per conference) for use with Claude Code. Each agent scans the relevant `output/{venue}-*/` directory and returns extracted paper content for a given research query.
