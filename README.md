# Nautilus — LLM-Driven ML Competition Solver with Paper-Informed Knowledge

Nautilus is an automated machine learning system that uses LLM agents to solve Kaggle-style ML competitions. It combines tree-search-based solution evolution with a structured knowledge base distilled from top-tier ML/AI conference papers, enabling the system to start with strong baselines and systematically improve through guided search.

## System Overview

```
nautilus/
├── mlevolve/          # Core ML solver engine
├── paper-skills/      # Paper knowledge extraction & structured KB
├── claudecode-memory/ # Cross-session memory (private)
└── claude_config/     # Claude Code config (private)
```

## MLEvolve — Core Solver

An LLM-powered agent that generates, debugs, and iteratively improves ML solutions through tree search.

### Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           AgentSearch (Coordinator)      │
                    │  node_selection → agent_step → evaluate  │
                    └────────┬────────────────────┬───────────┘
                             │                    │
              ┌──────────────┼───────────────┐    │
              ▼              ▼               ▼    ▼
         draft_agent   improve_agent    debug_agent   ...
              │              │               │
              ▼              ▼               ▼
         Interpreter (subprocess execution, CPU/GPU isolation)
              │
              ▼
         result_parse → evaluation → backpropagate
```

### Agent Types

| Agent | Trigger | Role |
|-------|---------|------|
| **draft_agent** | Root node expansion | Generate initial solution from scratch |
| **improve_agent** | Healthy parent node | Single-step improvement with 3-tier strategy |
| **debug_agent** | Buggy parent node | Fix bugs via SEARCH/REPLACE diff or full rewrite |
| **evolution_agent** | Branch stagnation (early) | Improve based on branch evolution trajectory |
| **fusion_agent** | Branch stagnation (late) | Cross-branch knowledge fusion |
| **aggregation_agent** | Draft limit reached | Multi-branch synthesis into new solution |

### Search Strategy

- **Phase 1 (Exploration)**: UCT-based tree search with decaying exploration constant
- **Phase 2 (Exploitation)**: Soft-switch to Top-K weighted selection from best nodes across branches
- **Stagnation detection**: Triggers 3-tier improvement (Optimization → Representation → Paradigm shift)
- **Multi-branch parallelism**: Independent solution branches evolve concurrently

### Knowledge Injection (Coldstart)

Before the search begins, task-specific knowledge is injected into agent prompts:

1. **Model template** (`engine/coldstart/models_guidance_classified.json`): Recommended pretrained models + code templates matched by task category
2. **Static methodology** (`engine/coldstart/methodology_map.json` → `experience_kb/*/experience_methodology.md`): Curated POSITIVE technique entries from past experiments
3. **Dynamic methodology** (`engine/coldstart/methodology_agent.py`): LLM selects relevant categories from experience_kb, reads HIGH-confidence references

### Code Generation Modes

- **Full rewrite**: LLM generates complete Python script
- **Diff mode** (`use_diff_mode: True`): Planner plans → SEARCH/REPLACE patcher applies incremental changes; falls back to full rewrite on failure
- **Stepwise mode** (draft stage): Separate plan generation → code generation

### Quick Start

```bash
cd mlevolve

# Run on a single competition task
bash run_single_task.sh <EXP_ID> <DATASET_DIR>

# Example: Spooky Author Identification
bash run_single_task.sh spooky-author-identification ./data
```

### Configuration

Key settings in `mlevolve/config/config.yaml`:

```yaml
agent:
  steps: 80                    # Total search steps
  initial_drafts: 3            # Initial draft solutions
  time_limit: 21600            # 6h per run
  use_diff_mode: true          # SEARCH/REPLACE patch mode
  fusion_vs_evolution_prob: 0.3

  search:
    parallel_search_num: 6     # Concurrent execution slots
    num_gpus: 3
    num_drafts: 5              # Max drafts per root
    num_improves: 3            # Max improve attempts per node

# LLM backend
agent.code.model: deepseek-v4-flash
agent.feedback.model: deepseek-v4-flash

# Knowledge base path
methodology_kb_path: "../paper-skills/experience_kb"
methodology_dynamic: true      # true = LLM matching, false = static map
```

### Directory Structure

```
mlevolve/
├── run.py                        # Entry point
├── run_single_task.sh            # Single-task runner with grading server
├── config/
│   ├── __init__.py               # Config loading, workspace prep
│   └── config.yaml               # Main configuration
├── engine/
│   ├── agent_search.py           # Search coordinator
│   ├── search_node.py            # SearchNode + Journal (tree structure)
│   ├── executor.py               # Subprocess code executor
│   ├── node_selection.py         # UCT + Top-K selection
│   ├── evaluation.py             # Reward + backpropagation
│   ├── execution.py              # Post-execution validation
│   ├── solution_manager.py       # Best solution tracking
│   ├── conditions.py             # Stagnation / fusion trigger conditions
│   └── coldstart/
│       ├── knowledge.py          # Guidance builder (model + methodology)
│       ├── methodology_agent.py  # Dynamic LLM-based category matching
│       ├── methodology_map.json  # Task → category mapping (static)
│       ├── models_guidance_classified.json  # Model templates by category
│       └── competition_tag_classified.json  # Task → category mapping
├── agents/
│   ├── draft_agent.py            # Initial solution generation
│   ├── improve_agent.py          # Solution improvement
│   ├── debug_agent.py            # Bug fixing
│   ├── evolution_agent.py        # Trajectory-guided evolution
│   ├── fusion_agent.py           # Cross-branch fusion
│   ├── aggregation_agent.py      # Multi-branch aggregation
│   ├── code_review_agent.py      # Code review pass
│   ├── result_parse_agent.py     # Metric extraction from output
│   ├── data_leakage_agent.py     # Leakage detection
│   ├── planner/                  # Two-stage plan-then-code pipeline
│   ├── coder/                    # Code generation (full + diff modes)
│   ├── memory/                   # Global memory (embedding-based retrieval)
│   ├── prompts/                  # Shared prompt fragments
│   └── triggers.py               # Patience counters, branch registration
├── llm/
│   ├── openai.py                 # OpenAI-compatible API backend
│   ├── gemini.py                 # Gemini backend + prompt compiler
│   └── model_profiles.py         # Per-model parameter profiles
├── utils/
│   ├── metric.py                 # MetricValue with maximize/minimize
│   ├── data_preview.py           # Dataset summary generation
│   ├── submission_fusion_utils.py # Post-run ensemble of top solutions
│   └── ...
└── inference/                    # Saved inference scripts from past runs
```

## Paper-Skills — Knowledge Extraction Pipeline

Scrapes papers from 8 major ML/AI conferences, clusters by topic, and generates structured knowledge bases for MLEvolve's coldstart system.

### Pipeline

```bash
cd paper-skills

# Single conference
bash run_all.sh neurips 2024

# All conferences
for venue in neurips icml cvpr acl naacl iclr aaai; do
  bash run_all.sh $venue 2024
done
```

Steps run individually:

```bash
python scripts/1_fetch.py --venue icml --year 2024       # Scrape papers
python scripts/2_embed_cluster.py --venue icml --year 2024 # Embed + cluster (~80 topics)
python scripts/3_classify.py --venue icml --year 2024     # Classify papers into categories
python scripts/4_generate_skills.py --venue icml --year 2024  # Generate SKILL.md + references
```

Steps 2-4 resume automatically if interrupted.

### Experience KB Plugins

Four plugins produce the experience knowledge base consumed by MLEvolve:

| Plugin | Role |
|--------|------|
| `plugin_a_methodology.py` | Map task to relevant methodology categories |
| `plugin_a2_insighter.py` | Extract structured insights from experiment trajectories |
| `plugin_b_experience.py` | Collect experiment experience records |
| `plugin_c_dreamer.py` | Generate creative hypothesis-driven insights |

### Directory Structure

```
paper-skills/
├── README.md
├── run_all.sh                    # Full pipeline runner
├── scripts/
│   ├── 1_fetch.py                # Paper scraping
│   ├── 2_embed_cluster.py        # Embedding + clustering
│   ├── 3_classify.py             # LLM classification
│   ├── 4_generate_skills.py      # Skill file generation
│   └── plugin_*.py               # Experience KB plugins
├── experience_kb/                # ← Consumed by MLEvolve coldstart
│   ├── small-data-transformer-finetuning/
│   │   ├── insight.md            # Insight index table
│   │   ├── experience_methodology.md  # POS entries for static path
│   │   └── references/           # 15 individual insight files
│   ├── winning-recipe-nlp-classification/
│   │   ├── insight.md
│   │   ├── experience_methodology.md
│   │   └── references/           # 12 insight files
│   └── ensemble-diversity-vs-validation-gap/
│       ├── insight.md
│       ├── experience_methodology.md
│       └── references/           # 12 insight files
├── methodology_kb/               # Paper-derived methodology (NAACL, manual, etc.)
│   └── paperinsight/
│       ├── naacl-2024/           # Auto-extracted from NAACL 2024
│       └── manual-2024/          # Curated BERT authorship attribution papers
├── agents/                       # Claude Code subagents for per-conference search
└── output/                       # Generated SKILL.md files
```

## Data Flow

```
paper-skills/experience_kb/ ──→ mlevolve coldstart ──→ agent prompt injection
paper-skills/methodology_kb/ ──→ mlevolve coldstart ──→ agent prompt injection
                ↓
        MLEvolve search loop
        ┌─ draft → improve → improve → ... (branch 1)
        ├─ draft → improve → debug → ...  (branch 2)
        ├─ draft → evolution → ...        (branch 3)
        └─ aggregation(fusion of branches)
                ↓
        runs/<timestamp>_<exp_id>/
        ├── best_solution.py
        ├── workspace/
        └── logs/
```

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU(s)
- OpenAI-compatible API key

### Install

```bash
pip install -r requirements.txt  # (if available)
# Core dependencies: torch, transformers, omegaconf, openai, rich, coolname, dataclasses-json
```

### Environment Variables

```bash
# LLM API (set in config.yaml or env)
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://apizh.net/v1
export OPENAI_MODEL=gpt-5.6-sol

# Optional: HuggingFace cache
export HF_DATASETS_CACHE=/path/to/hf_cache
export HF_MODELS_CACHE=/path/to/hf_cache

# Optional: Proxy
export http_proxy=http://YOUR_PROXY:PORT
export https_proxy=http://YOUR_PROXY:PORT
```

## Contributing

### Key Files to Understand

1. **`mlevolve/engine/agent_search.py`** — The central orchestrator; start here
2. **`mlevolve/agents/draft_agent.py`** — How solutions are generated
3. **`mlevolve/engine/node_selection.py`** — How the search tree is traversed
4. **`mlevolve/engine/coldstart/knowledge.py`** — How knowledge is injected
5. **`mlevolve/config/config.yaml`** — All tunable parameters

### Adding a New Agent

1. Create `mlevolve/agents/your_agent.py` with a `run(agent, parent_node) -> SearchNode` function
2. Import and call from `agent_search.py._run_single_step()`
3. Add trigger logic in the appropriate condition

### Adding Experience KB Entries

1. Add a new category directory under `paper-skills/experience_kb/<category>/`
2. Create `insight.md` with the index table (columns: #, Insight, Summary, Confidence, Reference)
3. Add individual reference files under `references/`
4. Create `experience_methodology.md` with `[POSITIVE]` sections for static path
5. Update `mlevolve/engine/coldstart/methodology_map.json` to map tasks to the new category

### Adding Model Templates

1. Add entries to `mlevolve/engine/coldstart/models_guidance_classified.json` under the appropriate category
2. Include `Description`, `Code_template` fields
3. Map the task to the category in `competition_tag_classified.json`
