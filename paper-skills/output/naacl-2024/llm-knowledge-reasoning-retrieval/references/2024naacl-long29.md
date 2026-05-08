---
title: "FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs"
source: "https://aclanthology.org/2024.naacl-long.29/"
categories: ['llm-knowledge-reasoning-retrieval', 'legal-ai-nlp-applications']
tags: ['task-oriented-dialogue', 'planning', 'constrained-decoding']
venue: "NAACL 2024"
tldr: "Proposes FLAP, a method for LLMs to perform workflow-based planning in dialogues using constrained decoding."
---

# FLAP: Flow-Adhering Planning with Constrained Decoding in LLMs

**Source**: [https://aclanthology.org/2024.naacl-long.29/](https://aclanthology.org/2024.naacl-long.29/)

**TLDR**: Proposes FLAP, a method for LLMs to perform workflow-based planning in dialogues using constrained decoding.

## Abstract

AbstractPlanning is a crucial task for agents in task oriented dialogs (TODs). Human agents typically resolve user issues by following predefined workflows, decomposing workflow steps into actionable items, and performing actions by executing APIs in order; all of which require reasoning and planning. With the recent advances in LLMs, there have been increasing attempts to use them for task planning and API usage. However, the faithfulness of the plans to predefined workflows and API dependencies, is not guaranteed with LLMs. Moreover, workflows in real life are often custom-defined and prone to changes; hence, adaptation is desirable. To study this, we propose the problem of faithful planning in TODs that needs to resolve user intents by following predefined flows and preserving API dependencies. To solve this problem, we propose FLAP, a Flow-Adhering Planning algorithm based on constrained decoding with lookahead heuristic for LLMs. Our algorithm alleviates the need for finetuning LLMs using domain specific (plan/dependency) data, enables quick adaptation to predefined flows, and outperforms other decoding and prompting-based baselines. Further, our algorithm empowers smaller LLMs (≈7B) to perform at par larger LLMs (≈30B-40B).