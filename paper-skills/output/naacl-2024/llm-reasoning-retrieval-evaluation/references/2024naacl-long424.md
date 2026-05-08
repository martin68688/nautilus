---
title: "Deceptive Semantic Shortcuts on Reasoning Chains: How Far Can Models Go without Hallucination?"
source: "https://aclanthology.org/2024.naacl-long.424/"
categories: ['code-search-clone-detection-graph', 'llm-reasoning-retrieval-evaluation', 'knowledge-graph-hallucination-reduction']
tags: ['hallucination', 'reasoning', 'semantic_shortcuts', 'faithfulness']
venue: "NAACL 2024"
tldr: "Investigates how semantic associations can induce hallucinations and unfaithful reasoning chains in large language models."
---

# Deceptive Semantic Shortcuts on Reasoning Chains: How Far Can Models Go without Hallucination?

**Source**: [https://aclanthology.org/2024.naacl-long.424/](https://aclanthology.org/2024.naacl-long.424/)

**TLDR**: Investigates how semantic associations can induce hallucinations and unfaithful reasoning chains in large language models.

## Abstract

AbstractDespite the high performances of large language models (LLMs) across numerous benchmarks, recent research has unveiled their suffering from hallucinations and unfaithful reasoning. This work studies a type of hallucination induced by semantic associations. We investigate to what extent LLMs take shortcuts from certain keyword/entity biases in the prompt instead of following correct reasoning paths. To quantify this phenomenon, we propose a novel probing method and benchmark called EUREQA. EUREQA is an entity-searching task where a model finds a missing entity based on described multi-hop relations with other entities. These deliberately designed multi-hop relations create deceptive semantic associations, and models must stick to the correct reasoning path instead of incorrect shortcuts to find the correct answer.Experiments show that existing LLMs cannot follow correct reasoning paths and resist the attempt of greedy shortcuts, with GPT-4 only achieving 62% accuracy. Analyses provide further evidence that LLMs rely on semantic biases to solve the task instead of proper reasoning, questioning the validity and generalizability of current LLMs’ high performances.