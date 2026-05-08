---
title: "QueryExplorer: An Interactive Query Generation Assistant for Search and Exploration"
source: "https://aclanthology.org/2024.naacl-demo.11/"
categories: ['llm-reasoning-retrieval-evaluation', 'query-aware-attention-user-modeling']
tags: ['information-retrieval', 'query-generation', 'interactive', 'assistant']
venue: "NAACL 2024"
tldr: "QueryExplorer is an interactive query generation assistant that helps users formulate search queries by allowing them to provide example documents and refine queries through interaction."
---

# QueryExplorer: An Interactive Query Generation Assistant for Search and Exploration

**Source**: [https://aclanthology.org/2024.naacl-demo.11/](https://aclanthology.org/2024.naacl-demo.11/)

**TLDR**: QueryExplorer is an interactive query generation assistant that helps users formulate search queries by allowing them to provide example documents and refine queries through interaction.

## Abstract

AbstractFormulating effective search queries remains a challenging task, particularly when users lack expertise in a specific domain or are not proficient in the language of the content. Providing example documents of interest might be easier for a user. However, such query-by-example scenarios are prone to concept drift, and the retrieval effectiveness is highly sensitive to the query generation method, without a clear way to incorporate user feedback. To enable exploration and to support Human-In-The-Loop experiments we propose QueryExplorer– an interactive query generation, reformulation, and retrieval interface with support for Hug-gingFace generation models and PyTerrier’sretrieval pipelines and datasets, and extensivelogging of human feedback. To allow users to create and modify effective queries, our demo supports complementary approaches of using LLMs interactively, assisting the user with edits and feedback at multiple stages of the query formulation process. With support for recording fine-grained interactions and user annotations, QueryExplorer can serve as a valuable experimental and research platform for annotation, qualitative evaluation, and conducting Human-in-the-Loop (HITL) experiments for complex search tasks where users struggle to formulate queries.