---
title: "ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems"
source: "https://aclanthology.org/2024.naacl-long.20/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-ranking-benchmarking-adaptation', 'language-model-evaluation-benchmarks']
tags: ['rag-evaluation', 'automated-evaluation', 'retrieval-augmented-generation']
venue: "NAACL 2024"
tldr: "Introduces ARES, an automated evaluation framework for retrieval-augmented generation systems."
---

# ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems

**Source**: [https://aclanthology.org/2024.naacl-long.20/](https://aclanthology.org/2024.naacl-long.20/)

**TLDR**: Introduces ARES, an automated evaluation framework for retrieval-augmented generation systems.

## Abstract

AbstractEvaluating retrieval-augmented generation (RAG) systems traditionally relies on hand annotations for input queries, passages to retrieve, and responses to generate. We introduce ARES, an Automated RAG Evaluation System, for evaluating RAG systems along the dimensions of context relevance, answer faithfulness, and answer relevance. By creating its own synthetic training data, ARES finetunes lightweight LM judges to assess the quality of individual RAG components. To mitigate potential prediction errors, ARES utilizes a small set of human-annotated datapoints for prediction-powered inference (PPI). Across eight different knowledge-intensive tasks in KILT, SuperGLUE, and AIS, ARES accurately evaluates RAG systems while using only a few hundred human annotations during evaluation. Furthermore, ARES judges remain effective across domain shifts, proving accurate even after changing the type of queries and/or documents used in the evaluated RAG systems. We make our code and datasets publicly available on Github.