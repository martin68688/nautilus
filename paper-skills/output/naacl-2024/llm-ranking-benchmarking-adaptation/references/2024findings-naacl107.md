---
title: "It’s All Relative! – A Synthetic Query Generation Approach for Improving Zero-Shot Relevance Prediction"
source: "https://aclanthology.org/2024.findings-naacl.107/"
categories: ['llm-knowledge-reasoning-retrieval', 'clinical-nlp-biomedical-applications', 'llm-ranking-benchmarking-adaptation']
tags: ['retrieval', 'synthetic-data', 'zero-shot']
venue: "NAACL 2024"
tldr: "Proposes a synthetic query generation method that creates relative comparisons to improve zero-shot relevance prediction for IR models."
---

# It’s All Relative! – A Synthetic Query Generation Approach for Improving Zero-Shot Relevance Prediction

**Source**: [https://aclanthology.org/2024.findings-naacl.107/](https://aclanthology.org/2024.findings-naacl.107/)

**TLDR**: Proposes a synthetic query generation method that creates relative comparisons to improve zero-shot relevance prediction for IR models.

## Abstract

AbstractLarge language models (LLMs) have shown promising ability to generate synthetic query-document pairs by prompting with as few as 8 demonstrations. This has enabled building better IR models, especially for tasks with no training data. Typically, such synthetic query generation (QGen) approaches condition on an input context (e.g. a text document) and generate a query relevant to that context, or condition the QGen additionally on the relevance label (e.g. relevant vs irrelevant) to generate queries across relevance buckets. However, we find that such QGen approaches are sub-optimal as they require the model to reason about the desired label and the input from a handful of examples. In this work, we propose to reduce this burden of LLMs by generating queries simultaneously for different labels. We hypothesize that instead of asking the model to generate, say, an irrelevant query given an input context, asking the model to generate an irrelevant query relative to a relevant query is a much simpler task. Extensive experimentation across nine IR datasets shows that synthetic queries generated in such a fashion translates to better downstream performance.