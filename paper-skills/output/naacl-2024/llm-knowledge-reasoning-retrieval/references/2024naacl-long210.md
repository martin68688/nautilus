---
title: "TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction"
source: "https://aclanthology.org/2024.naacl-long.210/"
categories: ['llm-knowledge-reasoning-retrieval', 'code-search-clone-detection']
tags: ['retrieval-augmented-generation', 'hallucination', 'conformal-prediction', 'trustworthy-ai']
venue: "NAACL 2024"
tldr: "Proposes a trustworthy RAG framework using conformal prediction to provide statistical guarantees on answer correctness."
---

# TRAQ: Trustworthy Retrieval Augmented Question Answering via Conformal Prediction

**Source**: [https://aclanthology.org/2024.naacl-long.210/](https://aclanthology.org/2024.naacl-long.210/)

**TLDR**: Proposes a trustworthy RAG framework using conformal prediction to provide statistical guarantees on answer correctness.

## Abstract

AbstractWhen applied to open-domain question answering, large language models (LLMs) frequently generate incorrect responses based on made-up facts, which are called hallucinations. Retrieval augmented generation (RAG) is a promising strategy to avoid hallucinations, but it does not provide guarantees on its correctness. To address this challenge, we propose the Trustworthy Retrieval Augmented Question Answering, or *TRAQ*, which provides the first end-to-end statistical correctness guarantee for RAG. TRAQ uses conformal prediction, a statistical technique for constructing prediction sets that are guaranteed to contain the semantically correct response with high probability. Additionally, TRAQ leverages Bayesian optimization to minimize the size of the constructed sets. In an extensive experimental evaluation, we demonstrate that TRAQ provides the desired correctness guarantee while reducing prediction set size by 16.2% on average compared to an ablation. The implementation is available: [https://github.com/shuoli90/TRAQ](https://github.com/shuoli90/TRAQ).