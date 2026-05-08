---
title: "Evidence-Driven Retrieval Augmented Response Generation for Online Misinformation"
source: "https://aclanthology.org/2024.naacl-long.313/"
categories: ['llm-knowledge-reasoning-retrieval', 'continual-learning-memory-transfer-llms']
tags: ['misinformation', 'retrieval-augmented-generation', 'fact-checking']
venue: "NAACL 2024"
tldr: "Proposes an evidence-driven retrieval-augmented generation method to produce polite and factual responses to online misinformation."
---

# Evidence-Driven Retrieval Augmented Response Generation for Online Misinformation

**Source**: [https://aclanthology.org/2024.naacl-long.313/](https://aclanthology.org/2024.naacl-long.313/)

**TLDR**: Proposes an evidence-driven retrieval-augmented generation method to produce polite and factual responses to online misinformation.

## Abstract

AbstractThe proliferation of online misinformation has posed significant threats to public interest. While numerous online users actively participate in the combat against misinformation, many of such responses can be characterized by the lack of politeness and supporting facts. As a solution, text generation approaches are proposed to automatically produce counter-misinformation responses. Nevertheless, existing methods are often trained end-to-end without leveraging external knowledge, resulting in subpar text quality and excessively repetitive responses. In this paper, we propose retrieval augmented response generation for online misinformation (RARG), which collects supporting evidence from scientific sources and generates counter-misinformation responses based on the evidences. In particular, our RARG consists of two stages: (1) evidence collection, where we design a retrieval pipeline to retrieve and rerank evidence documents using a database comprising over 1M academic articles; (2) response generation, in which we align large language models (LLMs) to generate evidence-based responses via reinforcement learning from human feedback (RLHF). We propose a reward function to maximize the utilization of the retrieved evidence while maintaining the quality of the generated text, which yields polite and factual responses that clearly refutes misinformation. To demonstrate the effectiveness of our method, we study the case of COVID-19 and perform extensive experiments with both in- and cross-domain datasets, where RARG consistently outperforms baselines by generating high-quality counter-misinformation responses.