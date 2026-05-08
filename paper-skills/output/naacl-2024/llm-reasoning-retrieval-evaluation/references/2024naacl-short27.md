---
title: "Discourse-Aware In-Context Learning for Temporal Expression Normalization"
source: "https://aclanthology.org/2024.naacl-short.27/"
categories: ['llm-reasoning-retrieval-evaluation', 'llm-ranking-adaptation-benchmarking']
tags: ['temporal-reasoning', 'in-context-learning', 'normalization', 'discourse']
venue: "NAACL 2024"
tldr: "Uses discourse-aware in-context learning with proprietary LLMs for temporal expression normalization, addressing data scarcity."
---

# Discourse-Aware In-Context Learning for Temporal Expression Normalization

**Source**: [https://aclanthology.org/2024.naacl-short.27/](https://aclanthology.org/2024.naacl-short.27/)

**TLDR**: Uses discourse-aware in-context learning with proprietary LLMs for temporal expression normalization, addressing data scarcity.

## Abstract

AbstractTemporal expression (TE) normalization is a well-studied problem. However, the predominately used rule-based systems are highly restricted to specific settings, and upcoming machine learning approaches suffer from a lack of labeled data. In this work, we explore the feasibility of proprietary and open-source large language models (LLMs) for TE normalization using in-context learning to inject task, document, and example information into the model. We explore various sample selection strategies to retrieve the most relevant set of examples. By using a window-based prompt design approach, we can perform TE normalization across sentences, while leveraging the LLM knowledge without training the model.Our experiments show competitive results to models designed for this task. In particular, our method achieves large performance improvements for non-standard settings by dynamically including relevant examples during inference.