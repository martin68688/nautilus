---
title: "RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization"
source: "https://aclanthology.org/2024.naacl-long.121/"
categories: ['knowledge-graph-and-information-extraction', 'llm-evaluation-summarization-argument-extraction']
tags: ['summarization', 'discourse', 'rst', 'lora', 'long-document']
venue: "NAACL 2024"
tldr: "Enhances long document summarization with a discourse-aware low-rank adaptation (RST-LoRA) using rhetorical structure."
---

# RST-LoRA: A Discourse-Aware Low-Rank Adaptation for Long Document Abstractive Summarization

**Source**: [https://aclanthology.org/2024.naacl-long.121/](https://aclanthology.org/2024.naacl-long.121/)

**TLDR**: Enhances long document summarization with a discourse-aware low-rank adaptation (RST-LoRA) using rhetorical structure.

## Abstract

AbstractFor long document summarization, discourse structure is important to discern the key content of the text and the differences in importance level between sentences. Unfortunately, the integration of rhetorical structure theory (RST) into parameter-efficient fine-tuning strategies for long document summarization remains unexplored. Therefore, this paper introduces RST-LoRA and proposes four RST-aware variants to explicitly incorporate RST into the LoRA model. Our empirical evaluation demonstrates that incorporating the type and uncertainty of rhetorical relations can complementarily enhance the performance of LoRA in summarization tasks. Furthermore, the best-performing variant we introduced outperforms the vanilla LoRA and full-parameter fine-tuning models, as confirmed by multiple automatic and human evaluations, and even surpasses previous state-of-the-art methods.