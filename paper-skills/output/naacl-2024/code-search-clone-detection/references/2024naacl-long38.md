---
title: "Language Agnostic Code Embeddings"
source: "https://aclanthology.org/2024.naacl-long.38/"
categories: ['llm-ranking-benchmarking-adaptation', 'code-search-clone-detection']
tags: ['code-embeddings', 'multilingual', 'representation']
venue: "NAACL 2024"
tldr: "Analyzes and improves language-agnostic code embeddings from multilingual code models."
---

# Language Agnostic Code Embeddings

**Source**: [https://aclanthology.org/2024.naacl-long.38/](https://aclanthology.org/2024.naacl-long.38/)

**TLDR**: Analyzes and improves language-agnostic code embeddings from multilingual code models.

## Abstract

AbstractRecently, code language models have achieved notable advancements in addressing a diverse array of essential code comprehension and generation tasks. Yet, the field lacks a comprehensive deep dive and understanding of the code embeddings of multilingual code models. In this paper, we present a comprehensive study on multilingual code embeddings, focusing on the cross-lingual capabilities of these embeddings across different programming languages. Through probing experiments, we demonstrate that code embeddings comprise two distinct components: one deeply tied to the nuances and syntax of a specific language, and the other remaining agnostic to these details, primarily focusing on semantics. Further, we show that when we isolate and eliminate this language-specific component, we witness significant improvements in downstream code retrieval tasks, leading to an absolute increase of up to +17 in the Mean Reciprocal Rank (MRR).