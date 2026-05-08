---
title: "BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings"
source: "https://aclanthology.org/2024.naacl-long.45/"
categories: ['bpe-subword-parsing-evaluation', 'zero-shot-few-shot-multimodal-optimization']
tags: ['sentence-embeddings', 'backward-dependency', 'llm-architecture']
venue: "NAACL 2024"
tldr: "Proposes a backward dependency enhanced LLM for improved sentence embeddings."
---

# BeLLM: Backward Dependency Enhanced Large Language Model for Sentence Embeddings

**Source**: [https://aclanthology.org/2024.naacl-long.45/](https://aclanthology.org/2024.naacl-long.45/)

**TLDR**: Proposes a backward dependency enhanced LLM for improved sentence embeddings.

## Abstract

AbstractSentence embeddings are crucial in measuring semantic similarity. Most recent studies employed large language models (LLMs) to learn sentence embeddings. Existing LLMs mainly adopted autoregressive architecture without explicit backward dependency modeling. Therefore, we examined the effects of backward dependencies in LLMs for semantic similarity measurements. Concretely, we propose a novel model: backward dependency enhanced large language model (BeLLM). It learns sentence embeddings via transforming specific attention layers from uni- to bi-directional. We extensively experiment across various semantic textual similarity (STS) tasks and downstream applications. BeLLM achieves state-of-the-art performance in varying scenarios. It shows that autoregressive LLMs benefit from backward dependencies for sentence embeddings.