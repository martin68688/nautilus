---
title: "More room for language: Investigating the effect of retrieval on language models"
source: "https://aclanthology.org/2024.naacl-short.26/"
categories: ['llm-knowledge-reasoning-retrieval']
tags: ['retrieval-augmented', 'language-modeling', 'pretraining']
venue: "NAACL 2024"
tldr: "Investigates the effect of retrieval on language models using an 'ideal retrieval' methodology during pretraining."
---

# More room for language: Investigating the effect of retrieval on language models

**Source**: [https://aclanthology.org/2024.naacl-short.26/](https://aclanthology.org/2024.naacl-short.26/)

**TLDR**: Investigates the effect of retrieval on language models using an 'ideal retrieval' methodology during pretraining.

## Abstract

AbstractRetrieval-augmented language models pose a promising alternative to standard language modeling. During pretraining, these models search in a corpus of documents for contextually relevant information that could aid the language modeling objective. We introduce an ‘ideal retrieval’ methodology to study these models in a fully controllable setting. We conduct an extensive evaluation to examine how retrieval augmentation affects the behavior of the underlying language model. Among other things, we observe that these models: (i) save substantially less world knowledge in their weights, (ii) are better at understanding local context and inter-word dependencies, but (iii) are worse at comprehending global context.