---
title: "Laying Anchors: Semantically Priming Numerals in Language Modeling"
source: "https://aclanthology.org/2024.findings-naacl.169/"
categories: ['llm-knowledge-reasoning-retrieval', 'transformer-language-model-probing']
tags: ['numerical-reasoning', 'language-model-pretraining']
venue: "NAACL 2024"
tldr: "Introduces semantic priming strategies for numerals during language model pretraining to improve numerical comprehension."
---

# Laying Anchors: Semantically Priming Numerals in Language Modeling

**Source**: [https://aclanthology.org/2024.findings-naacl.169/](https://aclanthology.org/2024.findings-naacl.169/)

**TLDR**: Introduces semantic priming strategies for numerals during language model pretraining to improve numerical comprehension.

## Abstract

AbstractOff-the-shelf pre-trained language models have become the de facto standard in NLP pipelines for a multitude of downstream tasks. However, the inability of these models to properly encode numerals limits their performance on tasks requiring numeric comprehension. We introduce strategies to semantically prime numerals in any corpus by generating anchors governed by the distribution of numerals in said corpus, thereby enabling mathematically grounded representations of these numeral tokens. We establish the superiority of our proposed techniques through evaluation on a range of numeracy tasks for both in-domain (seen) and out-domain (unseen) numerals. Further, we expand our empirical evaluations to numerals ranging from 1 to 10 billion, a significantly broader range compared to previous studies of the same nature, and we demonstrate significant improvements in the mathematical grounding of our learned embeddings.