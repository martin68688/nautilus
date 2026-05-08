---
title: "Neurocache: Efficient Vector Retrieval for Long-range Language Modeling"
source: "https://aclanthology.org/2024.naacl-long.50/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques', 'fast-exact-nearest-neighbor-retrieval']
tags: ['long-context', 'retrieval', 'cache', 'knn', 'language-modeling']
venue: "NAACL 2024"
tldr: "Extending LLM context size using an external vector cache and efficient kNN retrieval of past states."
---

# Neurocache: Efficient Vector Retrieval for Long-range Language Modeling

**Source**: [https://aclanthology.org/2024.naacl-long.50/](https://aclanthology.org/2024.naacl-long.50/)

**TLDR**: Extending LLM context size using an external vector cache and efficient kNN retrieval of past states.

## Abstract

AbstractThis paper introduces Neurocache, an approach to extend the effective context size of large language models (LLMs) using an external vector cache to store its past states. Like recent vector retrieval approaches, Neurocache uses an efficient k-nearest-neighbor (kNN) algorithm to retrieve relevant past states and incorporate them into the attention process. Neurocache improves upon previous methods by (1) storing compressed states, which reduces cache size; (2) performing a single retrieval operation per token which increases inference speed; and (3) extending the retrieval window to neighboring states, which improves both language modeling and downstream task accuracy. Our experiments show the effectiveness of Neurocache both for models trained from scratch and for pre-trained models such as Llama2-7B and Mistral-7B when enhanced with the cache mechanism. We also compare Neurocache with text retrieval methods and show improvements in single-document question-answering and few-shot learning tasks. We made the source code available under: https://github.com/alisafaya/neurocache