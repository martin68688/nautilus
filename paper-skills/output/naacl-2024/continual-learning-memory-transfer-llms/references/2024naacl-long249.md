---
title: "Memory Augmented Language Models through Mixture of Word Experts"
source: "https://aclanthology.org/2024.naacl-long.249/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques', 'continual-learning-memory-transfer-llms']
tags: ['mixture-of-experts', 'memory', 'sparse-models']
venue: "NAACL 2024"
tldr: "Augments language models with memory through a mixture of word experts to decouple learning capacity from computational FLOPs."
---

# Memory Augmented Language Models through Mixture of Word Experts

**Source**: [https://aclanthology.org/2024.naacl-long.249/](https://aclanthology.org/2024.naacl-long.249/)

**TLDR**: Augments language models with memory through a mixture of word experts to decouple learning capacity from computational FLOPs.

## Abstract

AbstractScaling up the number of parameters of language models has proven to be an effective approach to improve performance. For dense models, increasing their size proportionally increases their computational footprint. In this work, we seek to aggressively decouple learning capacity and FLOPs through Mixture-of-Experts (MoE) style models with large knowledge-rich vocabulary based routing functions. Our proposed approach, dubbed Mixture of Word Experts (MoWE), can be seen as a memory augmented model, where a large set of word-specific experts play the role of a sparse memory. We demonstrate that MoWE performs significantly better than the T5 family of models with similar number of FLOPs in a variety of NLP tasks. Moreover, MoWE outperforms traditional MoE models on knowledge intensive tasks and has similar performance to complex memory augmented approaches that often require to invoke custom mechanisms to search the sparse memory.