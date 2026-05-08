---
title: "The Unreasonable Effectiveness of Random Target Embeddings for Continuous-Output Neural Machine Translation"
source: "https://aclanthology.org/2024.naacl-short.56/"
categories: ['efficient-transformer-optimization-techniques', 'llm-ranking-benchmarking-adaptation']
tags: ['knowledge-distillation', 'bert-compression', 'task-agnostic']
venue: "NAACL 2024"
tldr: "A weight-inherited distillation method for task-agnostic BERT compression that directly transfers teacher parameters."
---

# The Unreasonable Effectiveness of Random Target Embeddings for Continuous-Output Neural Machine Translation

**Source**: [https://aclanthology.org/2024.naacl-short.56/](https://aclanthology.org/2024.naacl-short.56/)

**TLDR**: A weight-inherited distillation method for task-agnostic BERT compression that directly transfers teacher parameters.

## Abstract

AbstractContinuous-output neural machine translation (CoNMT) replaces the discrete next-word prediction problem with an embedding prediction.The semantic structure of the target embedding space (*i.e.*, closeness of related words) is intuitively believed to be crucial. We challenge this assumption and show that completely random output embeddings can outperform laboriously pre-trained ones, especially on larger datasets. Further investigation shows this surprising effect is strongest for rare words, due to the geometry of their embeddings. We shed further light on this finding by designing a mixed strategy that combines random and pre-trained embeddings, and that performs best overall.