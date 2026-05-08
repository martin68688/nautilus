---
title: "Structured Pruning for Large Language Models Using Coupled Components Elimination and Minor Fine-tuning"
source: "https://aclanthology.org/2024.findings-naacl.1/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['model-pruning', 'llm-efficiency', 'structured-pruning']
venue: "NAACL 2024"
tldr: "Proposes a structured pruning method for large language models using coupled component elimination and minor fine-tuning."
---

# Structured Pruning for Large Language Models Using Coupled Components Elimination and Minor Fine-tuning

**Source**: [https://aclanthology.org/2024.findings-naacl.1/](https://aclanthology.org/2024.findings-naacl.1/)

**TLDR**: Proposes a structured pruning method for large language models using coupled component elimination and minor fine-tuning.

## Abstract

AbstractLarge language models (LLMs) have demonstrated powerful capabilities in natural language processing, yet their vast number of parameters poses challenges for deployment and inference efficiency. Structured model pruning emerges as a viable approach to reduce model size and accelerate inference, without requiring specialized operators and libraries for deployment. However, structured pruning often severely weakens the model’s capability.Despite repetitive fine-tuning can restore the capability to a certain extent, it impairs LLMs’ utility as versatile problem solvers.To address this issue, we propose a novel structured pruning algorithm tailored for LLMs. It derives the importance of different components, namely rows and columns in parameter matrices, based on intermediate data dependencies. Then it removes coupled components across different layers simultaneously and preserves dependency relationships within remaining parameters, avoiding significant performance degradation. The pruned model requires only few epochs of fine-tuning to restore its performance, ensuring the model’s ability to generalize.Empirical evaluations on LLaMA, Vicuna, and ChatGLM3 demonstrate our algorithm’s efficacy, yielding 20% parameter reduction while retaining at least 94.4% of original performance metrics.