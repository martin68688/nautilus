---
title: "Prompt Tuned Embedding Classification for Industry Sector Allocation"
source: "https://aclanthology.org/2024.naacl-industry.10/"
categories: ['efficient-transformer-optimization-techniques', 'llm-ranking-benchmarking-adaptation']
tags: ['prompt-tuning', 'industry-classification', 'embedding']
venue: "NAACL 2024"
tldr: "Introduces a prompt-tuned embedding classification method for allocating companies to proprietary industry sectors."
---

# Prompt Tuned Embedding Classification for Industry Sector Allocation

**Source**: [https://aclanthology.org/2024.naacl-industry.10/](https://aclanthology.org/2024.naacl-industry.10/)

**TLDR**: Introduces a prompt-tuned embedding classification method for allocating companies to proprietary industry sectors.

## Abstract

AbstractWe introduce Prompt Tuned Embedding Classification (PTEC) for classifying companies within an investment firm’s proprietary industry taxonomy, supporting their thematic investment strategy. PTEC assigns companies to the sectors they primarily operate in, conceptualizing this process as a multi-label text classification task. Prompt Tuning, usually deployed as a text-to-text (T2T) classification approach, ensures low computational cost while maintaining high task performance. However, T2T classification has limitations on multi-label tasks due to the generation of non-existing labels, permutation invariance of the label sequence, and a lack of confidence scores. PTEC addresses these limitations by utilizing a classification head in place of the Large Language Models (LLMs) language head. PTEC surpasses both baselines and human performance while lowering computational demands. This indicates the continuing need to adapt state-of-the-art methods to domain-specific tasks, even in the era of LLMs with strong generalization abilities.