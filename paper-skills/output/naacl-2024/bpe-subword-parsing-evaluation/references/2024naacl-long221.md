---
title: "Take One Step at a Time to Know Incremental Utility of Demonstration: An Analysis on Reranking for Few-Shot In-Context Learning"
source: "https://aclanthology.org/2024.naacl-long.221/"
categories: ['bpe-subword-parsing-evaluation', 'zero-shot-few-shot-multimodal-optimization']
tags: ['in-context-learning', 'demonstration-selection', 'reranking']
venue: "NAACL 2024"
tldr: "Analyzes the incremental utility of demonstrations and proposes a reranking method for few-shot in-context learning."
---

# Take One Step at a Time to Know Incremental Utility of Demonstration: An Analysis on Reranking for Few-Shot In-Context Learning

**Source**: [https://aclanthology.org/2024.naacl-long.221/](https://aclanthology.org/2024.naacl-long.221/)

**TLDR**: Analyzes the incremental utility of demonstrations and proposes a reranking method for few-shot in-context learning.

## Abstract

AbstractIn-Context Learning (ICL) is an emergent capability of Large Language Models (LLMs). Only a few demonstrations enable LLMs to be used as blackbox for new tasks. Previous studies have shown that using LLMs’ outputs as labels is effective in training models to select demonstrations. Such a label is expected to estimate utility of a demonstration in ICL; however, it has not been well understood how different labeling strategies affect results on target tasks. This paper presents an analysis on different utility functions by focusing on LLMs’ output probability given ground-truth output, and task-specific reward given LLMs’ prediction. Unlike the previous work, we introduce a novel labeling method, incremental utility, which estimates how much incremental knowledge is brought into the LLMs by a demonstration. We conduct experiments with instruction-tuned LLMs on binary/multi-class classification, segmentation, and translation across Arabic, English, Finnish, Japanese, and Spanish. Our results show that (1) the probability is effective when the probability values are distributed across the whole value range (on the classification tasks), and (2) the downstream metric is more robust when nuanced reward values are provided with long outputs (on the segmentation and translation tasks). We then show that the proposed incremental utility further helps ICL by contrasting how the LLMs perform with and without the demonstrations.