---
title: "Order-Based Pre-training Strategies for Procedural Text Understanding"
source: "https://aclanthology.org/2024.naacl-short.74/"
categories: ['knowledge-graph-and-information-extraction', 'topic-modeling-and-keyphrase-generation']
tags: ['procedural-text', 'pre-training', 'entity-tracking']
venue: "NAACL 2024"
tldr: "Proposes sequence-based pre-training strategies to improve models' understanding of procedural texts with changing entity states."
---

# Order-Based Pre-training Strategies for Procedural Text Understanding

**Source**: [https://aclanthology.org/2024.naacl-short.74/](https://aclanthology.org/2024.naacl-short.74/)

**TLDR**: Proposes sequence-based pre-training strategies to improve models' understanding of procedural texts with changing entity states.

## Abstract

AbstractIn this paper, we propose sequence-based pre-training methods to enhance procedural understanding in natural language processing. Procedural text, containing sequential instructions to accomplish a task, is difficult to understand due to the changing attributes of entities in the context. We focus on recipes as they are commonly represented as ordered instructions, and use this order as a supervision signal. Our work is one of the first to compare several ‘order-as-supervision’ transformer pre-training methods, including Permutation Classification, Embedding Regression, and Skip-Clip, and show that these methods give improved results compared to baselines and SoTA LLMs on two downstream Entity-Tracking datasets: NPN-Cooking dataset in recipe domain and ProPara dataset in open domain. Our proposed methods address the non-trivial Entity Tracking Task that requires prediction of entity states across procedure steps, which requires understanding the order of steps. These methods show an improvement over the best baseline by 1.6% and 7-9% on NPN-Cooking and ProPara Datasets respectively across metrics.