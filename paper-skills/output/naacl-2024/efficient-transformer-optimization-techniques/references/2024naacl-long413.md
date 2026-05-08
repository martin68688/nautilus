---
title: "DUQGen: Effective Unsupervised Domain Adaptation of Neural Rankers by Diversifying Synthetic Query Generation"
source: "https://aclanthology.org/2024.naacl-long.413/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['domain-adaptation', 'query-generation', 'ranking']
venue: "NAACL 2024"
tldr: "Proposes an unsupervised domain adaptation method for neural rankers that diversifies synthetic query generation to improve zero-shot ranking performance."
---

# DUQGen: Effective Unsupervised Domain Adaptation of Neural Rankers by Diversifying Synthetic Query Generation

**Source**: [https://aclanthology.org/2024.naacl-long.413/](https://aclanthology.org/2024.naacl-long.413/)

**TLDR**: Proposes an unsupervised domain adaptation method for neural rankers that diversifies synthetic query generation to improve zero-shot ranking performance.

## Abstract

AbstractState-of-the-art neural rankers pre-trained on large task-specific training data such as MS-MARCO, have been shown to exhibit strong performance on various ranking tasks without domain adaptation, also called zero-shot. However, zero-shot neural ranking may be sub-optimal, as it does not take advantage of the target domain information. Unfortunately, acquiring sufficiently large and high quality target training data to improve a modern neural ranker can be costly and time-consuming. To address this problem, we propose a new approach to unsupervised domain adaptation for ranking, DUQGen, which addresses a critical gap in prior literature, namely how to automatically generate both effective and diverse synthetic training data to fine tune a modern neural ranker for a new domain. Specifically, DUQGen produces a more effective representation of the target domain by identifying clusters of similar documents; and generates a more diverse training dataset by probabilistic sampling over the resulting document clusters. Our extensive experiments, over the standard BEIR collection, demonstrate that DUQGen consistently outperforms all zero-shot baselines and substantially outperforms the SOTA baselines on 16 out of 18 datasets, for an average of 4% relative improvement across all datasets. We complement our results with a thorough analysis for more in-depth understanding of the proposed method’s performance and to identify promising areas for further improvements.