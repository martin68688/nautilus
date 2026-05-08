---
title: "FedLFC: Towards Efficient Federated Multilingual Modeling with LoRA-based Language Family Clustering"
source: "https://aclanthology.org/2024.findings-naacl.98/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'continual-learning-memory-transfer-llms']
tags: ['federated-learning', 'multilingual', 'lora']
venue: "NAACL 2024"
tldr: "Proposes a federated multilingual modeling method using LoRA clustered by language family to reduce communication costs."
---

# FedLFC: Towards Efficient Federated Multilingual Modeling with LoRA-based Language Family Clustering

**Source**: [https://aclanthology.org/2024.findings-naacl.98/](https://aclanthology.org/2024.findings-naacl.98/)

**TLDR**: Proposes a federated multilingual modeling method using LoRA clustered by language family to reduce communication costs.

## Abstract

AbstractFederated Multilingual Modeling (FMM) plays a crucial role in the applications of natural language processing due to the increasing diversity of languages and the growing demand for data privacy. However, FMM faces limitations stemming from (1) the substantial communication costs in networking and (2) the conflicts arising from parameter interference between different languages. To address these challenges, we introduce a communication-efficient federated learning framework with low-rank adaptation and language family clustering for Multilingual Modeling (MM). In this framework, we maintain the weights of the base model, exclusively updating the lightweight Low-rank adaptation (LoRA) parameters to minimize communication costs. Additionally, we mitigate parameter conflicts by grouping languages based on their language family affiliations, as opposed to aggregating all LoRA parameters. Experiments demonstrate that our proposed model not only surpasses the baseline models in performance but also reduces the communication overhead. Our code is available at https://github.com/zhihan-guo/FedLFC.