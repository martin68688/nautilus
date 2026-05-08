---
title: "Shears: Unstructured Sparsity with Neural Low-rank Adapter Search"
source: "https://aclanthology.org/2024.naacl-industry.34/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['sparsity', 'neural-architecture-search', 'low-rank-adapters', 'compression']
venue: "NAACL 2024"
tldr: "Introduces Shears, a method for unstructured sparsity using neural low-rank adapter search for efficient LLM fine-tuning."
---

# Shears: Unstructured Sparsity with Neural Low-rank Adapter Search

**Source**: [https://aclanthology.org/2024.naacl-industry.34/](https://aclanthology.org/2024.naacl-industry.34/)

**TLDR**: Introduces Shears, a method for unstructured sparsity using neural low-rank adapter search for efficient LLM fine-tuning.

## Abstract

AbstractRecently, several approaches successfully demonstrated that weight-sharing Neural Architecture Search (NAS) can effectively explore a search space of elastic low-rank adapters (LoRA), allowing the parameter-efficient fine-tuning (PEFT) and compression of large language models. In this paper, we introduce a novel approach called Shears, demonstrating how the integration of cost-effective sparsity and a proposed Neural Low-rank adapter Search (NLS) algorithm can further improve the efficiency of PEFT approaches. Results demonstrate the benefits of Shears compared to other methods, reaching high sparsity levels while improving or with little drop in accuracy, utilizing a single GPU for a pair of hours.