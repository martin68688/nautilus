---
title: "Synthetic Query Generation for Privacy-Preserving Deep Retrieval Systems using Differentially Private Language Models"
source: "https://aclanthology.org/2024.naacl-long.217/"
categories: ['llm-knowledge-reasoning-retrieval', 'differentially-private-query-generation']
tags: ['differential-privacy', 'query-generation', 'retrieval']
venue: "NAACL 2024"
tldr: "Synthetic query generation using differentially private language models enables privacy-preserving training of deep retrieval systems."
---

# Synthetic Query Generation for Privacy-Preserving Deep Retrieval Systems using Differentially Private Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.217/](https://aclanthology.org/2024.naacl-long.217/)

**TLDR**: Synthetic query generation using differentially private language models enables privacy-preserving training of deep retrieval systems.

## Abstract

AbstractWe address the challenge of ensuring differential privacy (DP) guarantees in training deep retrieval systems. Training these systems often involves the use of contrastive-style losses, which are typically non-per-example decomposable, making them difficult to directly DP-train with since common techniques require per-example gradients. To address this issue, we propose an approach that prioritizes ensuring query privacy prior to training a deep retrieval system. Our method employs DP language models (LMs) to generate private synthetic queries representative of the original data. These synthetic queries can be used in downstream retrieval system training without compromising privacy. Our approach demonstrates a significant enhancement in retrieval quality compared to direct DP-training, all while maintaining query-level privacy guarantees. This work highlights the potential of harnessing LMs to overcome limitations in standard DP-training methods.