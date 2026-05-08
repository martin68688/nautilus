---
title: "UEGP: Unified Expert-Guided Pre-training for Knowledge Rekindle"
source: "https://aclanthology.org/2024.findings-naacl.170/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['pre-training', 'expert-guided', 'parameter-efficient']
venue: "NAACL 2024"
tldr: "Proposes a unified expert-guided pre-training framework to improve knowledge retention and avoid local minima in fine-tuning."
---

# UEGP: Unified Expert-Guided Pre-training for Knowledge Rekindle

**Source**: [https://aclanthology.org/2024.findings-naacl.170/](https://aclanthology.org/2024.findings-naacl.170/)

**TLDR**: Proposes a unified expert-guided pre-training framework to improve knowledge retention and avoid local minima in fine-tuning.

## Abstract

AbstractPre-training and fine-tuning framework has become the standard training paradigm for NLP tasks and is also widely used in industrial-level applications. However, there are still a limitation with this paradigm: simply fine-tuning with task-specific objectives tends to converge to local minima, resulting in a sub-optimal performance. In this paper, we first propose a new paradigm: knowledge rekindle, which aims to re-incorporate the fine-tuned expert model into the training cycle and break through the performance upper bounds of experts without introducing additional annotated data. Then we further propose a unified expert-guided pre-training (UEGP) framework for knowledge rekindle. Specifically, we reuse fine-tuned expert models for various downstream tasks as knowledge sources and inject task-specific prior knowledge to pre-trained language models (PLMs) by means of knowledge distillation. In this process, we perform multi-task learning with knowledge distillation and masked language modeling (MLM) objectives. We also further explored whether mixture-of-expert guided pre-training (MoEGP) can further enhance the effect of knowledge rekindle. Experiments and analysis on eight datasets in GLUE benchmark and a industrial-level search re-ranking dataset show the effectiveness of our method.