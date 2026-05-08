---
title: "Unveiling the Magic: Investigating Attention Distillation in Retrieval-Augmented Generation"
source: "https://aclanthology.org/2024.naacl-short.65/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['retrieval-augmented-generation', 'attention-distillation', 'knowledge-update']
venue: "NAACL 2024"
tldr: "Investigates attention distillation in retrieval-augmented generation, finding it can be simplified to a token-level objective."
---

# Unveiling the Magic: Investigating Attention Distillation in Retrieval-Augmented Generation

**Source**: [https://aclanthology.org/2024.naacl-short.65/](https://aclanthology.org/2024.naacl-short.65/)

**TLDR**: Investigates attention distillation in retrieval-augmented generation, finding it can be simplified to a token-level objective.

## Abstract

AbstractRetrieval-augmented generation framework addresses the limitations of large language models by enabling real-time knowledge updates for more accurate answers. An efficient way in the training phase of retrieval-augmented models is attention distillation, which uses attention scores as supervision signals instead of manually annotated query-document pairs. Despite its growing popularity, the detailed mechanisms behind the success of attention distillation remain unexplored, particularly the specific patterns it leverages to benefit training. In this paper, we address this gap by conducting a comprehensive investigation of attention distillation workflow and identifying key factors influencing the learning performance of retrieval-augmented language models. We further propose several insightful indicators for optimizing models’ training methods and avoiding ineffective training.