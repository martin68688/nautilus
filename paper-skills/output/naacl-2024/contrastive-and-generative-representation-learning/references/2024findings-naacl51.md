---
title: "DivTOD: Unleashing the Power of LLMs for Diversifying Task-Oriented Dialogue Representations"
source: "https://aclanthology.org/2024.findings-naacl.51/"
categories: ['contrastive-and-generative-representation-learning', 'llm-edge-distillation']
tags: ['task-oriented-dialogue', 'diversification', 'representation-learning']
venue: "NAACL 2024"
tldr: "Proposes a method to diversify task-oriented dialogue representations using LLMs to better capture their distinct linguistic characteristics."
---

# DivTOD: Unleashing the Power of LLMs for Diversifying Task-Oriented Dialogue Representations

**Source**: [https://aclanthology.org/2024.findings-naacl.51/](https://aclanthology.org/2024.findings-naacl.51/)

**TLDR**: Proposes a method to diversify task-oriented dialogue representations using LLMs to better capture their distinct linguistic characteristics.

## Abstract

AbstractLanguage models pre-trained on general text have achieved impressive results in diverse fields. Yet, the distinct linguistic characteristics of task-oriented dialogues (TOD) compared to general text limit the practical utility of existing language models. Current task-oriented dialogue pre-training methods overlook the one-to-many property of conversations, where multiple responses can be appropriate given the same conversation context.In this paper, we propose a novel dialogue pre-training model called DivTOD, which collaborates with LLMs to learn diverse task-oriented dialogue representations. DivTOD guides LLMs in transferring diverse knowledge to smaller models while removing domain knowledge that contradicts task-oriented dialogues. Experiments show that our model outperforms strong TOD baselines on various downstream dialogue tasks and learns the intrinsic diversity of task-oriented dialogues.