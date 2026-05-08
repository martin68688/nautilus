---
title: "NeuroComparatives: Neuro-Symbolic Distillation of Comparative Knowledge"
source: "https://aclanthology.org/2024.findings-naacl.281/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['knowledge-distillation', 'comparative-knowledge', 'neuro-symbolic']
venue: "NAACL 2024"
tldr: "Distills comparative knowledge from large language models into a structured, neuro-symbolic knowledge base."
---

# NeuroComparatives: Neuro-Symbolic Distillation of Comparative Knowledge

**Source**: [https://aclanthology.org/2024.findings-naacl.281/](https://aclanthology.org/2024.findings-naacl.281/)

**TLDR**: Distills comparative knowledge from large language models into a structured, neuro-symbolic knowledge base.

## Abstract

AbstractComparative knowledge (e.g., steel is stronger and heavier than styrofoam) is an essential component of our world knowledge, yet understudied in prior literature. In this paper, we harvest the dramatic improvements in knowledge capabilities of language models into a large-scale comparative knowledge base. While the ease of acquisition of such comparative knowledge is much higher from extreme-scale models like GPT-4, compared to their considerably smaller and weaker counterparts such as GPT-2, not even the most powerful models are exempt from making errors. We thus ask: to what extent are models at different scales able to generate valid and diverse comparative knowledge?We introduce NeuroComparatives, a novel framework for comparative knowledge distillation overgenerated from language models such as GPT-variants and LLaMA, followed by stringent filtering of the generated knowledge. Our framework acquires comparative knowledge between everyday objects, producing a corpus of up to 8.8M comparisons over 1.74M entity pairs - 10X larger and 30% more diverse than existing resources. Moreover, human evaluations show that NeuroComparatives outperform existing resources in terms of validity (up to 32% absolute improvement). Our acquired NeuroComparatives leads to performance improvements on five downstream tasks.We find that neuro-symbolic manipulation of smaller models offers complementary benefits to the currently dominant practice of prompting extreme-scale language models for knowledge distillation.