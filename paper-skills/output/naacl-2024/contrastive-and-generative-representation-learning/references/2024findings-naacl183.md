---
title: "GoT: Effective Graph-of-Thought Reasoning in Language Models"
source: "https://aclanthology.org/2024.findings-naacl.183/"
categories: ['llm-knowledge-reasoning-retrieval', 'contrastive-and-generative-representation-learning', 'large-language-model-evaluation-augmentation']
tags: ['reasoning', 'graph-of-thought', 'chain-of-thought']
venue: "NAACL 2024"
tldr: "Proposes Graph-of-Thought reasoning to model non-linear human thought processes in language models for complex tasks."
---

# GoT: Effective Graph-of-Thought Reasoning in Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.183/](https://aclanthology.org/2024.findings-naacl.183/)

**TLDR**: Proposes Graph-of-Thought reasoning to model non-linear human thought processes in language models for complex tasks.

## Abstract

AbstractWith the widespread use of language models (LMs) in NLP tasks, researchers have discovered the potential of Chain-of-thought (CoT) to assist LMs in accomplishing complex reasoning tasks by generating intermediate steps. However, human thought processes are often non-linear, rather than simply sequential chains of thoughts. Therefore, we propose Graph-of-Thought (GoT) reasoning, which models human thought processes not only as a chain but also as a graph. By representing thought units as nodes and connections between them as edges, our approach captures the non-sequential nature of human thinking and allows for a more realistic modeling of thought processes. GoT adopts a two-stage framework with an additional GoT encoder for thought graph representation and fuses the graph representation with the original input representation through a gated fusion mechanism. We evaluate GoT’s performance on a text-only reasoning task (AQUA-RAT) and a multimodal reasoning task (ScienceQA). Our model achieves significant improvement over the strong CoT baseline on the AQUA-RAT test set and boosts accuracy from 85.19% to 87.59% using the T5-base model over the state-of-the-art Multimodal-CoT on the ScienceQA test set. Our code is publicly available at https://github.com/Zoeyyao27/Graph-of-Thought