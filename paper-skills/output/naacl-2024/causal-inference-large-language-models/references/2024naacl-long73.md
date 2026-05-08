---
title: "AutoPRM: Automating Procedural Supervision for Multi-Step Reasoning via Controllable Question Decomposition"
source: "https://aclanthology.org/2024.naacl-long.73/"
categories: ['llm-reasoning-retrieval-evaluation', 'causal-inference-large-language-models']
tags: ['reasoning', 'procedural-supervision', 'question-decomposition', 'self-supervised']
venue: "NAACL 2024"
tldr: "A self-supervised method for automating procedural supervision in multi-step reasoning via controllable question decomposition."
---

# AutoPRM: Automating Procedural Supervision for Multi-Step Reasoning via Controllable Question Decomposition

**Source**: [https://aclanthology.org/2024.naacl-long.73/](https://aclanthology.org/2024.naacl-long.73/)

**TLDR**: A self-supervised method for automating procedural supervision in multi-step reasoning via controllable question decomposition.

## Abstract

AbstractRecent advancements in large language models (LLMs) have shown promise in multi-step reasoning tasks, yet their reliance on extensive manual labeling to provide procedural feedback remains a significant impediment. To address this challenge, in this paper, we propose a novel self-supervised framework **AutoPRM** that efficiently enhances the fine-tuning of LLMs for intricate reasoning challenges. Specifically, **AutoPRM** first decomposes complex problems into more manageable subquestions with a controllable granularity switch, then sequentially apply reinforcement learning to iteratively improve the subquestion solver. Additionally, we propose context-guided decoding to avoid reward tampering and guide the subquestion solver towards the solution of the holistic problem. Extensive experiments show that **AutoPRM** significantly improves performance on mathematical and commonsense reasoning tasks over SOTA. More encouragingly, **AutoPRM** can be easily integrated with other orthogonal reasoning pipelines.