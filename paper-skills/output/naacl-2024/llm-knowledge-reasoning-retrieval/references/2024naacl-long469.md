---
title: "UNcommonsense Reasoning: Abductive Reasoning about Uncommon Situations"
source: "https://aclanthology.org/2024.naacl-long.469/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-backdoor-attacks-defense']
tags: ['reasoning', 'abductive', 'uncommon-situations']
venue: "NAACL 2024"
tldr: "Investigates abductive commonsense reasoning about unusual and uncommon situations, moving beyond everyday inferences."
---

# UNcommonsense Reasoning: Abductive Reasoning about Uncommon Situations

**Source**: [https://aclanthology.org/2024.naacl-long.469/](https://aclanthology.org/2024.naacl-long.469/)

**TLDR**: Investigates abductive commonsense reasoning about unusual and uncommon situations, moving beyond everyday inferences.

## Abstract

AbstractLanguage technologies that accurately model the dynamics of events must perform commonsense reasoning. Existing work evaluating commonsense reasoning focuses on making inferences about common, everyday situations. To instead investigate the ability to model unusual, unexpected, and unlikely situations, we explore the task of uncommonsense abductive reasoning. Given a piece of context with an unexpected outcome, this task requires reasoning abductively to generate an explanation that makes the unexpected outcome more likely in the context. To this end, we curate and release a new English language corpus called UNcommonsense. We characterize the performance differences between human explainers and the best-performing large language models, finding that model-enhanced human-written explanations achieve the highest quality by trading off between specificity and diversity. Finally, we experiment with several imitation learning algorithms to train open and accessible language models on this task. When compared with the vanilla supervised fine-tuning approach, these methods consistently reduce lose rates on both common and uncommonsense abductive reasoning judged by human evaluators.