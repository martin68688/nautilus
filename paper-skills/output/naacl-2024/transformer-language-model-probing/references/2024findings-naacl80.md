---
title: "Context Does Matter: Implications for Crowdsourced Evaluation Labels in Task-Oriented Dialogue Systems"
source: "https://aclanthology.org/2024.findings-naacl.80/"
categories: ['human-llm-opinion-dynamics-moderation', 'transformer-language-model-probing']
tags: ['evaluation', 'crowdsourcing', 'dialogue-systems']
venue: "NAACL 2024"
tldr: "Analyzes the importance of context for obtaining high-quality crowdsourced evaluation labels in task-oriented dialogue systems."
---

# Context Does Matter: Implications for Crowdsourced Evaluation Labels in Task-Oriented Dialogue Systems

**Source**: [https://aclanthology.org/2024.findings-naacl.80/](https://aclanthology.org/2024.findings-naacl.80/)

**TLDR**: Analyzes the importance of context for obtaining high-quality crowdsourced evaluation labels in task-oriented dialogue systems.

## Abstract

AbstractCrowdsourced labels play a crucial role in evaluating task-oriented dialogue systems (TDSs). Obtaining high-quality and consistent ground-truth labels from annotators presents challenges. When evaluating a TDS, annotators must fully comprehend the dialogue before providing judgments. Previous studies suggest using only a portion of the dialogue context in the annotation process. However, the impact of this limitation on label quality remains unexplored. This study investigates the influence of dialogue context on annotation quality, considering the truncated context for relevance and usefulness labeling. We further propose to use large language models ( LLMs) to summarize the dialogue context to provide a rich and short description of the dialogue context and study the impact of doing so on the annotator’s performance. Reducing context leads to more positive ratings. Conversely, providing the entire dialogue context yields higher-quality relevance ratings but introduces ambiguity in usefulness ratings. Using the first user utterance as context leads to consistent ratings, akin to those obtained using the entire dialogue, with significantly reduced annotation effort. Our findings show how task design, particularly the availability of dialogue context, affects the quality and consistency of crowdsourced evaluation labels.