---
title: "Crafting In-context Examples according to LMs’ Parametric Knowledge"
source: "https://aclanthology.org/2024.findings-naacl.133/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['in-context-learning', 'knowledge-retrieval', 'prompt-engineering']
venue: "NAACL 2024"
tldr: "Studies how to construct in-context examples to better trigger a language model's parametric knowledge."
---

# Crafting In-context Examples according to LMs’ Parametric Knowledge

**Source**: [https://aclanthology.org/2024.findings-naacl.133/](https://aclanthology.org/2024.findings-naacl.133/)

**TLDR**: Studies how to construct in-context examples to better trigger a language model's parametric knowledge.

## Abstract

AbstractIn-context learning can improve the performances of knowledge-rich tasks such as question answering. In such scenarios, in-context examples trigger a language model (LM) to surface information stored in its parametric knowledge. We study how to better construct in-context example sets, based on whether the model is aware of the in-context examples. We identify ‘known’ examples, where models can correctly answer from their parametric knowledge, and ‘unknown’ ones. Our experiments show that prompting with ‘unknown’ examples decreases the performance, potentially as it encourages hallucination rather than searching for its parametric knowledge. Constructing an in-context example set that presents both known and unknown information performs the best across diverse settings. We perform analysis on three multi-answer question answering datasets, which allows us to further study answer set ordering strategies based on the LM’s knowledge of each answer. Together, our study sheds light on how to best construct in-context example sets for knowledge-rich tasks.