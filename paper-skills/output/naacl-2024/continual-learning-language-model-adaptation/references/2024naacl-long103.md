---
title: "TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks"
source: "https://aclanthology.org/2024.naacl-long.103/"
categories: ['information-extraction-knowledge-graph-methods', 'continual-learning-language-model-adaptation']
tags: ['information-extraction', 'universal-schema', 'relation-extraction', 'unified-framework']
venue: "NAACL 2024"
tldr: "TRUE-UIE is a universal information extraction framework that unifies diverse IE tasks by modeling them with two"
---

# TRUE-UIE: Two Universal Relations Unify Information Extraction Tasks

**Source**: [https://aclanthology.org/2024.naacl-long.103/](https://aclanthology.org/2024.naacl-long.103/)

**TLDR**: TRUE-UIE is a universal information extraction framework that unifies diverse IE tasks by modeling them with two

## Abstract

AbstractInformation extraction (IE) encounters challenges due to the variety of schemas and objectives that differ across tasks. Recent advancements hint at the potential for universal approaches to model such tasks, referred to as Universal Information Extraction (UIE). While handling diverse tasks in one model, their generalization is limited since they are actually learning task-specific knowledge.In this study, we introduce an innovative paradigm known as TRUE-UIE, wherein all IE tasks are aligned to learn the same goals: extracting mention spans and two universal relations named NEXT and IS. During the decoding process, the NEXT relation is utilized to group related elements, while the IS relation, in conjunction with structured language prompts, undertakes the role of type recognition. Additionally, we consider the sequential dependency of tokens during span extraction, an aspect often overlooked in prevalent models.Our empirical experiments indicate that TRUE-UIE achieves state-of-the-art performance on established benchmarks encompassing 16 datasets, spanning 7 diverse IE tasks. Further evaluations reveal that our approach effectively share knowledge between different IE tasks, showcasing significant transferability in zero-shot and few-shot scenarios.