---
title: "TableLlama: Towards Open Large Generalist Models for Tables"
source: "https://aclanthology.org/2024.naacl-long.335/"
categories: ['llm-knowledge-reasoning-retrieval', 'llm-ranking-benchmarking-adaptation']
tags: ['tabular-data', 'llm-finetuning', 'generalist-model']
venue: "NAACL 2024"
tldr: "Proposes an open generalist model for interpreting and querying semi-structured tables without task-specific pretraining."
---

# TableLlama: Towards Open Large Generalist Models for Tables

**Source**: [https://aclanthology.org/2024.naacl-long.335/](https://aclanthology.org/2024.naacl-long.335/)

**TLDR**: Proposes an open generalist model for interpreting and querying semi-structured tables without task-specific pretraining.

## Abstract

AbstractSemi-structured tables are ubiquitous. There has been a variety of tasks that aim to automatically interpret, augment, and query tables. Current methods often require pretraining on tables or special model architecture design, are restricted to specific table types, or have simplifying assumptions about tables and tasks. This paper makes the first step towards developing open-source large language models (LLMs) as generalists for a diversity of table-based tasks. Towards that end, we construct TableInstruct, a new dataset with a variety of realistic tables and tasks, for instruction tuning and evaluating LLMs. We further develop the first open-source generalist model for tables, TableLlama, by fine-tuning Llama 2 (7B) with LongLoRA to address the long context challenge. We experiment under both in-domain setting and out-of-domain setting. On 7 out of 8 in-domain tasks, TableLlama achieves comparable or better performance than the SOTA for each task, despite the latter often has task-specific design. On 6 out-of-domain datasets, it achieves 5-44 absolute point gains compared with the base model, showing that training on TableInstruct enhances the model’s generalizability. We open-source our dataset and trained model to boost future work on developing open generalist models for tables.