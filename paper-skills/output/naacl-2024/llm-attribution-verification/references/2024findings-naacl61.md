---
title: "Planning and Editing What You Retrieve for Enhanced Tool Learning"
source: "https://aclanthology.org/2024.findings-naacl.61/"
categories: ['llm-knowledge-reasoning-retrieval', 'contrastive-and-generative-representation-learning', 'llm-attribution-verification']
tags: ['tool-learning', 'retrieval', 'planning']
venue: "NAACL 2024"
tldr: "Enhances tool-augmented LLMs by planning and editing retrieved information before use, improving performance on complex tasks."
---

# Planning and Editing What You Retrieve for Enhanced Tool Learning

**Source**: [https://aclanthology.org/2024.findings-naacl.61/](https://aclanthology.org/2024.findings-naacl.61/)

**TLDR**: Enhances tool-augmented LLMs by planning and editing retrieved information before use, improving performance on complex tasks.

## Abstract

AbstractRecent advancements in integrating external tools with Large Language Models (LLMs) have opened new frontiers, with applications in mathematical reasoning, code generators, and smart assistants. However, existing methods, relying on simple one-time retrieval strategies, fall short on effectively and accurately shortlisting relevant tools. This paper introduces a novel PLUTO (Planning, Learning, and Understanding for TOols) approach, encompassing “Plan-and-Retrieve (P&R)” and “Edit-and-Ground (E&G)” paradigms. The P&R paradigm consists of a neural retrieval module for shortlisting relevant tools and an LLM-based query planner that decomposes complex queries into actionable tasks, enhancing the effectiveness of tool utilization. The E&G paradigm utilizes LLMs to enrich tool descriptions based on user scenarios, bridging the gap between user queries and tool functionalities. Experiment results demonstrate that these paradigms significantly improve the recall and NDCG in tool retrieval tasks, significantly surpassing current state-of-the-art models.