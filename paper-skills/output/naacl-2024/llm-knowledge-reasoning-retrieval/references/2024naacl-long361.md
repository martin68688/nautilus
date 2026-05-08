---
title: "CoE-SQL: In-Context Learning for Multi-Turn Text-to-SQL with Chain-of-Editions"
source: "https://aclanthology.org/2024.naacl-long.361/"
categories: ['llm-knowledge-reasoning-retrieval']
tags: ['in-context-learning', 'text-to-sql', 'chain-of-thought']
venue: "NAACL 2024"
tldr: "Enhances LLMs for multi-turn text-to-SQL using in-context learning with a Chain-of-Editions prompting strategy."
---

# CoE-SQL: In-Context Learning for Multi-Turn Text-to-SQL with Chain-of-Editions

**Source**: [https://aclanthology.org/2024.naacl-long.361/](https://aclanthology.org/2024.naacl-long.361/)

**TLDR**: Enhances LLMs for multi-turn text-to-SQL using in-context learning with a Chain-of-Editions prompting strategy.

## Abstract

AbstractRecently, Large Language Models (LLMs) have been demonstrated to possess impressive capabilities in a variety of domains and tasks. We investigate the issue of prompt design in the multi-turn text-to-SQL task and attempt to enhance the LLMs’ reasoning capacity when generating SQL queries. In the conversational context, the current SQL query can be modified from the preceding SQL query with only a few operations due to the context dependency. We introduce our method called CoE-SQL which can prompt LLMs to generate the SQL query based on the previously generated SQL query with an edition chain. We also conduct extensive ablation studies to determine the optimal configuration of our approach. Our approach outperforms different in-context learning baselines stably and achieves state-of-the-art performances on two benchmarks SParC and CoSQL using LLMs, which is also competitive to the SOTA fine-tuned models.