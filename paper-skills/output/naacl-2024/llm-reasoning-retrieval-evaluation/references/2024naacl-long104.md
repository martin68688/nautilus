---
title: "zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.104/"
categories: ['llm-reasoning-retrieval-evaluation', 'temporal-reasoning-risk-prediction']
tags: ['temporal-knowledge-graphs', 'zero-shot', 'relational-learning']
venue: "NAACL 2024"
tldr: "Introduces a zero-shot relational learning method for forecasting links on temporal knowledge graphs using large language models."
---

# zrLLM: Zero-Shot Relational Learning on Temporal Knowledge Graphs with Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.104/](https://aclanthology.org/2024.naacl-long.104/)

**TLDR**: Introduces a zero-shot relational learning method for forecasting links on temporal knowledge graphs using large language models.

## Abstract

AbstractModeling evolving knowledge over temporal knowledge graphs (TKGs) has become a heated topic. Various methods have been proposed to forecast links on TKGs. Most of them are embedding-based, where hidden representations are learned to represent knowledge graph (KG) entities and relations based on the observed graph contexts. Although these methods show strong performance on traditional TKG forecasting (TKGF) benchmarks, they face a strong challenge in modeling the unseen zero-shot relations that have no prior graph context. In this paper, we try to mitigate this problem as follows. We first input the text descriptions of KG relations into large language models (LLMs) for generating relation representations, and then introduce them into embedding-based TKGF methods. LLM-empowered representations can capture the semantic information in the relation descriptions. This makes the relations, whether seen or unseen, with similar semantic meanings stay close in the embedding space, enabling TKGF models to recognize zero-shot relations even without any observed graph context. Experimental results show that our approach helps TKGF models to achieve much better performance in forecasting the facts with previously unseen relations, while still maintaining their ability in link forecasting regarding seen relations.