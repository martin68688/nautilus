---
title: "GenTKG: Generative Forecasting on Temporal Knowledge Graph with Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.268/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['temporal-knowledge-graph', 'forecasting', 'llm', 'generative']
venue: "NAACL 2024"
tldr: "Proposes a generative forecasting method for temporal knowledge graphs using LLMs."
---

# GenTKG: Generative Forecasting on Temporal Knowledge Graph with Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.268/](https://aclanthology.org/2024.findings-naacl.268/)

**TLDR**: Proposes a generative forecasting method for temporal knowledge graphs using LLMs.

## Abstract

AbstractThe rapid advancements in large language models (LLMs) have ignited interest in the temporal knowledge graph (tKG) domain, where conventional embedding-based and rule-based methods dominate. The question remains open of whether pre-trained LLMs can understand structured temporal relational data and replace them as the foundation model for temporal relational forecasting. Therefore, we bring temporal knowledge forecasting into the generative setting. However, challenges occur in the huge chasms between complex temporal graph data structure and sequential natural expressions LLMs can handle, and between the enormous data sizes of tKGs and heavy computation costs of finetuning LLMs. To address these challenges, we propose a novel retrieval-augmented generation framework named GenTKG combining a temporal logical rule-based retrieval strategy and few-shot parameter-efficient instruction tuning to solve the above challenges, respectively. Extensive experiments have shown that GenTKG outperforms conventional methods of temporal relational forecasting with low computation resources using extremely limited training data as few as 16 samples. GenTKG also highlights remarkable cross-domain generalizability with outperforming performance on unseen datasets without re-training, and in-domain generalizability regardless of time split in the same dataset. Our work reveals the huge potential of LLMs in the tKG domain and opens a new frontier for generative forecasting on tKGs. The code and data are released here: https://github.com/mayhugotong/GenTKG.