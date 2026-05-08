---
title: "SpeedE: Euclidean Geometric Knowledge Graph Embedding Strikes Back"
source: "https://aclanthology.org/2024.findings-naacl.6/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['knowledge-graph', 'embedding', 'geometric', 'link-prediction']
venue: "NAACL 2024"
tldr: "A new Euclidean geometric knowledge graph embedding model achieves strong link prediction performance with lower dimensionality and complexity."
---

# SpeedE: Euclidean Geometric Knowledge Graph Embedding Strikes Back

**Source**: [https://aclanthology.org/2024.findings-naacl.6/](https://aclanthology.org/2024.findings-naacl.6/)

**TLDR**: A new Euclidean geometric knowledge graph embedding model achieves strong link prediction performance with lower dimensionality and complexity.

## Abstract

AbstractGeometric knowledge graph embedding models (gKGEs) have shown great potential for knowledge graph completion (KGC), i.e., automatically predicting missing triples. However, contemporary gKGEs require high embedding dimensionalities or complex embedding spaces for good KGC performance, drastically limiting their space and time efficiency. Facing these challenges, we propose SpeedE, a lightweight Euclidean gKGE that (1) provides strong inference capabilities, (2) is competitive with state-of-the-art gKGEs, even significantly outperforming them on YAGO3-10 and WN18RR, and (3) dramatically increases their efficiency, in particular, needing solely a fifth of the training time and a fourth of the parameters of the state-of-the-art ExpressivE model on WN18RR to reach the same KGC performance.