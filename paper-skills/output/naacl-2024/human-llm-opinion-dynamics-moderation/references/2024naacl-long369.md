---
title: "The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing"
source: "https://aclanthology.org/2024.naacl-long.369/"
categories: ['knowledge-graph-and-information-extraction', 'human-llm-opinion-dynamics-moderation']
tags: ['knowledge-graph', 'entity-typing', 'semantics']
venue: "NAACL 2024"
tldr: "Improves knowledge graph entity typing by integrating both semantic knowledge from text and structural knowledge from the graph."
---

# The Integration of Semantic and Structural Knowledge in Knowledge Graph Entity Typing

**Source**: [https://aclanthology.org/2024.naacl-long.369/](https://aclanthology.org/2024.naacl-long.369/)

**TLDR**: Improves knowledge graph entity typing by integrating both semantic knowledge from text and structural knowledge from the graph.

## Abstract

AbstractThe Knowledge Graph Entity Typing (KGET) task aims to predict missing type annotations for entities in knowledge graphs. Recent works only utilize the structural knowledge in the local neighborhood of entities, disregarding semantic knowledge in the textual representations of entities, relations, and types that are also crucial for type inference. Additionally, we observe that the interaction between semantic and structural knowledge can be utilized to address the false-negative problem. In this paper, we propose a novel Semantic and Structure-aware KG Entity Typing (SSET) framework, which is composed of three modules. First, the Semantic Knowledge Encoding module encodes factual knowledge in the KG with a Masked Entity Typing task. Then, the Structural Knowledge Aggregation module aggregates knowledge from the multi-hop neighborhood of entities to infer missing types. Finally, the Unsupervised Type Re-ranking module utilizes the inference results from the two models above to generate type predictions that are robust to false-negative samples. Extensive experiments show that SSET significantly outperforms existing state-of-the-art methods.