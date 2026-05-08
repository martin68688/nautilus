---
title: "Biomedical Entity Representation with Graph-Augmented Multi-Objective Transformer"
source: "https://aclanthology.org/2024.findings-naacl.288/"
categories: ['clinical-nlp-biomedical-applications', 'contrastive-and-generative-representation-learning', 'knowledge-graph-and-information-extraction']
tags: ['biomedical-entities', 'graph-representation', 'multi-objective-learning']
venue: "NAACL 2024"
tldr: "Introduces a graph-augmented multi-objective transformer for biomedical entity representation."
---

# Biomedical Entity Representation with Graph-Augmented Multi-Objective Transformer

**Source**: [https://aclanthology.org/2024.findings-naacl.288/](https://aclanthology.org/2024.findings-naacl.288/)

**TLDR**: Introduces a graph-augmented multi-objective transformer for biomedical entity representation.

## Abstract

AbstractModern biomedical concept representations are mostly trained on synonymous concept names from a biomedical knowledge base, ignoring the inter-concept interactions and a concept’s local neighborhood in a knowledge base graph. In this paper, we introduce Biomedical Entity Representation with a Graph-Augmented Multi-Objective Transformer (BERGAMOT), which adopts the power of pre-trained language models (LMs) and graph neural networks to capture both inter-concept and intra-concept interactions from the multilingual UMLS graph. To obtain fine-grained graph representations, we introduce two additional graph-based objectives: (i) a node-level contrastive objective and (ii) the Deep Graph Infomax (DGI) loss, which maximizes the mutual information between a local subgraph and a high-level graph summary. We apply contrastive loss on textual and graph representations to make them less sensitive to surface forms and enable intermodal knowledge exchange. BERGAMOT achieves state-of-the-art results in zero-shot entity linking without task-specific supervision on 4 of 5 languages of the Mantra corpus and on 8 of 10 languages of the XL-BEL benchmark.