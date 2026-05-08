---
title: "Bilateral Masking with prompt for Knowledge Graph Completion"
source: "https://aclanthology.org/2024.findings-naacl.17/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-graph-and-information-extraction']
tags: ['knowledge-graph', 'completion', 'prompting']
venue: "NAACL 2024"
tldr: "Proposes a bilateral masking method with prompts to improve knowledge graph completion using pre-trained language models."
---

# Bilateral Masking with prompt for Knowledge Graph Completion

**Source**: [https://aclanthology.org/2024.findings-naacl.17/](https://aclanthology.org/2024.findings-naacl.17/)

**TLDR**: Proposes a bilateral masking method with prompts to improve knowledge graph completion using pre-trained language models.

## Abstract

AbstractThe pre-trained language model (PLM) has achieved significant success in the field of knowledge graph completion (KGC) by effectively modeling entity and relation descriptions. In recent studies, the research in this field has been categorized into methods based on word matching and sentence matching, with the former significantly lags behind. However, there is a critical issue in word matching methods, which is that these methods fail to obtain satisfactory single embedding representations for entities.To address this issue and enhance entity representation, we propose the Bilateral Masking with prompt for Knowledge Graph Completion (BMKGC) approach.Our methodology employs prompts to narrow the distance between the predicted entity and the known entity. Additionally, the BMKGC model incorporates a bi-encoder architecture, enabling simultaneous predictions at both the head and tail. Furthermore, we propose a straightforward technique to augment positive samples, mitigating the problem of degree bias present in knowledge graphs and thereby improving the model’s robustness. Experimental results conclusively demonstrate that BMKGC achieves state-of-the-art performance on the WN18RR dataset.