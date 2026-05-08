---
title: "RE2: Region-Aware Relation Extraction from Visually Rich Documents"
source: "https://aclanthology.org/2024.naacl-long.484/"
categories: ['knowledge-graph-and-information-extraction', 'llm-alignment-safety-detoxification']
tags: ['relation-extraction', 'visually-rich-documents', 'layout-structure', 'region-aware']
venue: "NAACL 2024"
tldr: "Proposes a region-aware relation extraction method for visually rich documents that leverages layout structure without requiring large pre-training data."
---

# RE2: Region-Aware Relation Extraction from Visually Rich Documents

**Source**: [https://aclanthology.org/2024.naacl-long.484/](https://aclanthology.org/2024.naacl-long.484/)

**TLDR**: Proposes a region-aware relation extraction method for visually rich documents that leverages layout structure without requiring large pre-training data.

## Abstract

AbstractCurrent research in form understanding predominantly relies on large pre-trained language models, necessitating extensive data for pre-training. However, the importance of layout structure (i.e., the spatial relationship between the entity blocks in the visually rich document) to relation extraction has been overlooked. In this paper, we propose REgion-Aware Relation Extraction (RE2) that leverages region-level spatial structure among the entity blocks to improve their relation prediction. We design an edge-aware graph attention network to learn the interaction between entities while considering their spatial relationship defined by their region-level representations. We also introduce a constraint objective to regularize the model towards consistency with the inherent constraints of the relation extraction task. To support the research on relation extraction from visually rich documents and demonstrate the generalizability of RE2, we build a new benchmark dataset, DiverseForm, that covers a wide range of domains. Extensive experiments on DiverseForm and several public benchmark datasets demonstrate significant superiority and transferability of RE2 across various domains and languages, with up to 18.88% absolute F-score gain over all high-performing baselines