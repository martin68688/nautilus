---
title: "GraSAME: Injecting Token-Level Structural Information to Pretrained Language Models via Graph-guided Self-Attention Mechanism"
source: "https://aclanthology.org/2024.findings-naacl.58/"
categories: ['llm-ranking-benchmarking-adaptation', 'efficient-transformer-optimization-techniques']
tags: ['graph-infusion', 'self-attention']
venue: "NAACL 2024"
tldr: "GraSAME injects token-level structural info into PLMs via a graph-guided self-attention mechanism."
---

# GraSAME: Injecting Token-Level Structural Information to Pretrained Language Models via Graph-guided Self-Attention Mechanism

**Source**: [https://aclanthology.org/2024.findings-naacl.58/](https://aclanthology.org/2024.findings-naacl.58/)

**TLDR**: GraSAME injects token-level structural info into PLMs via a graph-guided self-attention mechanism.

## Abstract

AbstractPretrained Language Models (PLMs) benefit from external knowledge stored in graph structures for various downstream tasks. However, bridging the modality gap between graph structures and text remains a significant challenge. Traditional methods like linearizing graphs for PLMs lose vital graph connectivity, whereas Graph Neural Networks (GNNs) require cumbersome processes for integration into PLMs. In this work, we propose a novel graph-guided self-attention mechanism, GraSAME. GraSAME seamlessly incorporates token-level structural information into PLMs without necessitating additional alignment or concatenation efforts. As an end-to-end, lightweight multimodal module, GraSAME follows a multi-task learning strategy and effectively bridges the gap between graph and textual modalities, facilitating dynamic interactions between GNNs and PLMs. Our experiments on the graph-to-text generation task demonstrate that GraSAME outperforms baseline models and achieves results comparable to state-of-the-art (SOTA) models on WebNLG datasets. Furthermore, compared to SOTA models, GraSAME eliminates the need for extra pre-training tasks to adjust graph inputs and reduces the number of trainable parameters by over 100 million.