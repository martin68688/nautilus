---
title: "Deja vu: Contrastive Historical Modeling with Prefix-tuning for Temporal Knowledge Graph Reasoning"
source: "https://aclanthology.org/2024.findings-naacl.75/"
categories: ['continual-learning-memory-transfer-llms', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['temporal-knowledge-graphs', 'reasoning', 'contrastive-learning', 'prefix-tuning']
venue: "NAACL 2024"
tldr: "Proposes a contrastive historical modeling method with prefix-tuning for reasoning on temporal knowledge graphs."
---

# Deja vu: Contrastive Historical Modeling with Prefix-tuning for Temporal Knowledge Graph Reasoning

**Source**: [https://aclanthology.org/2024.findings-naacl.75/](https://aclanthology.org/2024.findings-naacl.75/)

**TLDR**: Proposes a contrastive historical modeling method with prefix-tuning for reasoning on temporal knowledge graphs.

## Abstract

AbstractTemporal Knowledge Graph Reasoning (TKGR) is the task of inferring missing facts for incomplete TKGs in complex scenarios (e.g., transductive and inductive settings), which has been gaining increasing attention. Recently, to mitigate dependence on structured connections in TKGs, text-based methods have been developed to utilize rich linguistic information from entity descriptions. However, suffering from the enormous parameters and inflexibility of pre-trained language models, existing text-based methods struggle to balance the textual knowledge and temporal information with computationally expensive purpose-built training strategies. To tap the potential of text-based models for TKGR in various complex scenarios, we propose ChapTER, a Contrastive historical modeling framework with prefix-tuning for TEmporal Reasoning. ChapTER feeds history-contextualized text into the pseudo-Siamese encoders to strike a textual-temporal balance via contrastive estimation between queries and candidates. By introducing virtual time prefix tokens, it applies a prefix-based tuning method to facilitate the frozen PLM capable for TKGR tasks under different settings. We evaluate ChapTER on four transductive and three few-shot inductive TKGR benchmarks, and experimental results demonstrate that ChapTER achieves superior performance compared to competitive baselines with only 0.17% tuned parameters. We conduct thorough analysis to verify the effectiveness, flexibility and efficiency of ChapTER.