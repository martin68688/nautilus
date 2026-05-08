---
title: "Fusion Makes Perfection: An Efficient Multi-Grained Matching Approach for Zero-Shot Relation Extraction"
source: "https://aclanthology.org/2024.naacl-short.7/"
categories: ['llm-knowledge-reasoning-retrieval', 'knowledge-conflict-diagnostic-temporal-reasoning']
tags: ['zero-shot', 'relation-extraction', 'semantic-matching']
venue: "NAACL 2024"
tldr: "Proposes an efficient multi-grained matching approach for zero-shot relation extraction."
---

# Fusion Makes Perfection: An Efficient Multi-Grained Matching Approach for Zero-Shot Relation Extraction

**Source**: [https://aclanthology.org/2024.naacl-short.7/](https://aclanthology.org/2024.naacl-short.7/)

**TLDR**: Proposes an efficient multi-grained matching approach for zero-shot relation extraction.

## Abstract

AbstractPredicting unseen relations that cannot be observed during the training phase is a challenging task in relation extraction. Previous works have made progress by matching the semantics between input instances and label descriptions. However, fine-grained matching often requires laborious manual annotation, and rich interactions between instances and label descriptions come with significant computational overhead. In this work, we propose an efficient multi-grained matching approach that uses virtual entity matching to reduce manual annotation cost, and fuses coarse-grained recall and fine-grained classification for rich interactions with guaranteed inference speed.Experimental results show that our approach outperforms the previous State Of The Art (SOTA) methods, and achieves a balance between inference efficiency and prediction accuracy in zero-shot relation extraction tasks.Our code is available at https://github.com/longls777/EMMA.