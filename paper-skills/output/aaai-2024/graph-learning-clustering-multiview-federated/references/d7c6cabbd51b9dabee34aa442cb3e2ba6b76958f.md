---
title: "Temporal Graph Contrastive Learning for Sequential Recommendation"
source: "https://www.semanticscholar.org/paper/d7c6cabbd51b9dabee34aa442cb3e2ba6b76958f"
categories: ['graph-learning-clustering-multiview-federated', 'autonomous-hardware-design-agents']
tags: ['recommendation', 'graph-contrastive-learning', 'temporal', 'sequential', 'self-supervised']
venue: "AAAI 2024"
tldr: "Enhances sequential recommendation via temporal graph contrastive learning to better exploit temporal information in user interaction sequences."
---

# Temporal Graph Contrastive Learning for Sequential Recommendation

**Source**: [https://www.semanticscholar.org/paper/d7c6cabbd51b9dabee34aa442cb3e2ba6b76958f](https://www.semanticscholar.org/paper/d7c6cabbd51b9dabee34aa442cb3e2ba6b76958f)

**TLDR**: Enhances sequential recommendation via temporal graph contrastive learning to better exploit temporal information in user interaction sequences.

## Abstract

Sequential recommendation is a crucial task in understanding users' evolving interests and predicting their future behaviors. While existing approaches on sequence or graph modeling to learn interaction sequences of users have shown promising performance, how to effectively exploit temporal information and deal with the uncertainty noise in evolving user behaviors is still quite challenging. To this end, in this paper, we propose a Temporal Graph Contrastive Learning method for Sequential Recommendation (TGCL4SR) which leverages not only local interaction sequences but also global temporal graphs to comprehend item correlations and analyze user behaviors from a temporal perspective. Specifically, we first devise a Temporal Item Transition Graph (TITG) to fully leverage global interactions to understand item correlations, and augment this graph by dual transformations based on neighbor sampling and time disturbance. Accordingly, we design a Temporal item Transition graph Convolutional network (TiTConv) to capture temporal item transition patterns in TITG. Then, a novel Temporal Graph Contrastive Learning (TGCL) mechanism is designed to enhance the uniformity of representations between augmented graphs from identical sequences. For local interaction sequences, we design a temporal sequence encoder to incorporate time interval embeddings into the architecture of Transformer. At the training stage, we take maximum mean discrepancy and TGCL losses as auxiliary objectives. Extensive experiments on several real-world datasets show the effectiveness of TGCL4SR against state-of-the-art baselines of sequential recommendation.