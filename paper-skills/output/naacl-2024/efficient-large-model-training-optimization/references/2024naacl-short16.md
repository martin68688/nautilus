---
title: "CELI: Simple yet Effective Approach to Enhance Out-of-Domain Generalization of Cross-Encoders."
source: "https://aclanthology.org/2024.naacl-short.16/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['cross-encoder', 'generalization', 'token-interaction', 'ranking']
venue: "NAACL 2024"
tldr: "Enhancing cross-encoder generalization for text ranking by adding explicit token interaction in similarity layers."
---

# CELI: Simple yet Effective Approach to Enhance Out-of-Domain Generalization of Cross-Encoders.

**Source**: [https://aclanthology.org/2024.naacl-short.16/](https://aclanthology.org/2024.naacl-short.16/)

**TLDR**: Enhancing cross-encoder generalization for text ranking by adding explicit token interaction in similarity layers.

## Abstract

AbstractIn text ranking, it is generally believed that the cross-encoders already gather sufficient token interaction information via the attention mechanism in the hidden layers. However, our results show that the cross-encoders can consistently benefit from additional token interaction in the similarity computation at the last layer. We introduce CELI (Cross-Encoder with Late Interaction), which incorporates a late interaction layer into the current cross-encoder models. This simple method brings 5% improvement on BEIR without compromising in-domain effectiveness or search latency. Extensive experiments show that this finding is consistent across different sizes of the cross-encoder models and the first-stage retrievers. Our findings suggest that boiling all information into the [CLS] token is a suboptimal use for cross-encoders, and advocate further studies to investigate its relevance score mechanism.