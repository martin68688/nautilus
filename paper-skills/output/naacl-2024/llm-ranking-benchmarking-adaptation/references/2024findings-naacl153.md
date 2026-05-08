---
title: "Denoising Attention for Query-aware User Modeling"
source: "https://aclanthology.org/2024.findings-naacl.153/"
categories: ['llm-ranking-benchmarking-adaptation', 'query-aware-attention-user-modeling']
tags: ['personalization', 'user-modeling', 'attention', 'search']
venue: "NAACL 2024"
tldr: "Introduces a denoising attention mechanism for building query-aware user models at query time to personalize search results."
---

# Denoising Attention for Query-aware User Modeling

**Source**: [https://aclanthology.org/2024.findings-naacl.153/](https://aclanthology.org/2024.findings-naacl.153/)

**TLDR**: Introduces a denoising attention mechanism for building query-aware user models at query time to personalize search results.

## Abstract

AbstractPersonalization of search results has gained increasing attention in the past few years, also thanks to the development of Neural Networks-based approaches for Information Retrieval. Recent works have proposed to build user models at query time by leveraging the Attention mechanism, which allows weighing the contribution of the user-related information w.r.t. the current query.This approach allows giving more importance to the user’s interests related to the current search performed by the user.In this paper, we discuss some shortcomings of the Attention mechanism when employed for personalization and introduce a novel Attention variant, the Denoising Attention, to solve them.Denoising Attention adopts a robust normalization scheme and introduces a filtering mechanism to better discern among the user-related data those helpful for personalization.Experimental evaluation shows improvements in MAP, MRR, and NDCG above 15% w.r.t. other Attention variants at the state-of-the-art.