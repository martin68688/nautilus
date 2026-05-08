---
title: "Exploiting the Social-Like Prior in Transformer for Visual Reasoning"
source: "https://www.semanticscholar.org/paper/0eeedd5c5ad085b51dfaffa7386ba7cb539bbf4e"
categories: ['topological-geometric-data-representation', 'causal-inference-graph-machine-learning', 'vision-language-geospatial-analysis']
tags: ['visual-reasoning', 'transformer', 'attention-mechanism']
venue: "AAAI 2024"
tldr: "Enhances transformer-based visual reasoning by exploiting a social-like prior in attention."
---

# Exploiting the Social-Like Prior in Transformer for Visual Reasoning

**Source**: [https://www.semanticscholar.org/paper/0eeedd5c5ad085b51dfaffa7386ba7cb539bbf4e](https://www.semanticscholar.org/paper/0eeedd5c5ad085b51dfaffa7386ba7cb539bbf4e)

**TLDR**: Enhances transformer-based visual reasoning by exploiting a social-like prior in attention.

## Abstract

Benefiting from instrumental global dependency modeling of self-attention (SA), transformer-based approaches have become the pivotal choices for numerous downstream visual reasoning tasks, such as visual question answering (VQA) and referring expression comprehension (REC). However, some studies have recently suggested that SA tends to suffer from rank collapse thereby inevitably leads to representation degradation as the transformer layer goes deeper. Inspired by social network theory, we attempt to make an analogy between social behavior and regional information interaction in SA, and harness two crucial notions of structural hole and degree centrality in social network to explore the possible optimization towards SA learning, which naturally deduces two plug-and-play social-like modules. Based on structural hole, the former module allows to make information interaction in SA more structured, which effectively avoids redundant information aggregation and global feature homogenization for better rank remedy, followed by latter module to comprehensively characterize and refine the representation discrimination via considering degree centrality of regions and transitivity of relations. Without bells and whistles, our model outperforms a bunch of baselines by a noticeable margin when considering our social-like prior on five benchmarks in VQA and REC tasks, and a series of explanatory results are showcased to sufficiently reveal the social-like behaviors in SA.