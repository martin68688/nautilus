---
title: "The Effect of Data Partitioning Strategy on Model Generalizability: A Case Study of Morphological Segmentation"
source: "https://aclanthology.org/2024.naacl-long.157/"
categories: ['bpe-subword-parsing-evaluation', 'llm-research-trends-analysis']
tags: ['morphological-segmentation', 'data-partitioning', 'evaluation', 'generalization']
venue: "NAACL 2024"
tldr: "Studies the impact of data partitioning strategies on the generalization performance of morphological segmentation models."
---

# The Effect of Data Partitioning Strategy on Model Generalizability: A Case Study of Morphological Segmentation

**Source**: [https://aclanthology.org/2024.naacl-long.157/](https://aclanthology.org/2024.naacl-long.157/)

**TLDR**: Studies the impact of data partitioning strategies on the generalization performance of morphological segmentation models.

## Abstract

AbstractRecent work to enhance data partitioning strategies for more realistic model evaluation face challenges in providing a clear optimal choice. This study addresses these challenges, focusing on morphological segmentation and synthesizing limitations related to language diversity, adoption of multiple datasets and splits, and detailed model comparisons. Our study leverages data from 19 languages, including ten indigenous or endangered languages across 10 language families with diverse morphological systems (polysynthetic, fusional, and agglutinative) and different degrees of data availability. We conduct large-scale experimentation with varying sized combinations of training and evaluation sets as well as new test data. Our results show that, when faced with new test data: (1) models trained from random splits are able to achieve higher numerical scores; (2) model rankings derived from random splits tend to generalize more consistently.