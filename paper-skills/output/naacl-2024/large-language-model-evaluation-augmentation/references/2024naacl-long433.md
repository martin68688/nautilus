---
title: "Multilingual Nonce Dependency Treebanks: Understanding how Language Models Represent and Process Syntactic Structure"
source: "https://aclanthology.org/2024.naacl-long.433/"
categories: ['large-language-model-evaluation-augmentation', 'transformer-language-model-probing']
tags: ['syntax', 'dependency-trees', 'probing']
venue: "NAACL 2024"
tldr: "Introduces a framework to create nonce treebanks for probing how LMs represent syntactic structure."
---

# Multilingual Nonce Dependency Treebanks: Understanding how Language Models Represent and Process Syntactic Structure

**Source**: [https://aclanthology.org/2024.naacl-long.433/](https://aclanthology.org/2024.naacl-long.433/)

**TLDR**: Introduces a framework to create nonce treebanks for probing how LMs represent syntactic structure.

## Abstract

AbstractWe introduce SPUD (Semantically Perturbed Universal Dependencies), a framework for creating nonce treebanks for the multilingual Universal Dependencies (UD) corpora. SPUD data satisfies syntactic argument structure, provides syntactic annotations, and ensures grammaticality via language-specific rules. We create nonce data in Arabic, English, French, German, and Russian, and demonstrate two use cases of SPUD treebanks. First, we investigate the effect of nonce data on word co-occurrence statistics, as measured by perplexity scores of autoregressive (ALM) and masked language models (MLM). We find that ALM scores are significantly more affected by nonce data than MLM scores. Second, we show how nonce data affects the performance of syntactic dependency probes. We replicate the findings of Müller-Eberstein et al. (2022) on nonce test data and show that the performance declines on both MLMs and ALMs wrt. original test data. However, a majority of the performance is kept, suggesting that the probe indeed learns syntax independently from semantics.