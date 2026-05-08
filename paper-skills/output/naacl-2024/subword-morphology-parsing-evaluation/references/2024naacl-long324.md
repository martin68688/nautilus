---
title: "BPE-knockout: Pruning Pre-existing BPE Tokenisers with Backwards-compatible Morphological Semi-supervision"
source: "https://aclanthology.org/2024.naacl-long.324/"
categories: ['subword-morphology-parsing-evaluation']
tags: ['tokenization', 'BPE', 'morphology', 'pruning']
venue: "NAACL 2024"
tldr: "BPE-knockout prunes existing BPE tokenizers using morphological supervision to improve segmentation without retraining."
---

# BPE-knockout: Pruning Pre-existing BPE Tokenisers with Backwards-compatible Morphological Semi-supervision

**Source**: [https://aclanthology.org/2024.naacl-long.324/](https://aclanthology.org/2024.naacl-long.324/)

**TLDR**: BPE-knockout prunes existing BPE tokenizers using morphological supervision to improve segmentation without retraining.

## Abstract

AbstractByte-pair encoding (BPE) has become the default subword tokeniser in language models (LMs), allowing the representation of an infinite space of text with a finite set of units. Yet, BPE training is unsupervised, receiving no explicit information about a language’s morphology. This results in a subword vocabulary wherein many units are a concatenation of partial morphemes, preventing their formation as tokens. This, in turn, causes consistent intra-word patterns to be displayed inconsistently to downstream models, and bloats the vocabulary, hence requiring unnecessary embedding storage. In this paper, we address this issue by identifying blameworthy BPE merges and removing the resulting subwords from the BPE vocabulary, without impeding further use of merges that relied on them. We find that our method, BPE-knockout, is effective at making BPE’s segmentation positions adhere better to derivational and compound boundaries in English, Dutch and German, and improves token-based tasks in Dutch RoBERTa models, indicating that a tokeniser’s adherence to morphology impacts downstream models. We demonstrate the latter not only by training LMs from scratch, but also by continuing the pre-training of existing LMs. This proves promising, showing that suboptimal tokenisers can be remedied whilst salvaging training cost of downstream LMs.