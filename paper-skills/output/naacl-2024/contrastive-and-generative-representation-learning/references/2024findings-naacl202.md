---
title: "Learning Mutually Informed Representations for Characters and Subwords"
source: "https://aclanthology.org/2024.findings-naacl.202/"
categories: ['bpe-subword-parsing-evaluation', 'contrastive-and-generative-representation-learning']
tags: ['subword', 'character', 'representation-learning']
venue: "NAACL 2024"
tldr: "Learns mutually informed representations for characters and subwords to combine different granularities of text information."
---

# Learning Mutually Informed Representations for Characters and Subwords

**Source**: [https://aclanthology.org/2024.findings-naacl.202/](https://aclanthology.org/2024.findings-naacl.202/)

**TLDR**: Learns mutually informed representations for characters and subwords to combine different granularities of text information.

## Abstract

AbstractMost pretrained language models rely on subword tokenization, which processes text as a sequence of subword tokens. However, different granularities of text, such as characters, subwords, and words, can contain different kinds of information. Previous studies have shown that incorporating multiple input granularities improves model generalization, yet very few of them outputs useful representations for each granularity. In this paper, we introduce the entanglement model, aiming to combine character and subword language models. Inspired by vision-language models, our model treats characters and subwords as separate modalities, and it generates mutually informed representations for both granularities as output. We evaluate our model on text classification, named entity recognition, POS-tagging, and character-level sequence labeling (intraword code-switching). Notably, the entanglement model outperforms its backbone language models, particularly in the presence of noisy texts and low-resource languages. Furthermore, the entanglement model even outperforms larger pre-trained models on all English sequence labeling tasks and classification tasks. We make our code publically available.