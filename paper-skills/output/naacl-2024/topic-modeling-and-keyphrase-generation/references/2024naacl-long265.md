---
title: "HILL: Hierarchy-aware Information Lossless Contrastive Learning for Hierarchical Text Classification"
source: "https://aclanthology.org/2024.naacl-long.265/"
categories: ['contrastive-and-generative-representation-learning', 'topic-modeling-and-keyphrase-generation']
tags: ['hierarchical-classification', 'contrastive-learning', 'self-supervised']
venue: "NAACL 2024"
tldr: "Proposes a contrastive learning method for hierarchical text classification that preserves label hierarchy without data augmentation."
---

# HILL: Hierarchy-aware Information Lossless Contrastive Learning for Hierarchical Text Classification

**Source**: [https://aclanthology.org/2024.naacl-long.265/](https://aclanthology.org/2024.naacl-long.265/)

**TLDR**: Proposes a contrastive learning method for hierarchical text classification that preserves label hierarchy without data augmentation.

## Abstract

AbstractExisting self-supervised methods in natural language processing (NLP), especially hierarchical text classification (HTC), mainly focus on self-supervised contrastive learning, extremely relying on human-designed augmentation rules to generate contrastive samples, which can potentially corrupt or distort the original information. In this paper, we tend to investigate the feasibility of a contrastive learning scheme in which the semantic and syntactic information inherent in the input sample is adequately reserved in the contrastive samples and fused during the learning process. Specifically, we propose an information lossless contrastive learning strategy for HTC, namely Hierarchy-aware Information Lossless contrastive Learning (HILL), which consists of a text encoder representing the input document, and a structure encoder directly generating the positive sample. The structure encoder takes the document embedding as input, extracts the essential syntactic information inherent in the label hierarchy with the principle of structural entropy minimization, and injects the syntactic information into the text representation via hierarchical representation learning. Experiments on three common datasets are conducted to verify the superiority of HILL.