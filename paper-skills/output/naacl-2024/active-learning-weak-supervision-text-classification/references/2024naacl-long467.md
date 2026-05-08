---
title: "AnchorAL: Computationally Efficient Active Learning for Large and Imbalanced Datasets"
source: "https://aclanthology.org/2024.naacl-long.467/"
categories: ['active-learning-weak-supervision-text-classification', 'bpe-subword-parsing-evaluation']
tags: ['active-learning', 'imbalanced-data', 'efficiency']
venue: "NAACL 2024"
tldr: "Proposes an efficient anchor-based active learning method for large, imbalanced datasets."
---

# AnchorAL: Computationally Efficient Active Learning for Large and Imbalanced Datasets

**Source**: [https://aclanthology.org/2024.naacl-long.467/](https://aclanthology.org/2024.naacl-long.467/)

**TLDR**: Proposes an efficient anchor-based active learning method for large, imbalanced datasets.

## Abstract

AbstractActive learning for imbalanced classification tasks is challenging as the minority classes naturally occur rarely. Gathering a large pool of unlabelled data is thus essential to capture minority instances. Standard pool-based active learning is computationally expensive on large pools and often reaches low accuracy by overfitting the initial decision boundary, thus failing to explore the input space and find minority instances. To address these issues we propose AnchorAL. At each iteration, AnchorAL chooses class-specific instances from the labelled set, or *anchors*, and retrieves the most similar unlabelled instances from the pool. This resulting *subpool* is then used for active learning. Using a small, fixed-sized subpool AnchorAL allows scaling any active learning strategy to large pools. By dynamically selecting different anchors at each iteration it promotes class balance and prevents overfitting the initial decision boundary, thus promoting the discovery of new clusters of minority instances. Experiments across different classification tasks, active learning strategies, and model architectures AnchorAL is *(i)* faster, often reducing runtime from hours to minutes, *(ii)* trains more performant models, *(iii)* and returns more balanced datasets than competing methods.