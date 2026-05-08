---
title: "Non-contrastive sentence representations via self-supervision"
source: "https://aclanthology.org/2024.findings-naacl.266/"
categories: ['contrastive-and-generative-representation-learning']
tags: ['sentence-embeddings', 'self-supervised', 'non-contrastive']
venue: "NAACL 2024"
tldr: "Explores non-contrastive self-supervised methods for learning sentence representations."
---

# Non-contrastive sentence representations via self-supervision

**Source**: [https://aclanthology.org/2024.findings-naacl.266/](https://aclanthology.org/2024.findings-naacl.266/)

**TLDR**: Explores non-contrastive self-supervised methods for learning sentence representations.

## Abstract

AbstractSample contrastive methods, typically referred to simply as contrastive are the foundation of most unsupervised methods to learn text and sentence embeddings. On the other hand, a different class of self-supervised non-contrastive loss functions and methods have been considered in the computer vision community and referred to as dimension contrastive. In this paper, we thoroughly compare this class of methods with the standard baseline for contrastive sentence embeddings, SimCSE. We find that self-supervised embeddings trained using dimension contrastive objectives can outperform SimCSE on downstream tasks without needing auxiliary loss functions.