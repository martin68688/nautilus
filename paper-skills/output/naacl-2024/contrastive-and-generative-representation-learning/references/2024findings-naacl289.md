---
title: "Cross-Lingual Summarization with Pseudo-Label Regularization"
source: "https://aclanthology.org/2024.findings-naacl.289/"
categories: ['llm-evaluation-summarization-argument-extraction', 'contrastive-and-generative-representation-learning']
tags: ['cross-lingual', 'summarization', 'pseudo-label', 'regularization']
venue: "NAACL 2024"
tldr: "A pseudo-label regularization method is proposed to improve cross-lingual summarization by leveraging multiple candidate summaries."
---

# Cross-Lingual Summarization with Pseudo-Label Regularization

**Source**: [https://aclanthology.org/2024.findings-naacl.289/](https://aclanthology.org/2024.findings-naacl.289/)

**TLDR**: A pseudo-label regularization method is proposed to improve cross-lingual summarization by leveraging multiple candidate summaries.

## Abstract

AbstractCross-Lingual Summarization (XLS) aims to summarize a document in the source language into a condensed version in the target language, effectively removing language barriers for non-native readers. Previous approaches, however, have the same limitation that only a single reference (gold summary) is exploited during model training, making the base model exposed to an underrepresented hypothesis space since the actual number of possible hypotheses is exponentially large. To alleviate this problem, we present a study adopting pseudo-labels in regularizing standard cross-lingual summarization training. We investigate several components leading to the gains in regularization training with verified experiments involving 8 diverse languages from different families. Conclusively, we show that pseudo-labeling is a simple and effective approach that significantly improves over standard gold reference training in XLS.