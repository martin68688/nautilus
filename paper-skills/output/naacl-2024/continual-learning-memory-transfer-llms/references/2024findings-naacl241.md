---
title: "RobustSentEmbed: Robust Sentence Embeddings Using Adversarial Self-Supervised Contrastive Learning"
source: "https://aclanthology.org/2024.findings-naacl.241/"
categories: ['contrastive-and-generative-representation-learning', 'continual-learning-memory-transfer-llms']
tags: ['sentence-embeddings', 'robustness', 'adversarial-learning', 'self-supervised', 'contrastive-learning']
venue: "NAACL 2024"
tldr: "Proposes RobustSentEmbed, a method for learning robust sentence embeddings using adversarial self-supervised contrastive learning."
---

# RobustSentEmbed: Robust Sentence Embeddings Using Adversarial Self-Supervised Contrastive Learning

**Source**: [https://aclanthology.org/2024.findings-naacl.241/](https://aclanthology.org/2024.findings-naacl.241/)

**TLDR**: Proposes RobustSentEmbed, a method for learning robust sentence embeddings using adversarial self-supervised contrastive learning.

## Abstract

AbstractPre-trained language models (PLMs) have consistently demonstrated outstanding performance across a diverse spectrum of natural language processing tasks. Nevertheless, despite their success with unseen data, current PLM-based representations often exhibit poor robustness in adversarial settings. In this paper, we introduce RobustSentEmbed, a self-supervised sentence embedding framework designed to improve both generalization and robustness in diverse text representation tasks and against a diverse set of adversarial attacks. Through the generation of high-risk adversarial perturbations and their utilization in a novel objective function, RobustSentEmbed adeptly learns high-quality and robust sentence embeddings. Our experiments confirm the superiority of RobustSentEmbed over state-of-the-art representations. Specifically, Our framework achieves a significant reduction in the success rate of various adversarial attacks, notably reducing the BERTAttack success rate by almost half (from 75.51% to 38.81%). The framework also yields improvements of 1.59% and 0.23% in semantic textual similarity tasks and various transfer tasks, respectively.