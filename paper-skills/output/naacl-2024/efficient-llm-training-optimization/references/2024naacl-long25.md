---
title: "Simple and effective data augmentation for compositional generalization"
source: "https://aclanthology.org/2024.naacl-long.25/"
categories: ['continual-learning-language-model-adaptation', 'efficient-llm-training-optimization']
tags: ['compositional_generalization', 'data_augmentation', 'seq2seq']
venue: "NAACL 2024"
tldr: "Simple data augmentation via sampling and backtranslation improves compositional generalization in seq2seq models."
---

# Simple and effective data augmentation for compositional generalization

**Source**: [https://aclanthology.org/2024.naacl-long.25/](https://aclanthology.org/2024.naacl-long.25/)

**TLDR**: Simple data augmentation via sampling and backtranslation improves compositional generalization in seq2seq models.

## Abstract

AbstractCompositional generalization, the ability to predict complex meanings from training on simpler sentences, poses challenges for powerful pretrained seq2seq models. In this paper, we show that data augmentation methods that sample MRs and backtranslate them can be effective for compositional generalization, but only if we sample from the right distribution. Remarkably, sampling from a uniform distribution performs almost as well as sampling from the test distribution, and greatly outperforms earlier methods that sampled from the training distribution.We further conduct experiments to investigate the reason why this happens and where the benefit of such data augmentation methods come from.