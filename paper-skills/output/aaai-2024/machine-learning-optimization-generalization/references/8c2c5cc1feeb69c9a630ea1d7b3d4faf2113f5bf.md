---
title: "Pushing the Limit of Fine-Tuning for Few-Shot Learning: Where Feature Reusing Meets Cross-Scale Attention"
source: "https://www.semanticscholar.org/paper/8c2c5cc1feeb69c9a630ea1d7b3d4faf2113f5bf"
categories: ['machine-learning-optimization-generalization', 'vision-transformer-semi-supervised-learning']
tags: ['few-shot-learning', 'fine-tuning', 'cross-scale-attention', 'feature-reuse']
venue: "AAAI 2024"
tldr: "Improves few-shot learning by combining feature reuse from pre-training with cross-scale attention during fine-tuning."
---

# Pushing the Limit of Fine-Tuning for Few-Shot Learning: Where Feature Reusing Meets Cross-Scale Attention

**Source**: [https://www.semanticscholar.org/paper/8c2c5cc1feeb69c9a630ea1d7b3d4faf2113f5bf](https://www.semanticscholar.org/paper/8c2c5cc1feeb69c9a630ea1d7b3d4faf2113f5bf)

**TLDR**: Improves few-shot learning by combining feature reuse from pre-training with cross-scale attention during fine-tuning.

## Abstract

Due to the scarcity of training samples, Few-Shot Learning (FSL) poses a significant challenge to capture discriminative object features effectively. The combination of transfer learning and meta-learning has recently been explored by pre-training the backbone features using labeled base data and subsequently fine-tuning the model with target data. However, existing meta-learning methods, which use embedding networks, suffer from scaling limitations when dealing with a few labeled samples, resulting in suboptimal results. Inspired by the latest advances in FSL, we further advance the approach of fine-tuning a pre-trained architecture by a strengthened hierarchical feature representation. The technical contributions of this work include: 1) a hybrid design named Intra-Block Fusion (IBF) to strengthen the extracted features within each convolution block; and 2) a novel Cross-Scale Attention (CSA) module to mitigate the scaling inconsistencies arising from the limited training samples, especially for cross-domain tasks. We conducted comprehensive evaluations on standard benchmarks, including three in-domain tasks (miniImageNet, CIFAR-FS, and FC100), as well as two cross-domain tasks (CDFSL and Meta-Dataset). The results have improved significantly over existing state-of-the-art approaches on all benchmark datasets. In particular, the FSL performance on the in-domain FC100 dataset is more than three points better than the latest PMF (Hu et al. 2022).