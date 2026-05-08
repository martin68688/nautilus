---
title: "BVT-IMA: Binary Vision Transformer with Information-Modified Attention"
source: "https://www.semanticscholar.org/paper/a3e7cbba2dcfca089e96afb9b31e91a0ba019e07"
categories: ['vision-transformer-semi-supervised-learning']
tags: ['vision-transformer', 'model-compression', 'binarization']
venue: "AAAI 2024"
tldr: "Binarizes vision transformers using an information-modified attention mechanism to reduce computational cost."
---

# BVT-IMA: Binary Vision Transformer with Information-Modified Attention

**Source**: [https://www.semanticscholar.org/paper/a3e7cbba2dcfca089e96afb9b31e91a0ba019e07](https://www.semanticscholar.org/paper/a3e7cbba2dcfca089e96afb9b31e91a0ba019e07)

**TLDR**: Binarizes vision transformers using an information-modified attention mechanism to reduce computational cost.

## Abstract

As a compression method that can significantly reduce the cost of calculations and memories, model binarization has been extensively studied in convolutional neural networks. However, the recently popular vision transformer models pose new challenges to such a technique, in which the binarized models suffer from serious performance drops. In this paper, an attention shifting is observed in the binary multi-head self-attention module, which can influence the information fusion between tokens and thus hurts the model performance. From the perspective of information theory, we find a correlation between attention scores and the information quantity, further indicating that a reason for such a phenomenon may be the loss of the information quantity induced by constant moduli of binarized tokens. Finally, we reveal the information quantity hidden in the attention maps of binary vision transformers and propose a simple approach to modify the attention values with look-up information tables so that improve the model performance. Extensive experiments on CIFAR-100/TinyImageNet/ImageNet-1k demonstrate the effectiveness of the proposed information-modified attention on binary vision transformers.