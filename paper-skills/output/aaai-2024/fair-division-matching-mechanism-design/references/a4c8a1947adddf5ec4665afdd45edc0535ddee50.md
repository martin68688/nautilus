---
title: "Data Adaptive Traceback for Vision-Language Foundation Models in Image Classification"
source: "https://www.semanticscholar.org/paper/a4c8a1947adddf5ec4665afdd45edc0535ddee50"
categories: ['vision-transformer-semi-supervised-learning', 'fair-division-matching-mechanism-design']
tags: ['vision-language-models', 'adaptation', 'data-quality', 'image-classification']
venue: "AAAI 2024"
tldr: "A data adaptive traceback method to improve vision-language foundation model adaptation by handling weak image-text correlations in pre-training data."
---

# Data Adaptive Traceback for Vision-Language Foundation Models in Image Classification

**Source**: [https://www.semanticscholar.org/paper/a4c8a1947adddf5ec4665afdd45edc0535ddee50](https://www.semanticscholar.org/paper/a4c8a1947adddf5ec4665afdd45edc0535ddee50)

**TLDR**: A data adaptive traceback method to improve vision-language foundation model adaptation by handling weak image-text correlations in pre-training data.

## Abstract

Vision-language foundation models have been incredibly successful in a wide range of downstream computer vision tasks using adaptation methods. However, due to the high cost of obtaining pre-training datasets, pairs with weak image-text correlation in the data exist in large numbers. We call them weak-paired samples. Due to the limitations of these weak-paired samples, the pre-training model are unable to mine all the knowledge from pre-training data. The existing adaptation methods do not consider the missing knowledge, which may lead to crucial task-related knowledge for the downstream tasks being ignored. To address this issue, we propose a new adaptation framework called Data Adaptive Traceback (DAT). Specifically, we utilize a zero-shot-based method to extract the most downstream task-related subset of the pre-training data to enable the downstream tasks. Furthermore, we adopt a pseudo-label-based semi-supervised technique to reuse the pre-training images and a vision-language contrastive learning method to address the confirmation bias issue in semi-supervised learning. We conduct extensive experiments that show our proposed DAT approach meaningfully improves various benchmark datasets’ performance over traditional adaptation methods by simply.