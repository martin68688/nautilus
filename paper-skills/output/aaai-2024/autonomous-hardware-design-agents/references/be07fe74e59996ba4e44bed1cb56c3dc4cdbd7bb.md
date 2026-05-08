---
title: "Trend-Aware Supervision: On Learning Invariance for Semi-supervised Facial Action Unit Intensity Estimation"
source: "https://www.semanticscholar.org/paper/be07fe74e59996ba4e44bed1cb56c3dc4cdbd7bb"
categories: ['vision-transformer-semi-supervised-learning', 'autonomous-hardware-design-agents']
tags: ['facial-action-units', 'semi-supervised', 'invariance-learning', 'trend-awareness']
venue: "AAAI 2024"
tldr: "Uses trend-aware supervision to learn invariance for semi-supervised facial action unit intensity estimation."
---

# Trend-Aware Supervision: On Learning Invariance for Semi-supervised Facial Action Unit Intensity Estimation

**Source**: [https://www.semanticscholar.org/paper/be07fe74e59996ba4e44bed1cb56c3dc4cdbd7bb](https://www.semanticscholar.org/paper/be07fe74e59996ba4e44bed1cb56c3dc4cdbd7bb)

**TLDR**: Uses trend-aware supervision to learn invariance for semi-supervised facial action unit intensity estimation.

## Abstract

With the increasing need for facial behavior analysis, semi-supervised AU intensity estimation using only keyframe annotations has emerged as a practical and effective solution to relieve the burden of annotation. However, the lack of annotations makes the spurious correlation problem caused by AU co-occurrences and subject variation much more prominent, leading to non-robust intensity estimation that is entangled among AUs and biased among subjects. We observe that trend information inherent in keyframe annotations could act as extra supervision and raising the awareness of AU-specific facial appearance changing trends during training is the key to learning invariant AU-specific features. To this end, we propose Trend-AwareSupervision (TAS), which pursues three kinds of trend awareness, including intra-trend ranking awareness, intra-trend speed awareness, and inter-trend subject awareness. TAS alleviates the spurious correlation problem by raising trend awareness during training to learn AU-specific features that represent the corresponding facial appearance changes, to achieve intensity estimation invariance. Experiments conducted on two commonly used AU benchmark datasets, BP4D and DISFA, show the effectiveness of each kind of awareness. And under trend-aware supervision, the performance can be improved without extra computational or storage costs during inference.