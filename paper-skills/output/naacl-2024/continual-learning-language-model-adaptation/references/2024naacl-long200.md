---
title: "SharpSeq: Empowering Continual Event Detection through Sharpness-Aware Sequential-task Learning"
source: "https://aclanthology.org/2024.naacl-long.200/"
categories: ['continual-learning-language-model-adaptation']
tags: ['continual-learning', 'event-detection', 'optimization']
venue: "NAACL 2024"
tldr: "Proposes a sharpness-aware learning method to mitigate catastrophic forgetting in continual event detection."
---

# SharpSeq: Empowering Continual Event Detection through Sharpness-Aware Sequential-task Learning

**Source**: [https://aclanthology.org/2024.naacl-long.200/](https://aclanthology.org/2024.naacl-long.200/)

**TLDR**: Proposes a sharpness-aware learning method to mitigate catastrophic forgetting in continual event detection.

## Abstract

AbstractContinual event detection is a cornerstone in uncovering valuable patterns in many dynamic practical applications, where novel events emerge daily. Existing state-of-the-art approaches with replay buffers still suffer from catastrophic forgetting, partially due to overly simplistic objective aggregation. This oversight disregards complex trade-offs and leads to sub-optimal gradient updates, resulting in performance deterioration across objectives. While there are successful, widely cited multi-objective optimization frameworks for multi-task learning, they lack mechanisms to address data imbalance and evaluate whether a Pareto-optimal solution can effectively mitigate catastrophic forgetting, rendering them unsuitable for direct application to continual learning. To address these challenges, we propose **SharpSeq**, a novel continual learning paradigm leveraging sharpness-aware minimization combined with a generative model to balance training data distribution. Through extensive experiments on multiple real-world datasets, we demonstrate the superior performance of SharpSeq in continual event detection, proving the importance of our approach in mitigating catastrophic forgetting in continual event detection.