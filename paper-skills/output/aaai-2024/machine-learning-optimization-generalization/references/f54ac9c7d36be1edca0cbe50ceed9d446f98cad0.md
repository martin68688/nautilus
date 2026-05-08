---
title: "Combating Semantic Contamination in Learning with Label Noise"
source: "https://www.semanticscholar.org/paper/f54ac9c7d36be1edca0cbe50ceed9d446f98cad0"
categories: ['machine-learning-optimization-generalization', 'fair-division-matching-mechanism-design', 'semantic-anomaly-prediction-analysis']
tags: ['label-noise', 'semantic-contamination', 'robust-learning']
venue: "AAAI 2024"
tldr: "Identifies and combats semantic contamination introduced by label refurbishment methods in learning with noisy labels."
---

# Combating Semantic Contamination in Learning with Label Noise

**Source**: [https://www.semanticscholar.org/paper/f54ac9c7d36be1edca0cbe50ceed9d446f98cad0](https://www.semanticscholar.org/paper/f54ac9c7d36be1edca0cbe50ceed9d446f98cad0)

**TLDR**: Identifies and combats semantic contamination introduced by label refurbishment methods in learning with noisy labels.

## Abstract

Noisy labels can negatively impact the performance of deep neural networks. One common solution is label refurbishment, which involves reconstructing noisy labels through predictions and distributions. However, these methods may introduce problematic semantic associations, a phenomenon that we identify as Semantic Contamination. Through an analysis of Robust LR, a representative label refurbishment method, we found that utilizing the logits of views for refurbishment does not adequately balance the semantic information of individual classes. Conversely, using the logits of models fails to maintain consistent semantic relationships across models, which explains why label refurbishment methods frequently encounter issues related to Semantic Contamination. To address this issue, we propose a novel method called Collaborative Cross Learning, which utilizes semi-supervised learning on refurbished labels to extract appropriate semantic associations from embeddings across views and models. Experimental results show that our method outperforms existing approaches on both synthetic and real-world noisy datasets, effectively mitigating the impact of label noise and Semantic Contamination.