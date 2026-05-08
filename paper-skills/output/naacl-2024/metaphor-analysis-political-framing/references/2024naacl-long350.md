---
title: "ContraSim – Analyzing Neural Representations Based on Contrastive Learning"
source: "https://aclanthology.org/2024.naacl-long.350/"
categories: ['contrastive-and-generative-representation-learning', 'metaphor-analysis-political-framing']
tags: ['representation-learning', 'contrastive', 'similarity']
venue: "NAACL 2024"
tldr: "Introduces a framework for analyzing neural representations based on contrastive learning to evaluate similarity measures."
---

# ContraSim – Analyzing Neural Representations Based on Contrastive Learning

**Source**: [https://aclanthology.org/2024.naacl-long.350/](https://aclanthology.org/2024.naacl-long.350/)

**TLDR**: Introduces a framework for analyzing neural representations based on contrastive learning to evaluate similarity measures.

## Abstract

AbstractRecent work has compared neural network representations via similarity-based analyses to improve model interpretation. The quality of a similarity measure is typically evaluated by its success in assigning a high score to representations that are expected to be matched. However, existing similarity measures perform mediocrely on standard benchmarks. In this work, we develop a new similarity measure, dubbed ContraSim, based on contrastive learning. In contrast to common closed-form similarity measures, ContraSim learns a parameterized measure by using both similar and dissimilar examples. We perform an extensive experimental evaluation of our method, with both language and vision models, on the standard layer prediction benchmark and two new benchmarks that we introduce: the multilingual benchmark and the image–caption benchmark. In all cases, ContraSim achieves much higher accuracy than previous similarity measures, even when presented with challenging examples. Finally, ContraSim is more suitable for the analysis of neural networks, revealing new insights not captured by previous measures.