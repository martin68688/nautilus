---
title: "Identifying Linear Relational Concepts in Large Language Models"
source: "https://aclanthology.org/2024.naacl-long.85/"
categories: ['transformer-language-model-probing']
tags: ['concept-direction', 'latent-space', 'interpretability', 'linear-relations']
venue: "NAACL 2024"
tldr: "This paper presents a technique called linear relational concepts to identify human-interpretable concept directions in the latent space of transformer language models."
---

# Identifying Linear Relational Concepts in Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.85/](https://aclanthology.org/2024.naacl-long.85/)

**TLDR**: This paper presents a technique called linear relational concepts to identify human-interpretable concept directions in the latent space of transformer language models.

## Abstract

AbstractTransformer language models (LMs) have been shown to represent concepts as directions in the latent space of hidden activations. However, for any human-interpretable concept, how can we find its direction in the latent space? We present a technique called linear relational concepts (LRC) for finding concept directions corresponding to human-interpretable concepts by first modeling the relation between subject and object as a linear relational embedding (LRE). We find that inverting the LRE and using earlier object layers results in a powerful technique for finding concept directions that outperforms standard black-box probing classifiers. We evaluate LRCs on their performance as concept classifiers as well as their ability to causally change model output.