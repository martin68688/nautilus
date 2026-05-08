---
title: "AdaPT: A Set of Guidelines for Hyperbolic Multimodal Multilingual NLP"
source: "https://aclanthology.org/2024.findings-naacl.114/"
categories: ['efficient-transformer-optimization-techniques', 'zero-shot-few-shot-multimodal-optimization']
tags: ['multimodal', 'multilingual', 'hyperbolic', 'geometry']
venue: "NAACL 2024"
tldr: "AdaPT provides guidelines for using hyperbolic spaces to better capture complex geometries in multimodal, multilingual NLP models."
---

# AdaPT: A Set of Guidelines for Hyperbolic Multimodal Multilingual NLP

**Source**: [https://aclanthology.org/2024.findings-naacl.114/](https://aclanthology.org/2024.findings-naacl.114/)

**TLDR**: AdaPT provides guidelines for using hyperbolic spaces to better capture complex geometries in multimodal, multilingual NLP models.

## Abstract

AbstractThe Euclidean space is the familiar space for training neural models and performing arithmetic operations.However, many data types inherently possess complex geometries, and model training methods involve operating over their latent representations, which cannot be effectively captured in the Euclidean space.The hyperbolic space provides a more generalized representative geometry to model the hierarchical complexities of the tree-like structure of natural language.We propose AdaPT a set of guidelines for initialization, parametrization, and training of neural networks, which adapts to the dataset and can be used with different manifolds. AdaPT can be generalized over any existing neural network training methodology and leads to more stable training without a substantial increase in training time.We apply AdaPT guidelines over two state-of-the-art deep learning approaches and empirically demonstrate its effectiveness through experiments on three tasks over 12 languages across speech and text.Through extensive qualitative analysis, we put forward the applicability of AdaPT as a set of guidelines optimally utilizing the manifold geometry, which can be extended to various downstream tasks across languages and modalities.