---
title: "FastFit: Fast and Effective Few-Shot Text Classification with a Multitude of Classes"
source: "https://aclanthology.org/2024.naacl-demo.18/"
categories: ['llm-ranking-adaptation-benchmarking', 'llm-alignment-safety-detoxification']
tags: ['few-shot-classification', 'contrastive-learning', 'many-classes']
venue: "NAACL 2024"
tldr: "Presents a fast and effective few-shot text classification method for many semantically similar classes using contrastive learning."
---

# FastFit: Fast and Effective Few-Shot Text Classification with a Multitude of Classes

**Source**: [https://aclanthology.org/2024.naacl-demo.18/](https://aclanthology.org/2024.naacl-demo.18/)

**TLDR**: Presents a fast and effective few-shot text classification method for many semantically similar classes using contrastive learning.

## Abstract

AbstractWe present FastFit, a Python package designed to provide fast and accurate few-shot classification, especially for scenarios with many semantically similar classes. FastFit utilizes a novel approach integrating batch contrastive learning and token-level similarity score. Compared to existing few-shot learning packages, such as SetFit, Transformers, or few-shot prompting of large language models via API calls, FastFit significantly improves multi-class classification performance in speed and accuracy across various English and Multilingual datasets. FastFit demonstrates a 3-20x improvement in training speed, completing training in just a few seconds. The FastFit package is now available on GitHub, presenting a user-friendly solution for NLP practitioners.