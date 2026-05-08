---
title: "CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP"
source: "https://aclanthology.org/2024.findings-naacl.238/"
categories: ['llm-ranking-benchmarking-adaptation', 'active-learning-weak-supervision-text-classification']
tags: ['data-augmentation', 'low-resource', 'constrained-generation']
venue: "NAACL 2024"
tldr: "Presents a training-free data augmentation method using constrained generation with LLMs for low-resource NLP tasks."
---

# CoDa: Constrained Generation based Data Augmentation for Low-Resource NLP

**Source**: [https://aclanthology.org/2024.findings-naacl.238/](https://aclanthology.org/2024.findings-naacl.238/)

**TLDR**: Presents a training-free data augmentation method using constrained generation with LLMs for low-resource NLP tasks.

## Abstract

AbstractWe present CoDa (**Co**nstrained Generation based **Da**ta Augmentation), a controllable, effective, and *training-free* data augmentation technique for low-resource (data-scarce) NLP. Our approach is based on prompting off-the-shelf instruction-following Large Language Models (LLMs) for generating text that satisfies a set of constraints. Precisely, we extract a set of simple constraints from every instance in the low-resource dataset and verbalize them to prompt an LLM to generate novel and diverse training instances. Our findings reveal that synthetic data that follows simple constraints in the downstream dataset act as highly effective augmentations, and CoDa can achieve this without intricate decoding-time constrained generation techniques or fine-tuning with complex algorithms that eventually make the model biased toward the small number of training instances. Additionally, CoDa is the first framework that provides users explicit control over the augmentation generation process, thereby also allowing easy adaptation to several domains. We demonstrate the effectiveness of CoDa across 11 datasets spanning 3 tasks and 3 low-resource settings. CoDa outperforms all our baselines, qualitatively and quantitatively, with improvements of 0.12%-7.19%. Code is available.