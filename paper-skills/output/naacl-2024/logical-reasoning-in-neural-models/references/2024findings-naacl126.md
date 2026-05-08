---
title: "Beta-LR: Interpretable Logical Reasoning based on Beta Distribution"
source: "https://aclanthology.org/2024.findings-naacl.126/"
categories: ['logical-reasoning-in-neural-models']
tags: ['logical-reasoning', 'interpretable', 'beta-distribution']
venue: "NAACL 2024"
tldr: "This paper proposes an interpretable logical reasoning method based on the Beta distribution to capture logical information in text."
---

# Beta-LR: Interpretable Logical Reasoning based on Beta Distribution

**Source**: [https://aclanthology.org/2024.findings-naacl.126/](https://aclanthology.org/2024.findings-naacl.126/)

**TLDR**: This paper proposes an interpretable logical reasoning method based on the Beta distribution to capture logical information in text.

## Abstract

AbstractThe logical information contained in text isof significant importance for logical reasoning.Previous approaches have relied on embeddingtext into a low-dimensional vector to capturelogical information and perform reasoning inEuclidean space. These methods involve constructing special graph architectures that matchlogical relations or designing data augmentation frameworks by extending texts based onsymbolic logic. However, it presents two obvious problems. 1) The logical informationreflected in the text exhibits uncertainty that isdifficult to represent using a vector. 2) Integrating logical information requires modeling logical operations (such as ∪, ∩, and ¬), while onlysimple arithmetic operations can be performedin Euclidean space. To address both the problems, we propose Beta-LR, a probabilistic embedding method to capture logical information.Specifically, we embed texts into beta distribution on each dimension to eliminate logical uncertainty. We also define neural operators thatenable interpretability and perform logical operations based on the characteristics of the betadistribution. We conduct experiments on twodatasets, ReClor and LogiQA, and our Beta-LRachieves competitive results. The experimentsdemonstrate that our method effectively captures the logical information in text for reasoning purposes. The source code is available athttps://github.com/myz12138/Beta-LR.