---
title: "Noisy Multi-Label Text Classification via Instance-Label Pair Correction"
source: "https://aclanthology.org/2024.findings-naacl.93/"
categories: ['multi-label-active-weak-supervision']
tags: ['noisy-labels', 'multi-label', 'text-classification', 'instance-correction']
venue: "NAACL 2024"
tldr: "A method for noisy multi-label text classification via correcting instance-label pairs."
---

# Noisy Multi-Label Text Classification via Instance-Label Pair Correction

**Source**: [https://aclanthology.org/2024.findings-naacl.93/](https://aclanthology.org/2024.findings-naacl.93/)

**TLDR**: A method for noisy multi-label text classification via correcting instance-label pairs.

## Abstract

AbstractIn noisy label learning, instance selection based on small-loss criteria has been proven to be highly effective. However, in the case of noisy multi-label text classification (NMLTC), the presence of noise is not limited to the instance-level but extends to the (instance-label) pair-level.This gives rise to two main challenges.(1) The loss information at the pair-level fails to capture the variations between instances. (2) There are two types of noise at the pair-level: false positives and false negatives. Identifying false negatives from a large pool of negative pairs presents an exceedingly difficult task. To tackle these issues, we propose a novel approach called instance-label pair correction (iLaCo), which aims to address the problem of noisy pair selection and correction in NMLTC tasks.Specifically, we first introduce a holistic selection metric that identifies noisy pairs by simultaneously considering global loss information and instance-specific ranking information.Secondly, we employ a filter guided by label correlation to focus exclusively on negative pairs with label relevance. This filter significantly reduces the difficulty of identifying false negatives.Experimental analysis indicates that our framework effectively corrects noisy pairs in NMLTC datasets, leading to a significant improvement in model performance.