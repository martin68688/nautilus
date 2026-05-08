---
title: "Self-Cleaning: Improving a Named Entity Recognizer Trained on Noisy Data with a Few Clean Instances"
source: "https://aclanthology.org/2024.findings-naacl.14/"
categories: ['efficient-transformer-acceleration-techniques', 'multi-label-active-weak-supervision']
tags: ['ner', 'noisy-data', 'data-cleaning', 'weak-supervision']
venue: "NAACL 2024"
tldr: "A method called Self-Cleaning improves a Named Entity Recognizer trained on noisy data by using a small set of clean instances to identify and correct systematic annotation errors."
---

# Self-Cleaning: Improving a Named Entity Recognizer Trained on Noisy Data with a Few Clean Instances

**Source**: [https://aclanthology.org/2024.findings-naacl.14/](https://aclanthology.org/2024.findings-naacl.14/)

**TLDR**: A method called Self-Cleaning improves a Named Entity Recognizer trained on noisy data by using a small set of clean instances to identify and correct systematic annotation errors.

## Abstract

AbstractTo achieve state-of-the-art performance, one still needs to train NER models on large-scale, high-quality annotated data, an asset that is both costly and time-intensive to accumulate. In contrast, real-world applications often resort to massive low-quality labeled data through non-expert annotators via crowdsourcing and external knowledge bases via distant supervision as a cost-effective alternative. However, these annotation methods result in noisy labels, which in turn lead to a notable decline in performance. Hence, we propose to denoise the noisy NER data with guidance from a small set of clean instances. Along with the main NER model we train a discriminator model and use its outputs to recalibrate the sample weights. The discriminator is capable of detecting both span and category errors with different discriminative prompts. Results on public crowdsourcing and distant supervision datasets show that the proposed method can consistently improve performance with a small guidance set.