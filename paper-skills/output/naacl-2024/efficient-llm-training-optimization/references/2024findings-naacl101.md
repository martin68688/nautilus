---
title: "TagDebias: Entity and Concept Tagging for Social Bias Mitigation in Pretrained Language Models"
source: "https://aclanthology.org/2024.findings-naacl.101/"
categories: ['efficient-llm-training-optimization', 'efficient-transformer-acceleration-techniques']
tags: ['social-bias', 'mitigation', 'pretrained-language-models']
venue: "NAACL 2024"
tldr: "Proposes an entity and concept tagging method to mitigate social biases in pretrained language models."
---

# TagDebias: Entity and Concept Tagging for Social Bias Mitigation in Pretrained Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.101/](https://aclanthology.org/2024.findings-naacl.101/)

**TLDR**: Proposes an entity and concept tagging method to mitigate social biases in pretrained language models.

## Abstract

AbstractPre-trained language models (PLMs) play a crucial role in various applications, including sensitive domains such as the hiring process. However, extensive research has unveiled that these models tend to replicate social biases present in their pre-training data, raising ethical concerns. In this study, we propose the TagDebias method, which proposes debiasing a dataset using type tags. It then proceeds to fine-tune PLMs on this debiased dataset. Experiments show that our proposed TagDebias model, when applied to a ranking task, exhibits significant improvements in bias scores.