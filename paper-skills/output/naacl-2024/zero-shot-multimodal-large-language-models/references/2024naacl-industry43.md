---
title: "Self-Regulated Data-Free Knowledge Amalgamation for Text Classification"
source: "https://aclanthology.org/2024.naacl-industry.43/"
categories: ['efficient-llm-training-optimization', 'zero-shot-multimodal-large-language-models', 'continual-learning-language-model-adaptation']
tags: ['knowledge-amalgamation', 'text-classification', 'data-free', 'model-merging']
venue: "NAACL 2024"
tldr: "A self-regulated, data-free knowledge amalgamation method for text classification that merges multiple pre-trained teacher models into a compact student model without requiring original training data."
---

# Self-Regulated Data-Free Knowledge Amalgamation for Text Classification

**Source**: [https://aclanthology.org/2024.naacl-industry.43/](https://aclanthology.org/2024.naacl-industry.43/)

**TLDR**: A self-regulated, data-free knowledge amalgamation method for text classification that merges multiple pre-trained teacher models into a compact student model without requiring original training data.

## Abstract

AbstractRecently, there has been a growing availability of pre-trained text models on various model repositories. These models greatly reduce the cost of training new models from scratch as they can be fine-tuned for specific tasks or trained on large datasets. However, these datasets may not be publicly accessible due to the privacy, security, or intellectual property issues. In this paper, we aim to develop a lightweight student network that can learn from multiple teacher models without accessing their original training data. Hence, we investigate Data-Free Knowledge Amalgamation (DFKA), a knowledge-transfer task that combines insights from multiple pre-trained teacher models and transfers them effectively to a compact student network. To accomplish this, we propose STRATANET, a modeling framework comprising: (a) a steerable data generator that produces text data tailored to each teacher and (b) an amalgamation module that implements a self-regulative strategy using confidence estimates from the teachers’ different layers to selectively integrate their knowledge and train a versatile student. We evaluate our method on three benchmark text classification datasets with varying labels or domains. Empirically, we demonstrate that the student model learned using our STRATANET outperforms several baselines significantly under data-driven and data-free constraints.