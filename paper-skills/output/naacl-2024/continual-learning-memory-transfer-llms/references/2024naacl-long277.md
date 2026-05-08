---
title: "Generalizable and Stable Finetuning of Pretrained Language Models on Low-Resource Texts"
source: "https://aclanthology.org/2024.naacl-long.277/"
categories: ['efficient-large-model-training-optimization', 'sentiment-emotion-analysis-multimodal-llms', 'continual-learning-memory-transfer-llms']
tags: ['fine-tuning', 'low-resource', 'generalization']
venue: "NAACL 2024"
tldr: "Addresses instability and overfitting in finetuning PLMs on low-resource texts for better generalization."
---

# Generalizable and Stable Finetuning of Pretrained Language Models on Low-Resource Texts

**Source**: [https://aclanthology.org/2024.naacl-long.277/](https://aclanthology.org/2024.naacl-long.277/)

**TLDR**: Addresses instability and overfitting in finetuning PLMs on low-resource texts for better generalization.

## Abstract

AbstractPretrained Language Models (PLMs) have advanced Natural Language Processing (NLP) tasks significantly, but finetuning PLMs on low-resource datasets poses significant challenges such as instability and overfitting. Previous methods tackle these issues by finetuning a strategically chosen subnetwork on a downstream task, while keeping the remaining weights fixed to the pretrained weights. However, they rely on a suboptimal criteria for sub-network selection, leading to suboptimal solutions. To address these limitations, we propose a regularization method based on attention-guided weight mixup for finetuning PLMs. Our approach represents each network weight as a mixup of task-specific weight and pretrained weight, controlled by a learnable attention parameter, providing finer control over sub-network selection. Furthermore, we employ a bi-level optimization (BLO) based framework on two separate splits of the training dataset, improving generalization and combating overfitting. We validate the efficacy of our proposed method through extensive experiments, demonstrating its superiority over previous methods, particularly in the context of finetuning PLMs on low-resource datasets. Our code is available at https://github.com/Sai-Ashish/Attention_guided_weight_mixup_BLO.