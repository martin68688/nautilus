---
title: "A Novel Two-step Fine-tuning Framework for Transfer Learning in Low-Resource Neural Machine Translation"
source: "https://aclanthology.org/2024.findings-naacl.203/"
categories: ['continual-learning-language-model-adaptation']
tags: ['transfer-learning', 'low-resource', 'nmt', 'fine-tuning']
venue: "NAACL 2024"
tldr: "A two-step fine-tuning framework for transfer learning in low-resource neural machine translation."
---

# A Novel Two-step Fine-tuning Framework for Transfer Learning in Low-Resource Neural Machine Translation

**Source**: [https://aclanthology.org/2024.findings-naacl.203/](https://aclanthology.org/2024.findings-naacl.203/)

**TLDR**: A two-step fine-tuning framework for transfer learning in low-resource neural machine translation.

## Abstract

AbstractExisting transfer learning methods for neural machine translation typically use a well-trained translation model (i.e., a parent model) of a high-resource language pair to directly initialize a translation model (i.e., a child model) of a low-resource language pair, and the child model is then fine-tuned with corresponding datasets. In this paper, we propose a novel two-step fine-tuning (TSFT) framework for transfer learning in low-resource neural machine translation. In the first step, we adjust the parameters of the parent model to fit the child language by using the child source data. In the second step, we transfer the adjusted parameters to the child model and fine-tune it with a proposed distillation loss for efficient optimization. Our experimental results on five low-resource translations demonstrate that our framework yields significant improvements over various strong transfer learning baselines. Further analysis demonstrated the effectiveness of different components in our framework.