---
title: "In-Context Example Ordering Guided by Label Distributions"
source: "https://aclanthology.org/2024.findings-naacl.167/"
categories: ['llm-ranking-adaptation-benchmarking', 'large-language-model-evaluation-augmentation']
tags: ['in-context-learning', 'example-ordering', 'label-distributions']
venue: "NAACL 2024"
tldr: "Improves in-context learning performance by ordering examples based on label distributions."
---

# In-Context Example Ordering Guided by Label Distributions

**Source**: [https://aclanthology.org/2024.findings-naacl.167/](https://aclanthology.org/2024.findings-naacl.167/)

**TLDR**: Improves in-context learning performance by ordering examples based on label distributions.

## Abstract

AbstractBy allowing models to predict without task-specific training, in-context learning (ICL) with pretrained LLMs has enormous potential in NLP. However, a number of problems persist in ICL. In particular, its performance is sensitive to the choice and order of in-context examples. Given the same set of in-context examples with different orderings, model performance may vary from near random to near state-of-the-art. In this work, we formulate in-context example ordering as an optimization problem. We examine three problem settings that differ in the assumptions they make about what is known about the task. Inspired by the idea of learning from label proportions, we propose two principles for in-context example ordering guided by model’s probability predictions. We apply our proposed principles to thirteen text classification datasets and nine different autoregressive LLMs with 700M to 13B parameters. We demonstrate that our approach outperforms the baselines by improving the classification accuracy, reducing model miscalibration, and also by selecting better in-context examples.