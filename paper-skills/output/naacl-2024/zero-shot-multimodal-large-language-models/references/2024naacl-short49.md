---
title: "Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models"
source: "https://aclanthology.org/2024.naacl-short.49/"
categories: ['zero-shot-multimodal-large-language-models', 'causal-inference-large-language-models']
tags: ['zero-shot_NER', 'self_improvement', 'LLM']
venue: "NAACL 2024"
tldr: "A training-free self-improving framework enhances zero-shot named entity recognition performance using large language models."
---

# Self-Improving for Zero-Shot Named Entity Recognition with Large Language Models

**Source**: [https://aclanthology.org/2024.naacl-short.49/](https://aclanthology.org/2024.naacl-short.49/)

**TLDR**: A training-free self-improving framework enhances zero-shot named entity recognition performance using large language models.

## Abstract

AbstractExploring the application of powerful large language models (LLMs) on the named entity recognition (NER) task has drawn much attention recently. This work pushes the performance boundary of zero-shot NER with LLMs by proposing a training-free self-improving framework, which utilizes an unlabeled corpus to stimulate the self-learning ability of LLMs. First, we use the LLM to make predictions on the unlabeled corpus using self-consistency and obtain a self-annotated dataset. Second, we explore various strategies to select reliable annotations to form a reliable self-annotated dataset. Finally, for each test input, we retrieve demonstrations from the reliable self-annotated dataset and perform inference via in-context learning. Experiments on four benchmarks show substantial performance improvements achieved by our framework. Through comprehensive experimental analysis, we find that increasing the size of unlabeled corpus or iterations of self-improving does not guarantee further improvement, but the performance might be boosted via more advanced strategies for reliable annotation selection.