---
title: "ICXML: An In-Context Learning Framework for Zero-Shot Extreme Multi-Label Classification"
source: "https://aclanthology.org/2024.findings-naacl.134/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-ranking-benchmarking-adaptation']
tags: ['in-context-learning', 'zero-shot', 'extreme-multi-label', 'classification']
venue: "NAACL 2024"
tldr: "ICXML is an in-context learning framework for zero-shot extreme multi-label classification."
---

# ICXML: An In-Context Learning Framework for Zero-Shot Extreme Multi-Label Classification

**Source**: [https://aclanthology.org/2024.findings-naacl.134/](https://aclanthology.org/2024.findings-naacl.134/)

**TLDR**: ICXML is an in-context learning framework for zero-shot extreme multi-label classification.

## Abstract

AbstractThis paper focuses on the task of Extreme Multi-Label Classification (XMC) whose goal is to predict multiple labels for each instance from an extremely large label space. While existing research has primarily focused on fully supervised XMC, real-world scenarios often lack supervision signals, highlighting the importance of zero-shot settings. Given the large label space, utilizing in-context learning approaches is not trivial. We address this issue by introducing In-Context Extreme Multi-label Learning (ICXML), a two-stage framework that cuts down the search space by generating a set of candidate labels through in-context learning and then reranks them. Extensive experiments suggest that ICXML advances the state of the art on two diverse public benchmarks.