---
title: "Time Machine GPT"
source: "https://aclanthology.org/2024.findings-naacl.208/"
categories: ['knowledge-conflict-diagnostic-temporal-reasoning', 'efficient-transformer-optimization-techniques']
tags: ['temporal-adaptation', 'knowledge-updating', 'language-evolution']
venue: "NAACL 2024"
tldr: "Time Machine GPT creates temporally adapted language models by incrementally training on time-stamped data to reflect language evolution."
---

# Time Machine GPT

**Source**: [https://aclanthology.org/2024.findings-naacl.208/](https://aclanthology.org/2024.findings-naacl.208/)

**TLDR**: Time Machine GPT creates temporally adapted language models by incrementally training on time-stamped data to reflect language evolution.

## Abstract

AbstractLarge language models (LLMs) are often trained on extensive, temporally indiscriminate text corpora, reflecting the lack of datasets with temporal metadata. This approach is not aligned with the evolving nature of language. Conventional methods for creating temporally adapted language models often depend on further pre-training static models on time-specific data. This paper presents a new approach: a series of point-in-time LLMs called TimeMachineGPT (TiMaGPT), specifically designed to be nonprognosticative. This ensures they remain uninformed about future factual information and linguistic changes. This strategy is beneficial for understanding language evolution and is of critical importance when applying models in dynamic contexts, such as time-series forecasting, where foresight of future information can prove problematic. We provide access to both the models and training datasets.