---
title: "UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios"
source: "https://www.semanticscholar.org/paper/a11df2bdba5d8d314c43b41be1e2ebaeebcad3a6"
categories: ['federated-learning-optimization-and-robustness', 'vision-language-geospatial-analysis']
tags: ['multimodal-benchmark', 'urban-scenarios', 'large-multimodal-models', 'multi-view']
venue: "AAAI 2024"
tldr: "Introduces a comprehensive benchmark for evaluating large multimodal models on multi-view urban understanding tasks."
---

# UrBench: A Comprehensive Benchmark for Evaluating Large Multimodal Models in Multi-View Urban Scenarios

**Source**: [https://www.semanticscholar.org/paper/a11df2bdba5d8d314c43b41be1e2ebaeebcad3a6](https://www.semanticscholar.org/paper/a11df2bdba5d8d314c43b41be1e2ebaeebcad3a6)

**TLDR**: Introduces a comprehensive benchmark for evaluating large multimodal models on multi-view urban understanding tasks.

## Abstract

Recent evaluations of Large Multimodal Models (LMMs) have explored their capabilities in various domains, with only few benchmarks specifically focusing on urban environments. Moreover, existing urban benchmarks have been limited to evaluating LMMs with basic region-level urban tasks under singular views, leading to incomplete evaluations of LMMs' abilities in urban environments.
To address these issues, we present UrBench, a comprehensive benchmark designed for evaluating LMMs in complex multi-view urban scenarios. UrBench contains 11.6K meticulously curated questions at both region-level and role-level that cover 4 task dimensions: Geo-Localization, Scene Reasoning, Scene Understanding, and Object Understanding, totaling 14 task types. In constructing UrBench, we utilize data from existing datasets and additionally collect data from 11 cities, creating new annotations using a cross-view detection-matching method. With these images and annotations, we then integrate LMM-based, rule-based, and human-based methods to construct large-scale high-quality questions. Our evaluations on 21 LMMs show that current LMMs struggle in the urban environments in several aspects. Even the best performing GPT-4o lags behind humans in most tasks, ranging from simple tasks such as counting to complex tasks such as orientation, localization and object attribute recognition, with an average performance gap of 17.4%. Our benchmark also reveals that LMMs exhibit inconsistent behaviors with different urban views, especially with respect to understanding cross-view relations.