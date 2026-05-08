---
title: "XferBench: a Data-Driven Benchmark for Emergent Language"
source: "https://aclanthology.org/2024.naacl-long.82/"
categories: ['metaphor-identification-political-framing']
tags: ['emergent-language', 'benchmark', 'evaluation']
venue: "NAACL 2024"
tldr: "Introduces a data-driven benchmark to evaluate the quality of emergent languages by their similarity to human language."
---

# XferBench: a Data-Driven Benchmark for Emergent Language

**Source**: [https://aclanthology.org/2024.naacl-long.82/](https://aclanthology.org/2024.naacl-long.82/)

**TLDR**: Introduces a data-driven benchmark to evaluate the quality of emergent languages by their similarity to human language.

## Abstract

AbstractIn this paper, we introduce a benchmark for evaluating the overall quality of emergent languages using data-driven methods. Specifically, we interpret the notion of the “quality” of an emergent language as its similarity to human language within a deep learning framework. We measure this by using the emergent language as pretraining data for a downstream NLP tasks in human language—the better the downstream performance, the better the emergent language. We implement this benchmark as an easy-to-use Python package that only requires a text file of utterances from the emergent language to be evaluated. Finally, we empirically test the benchmark’s validity using human, synthetic, and emergent language baselines.