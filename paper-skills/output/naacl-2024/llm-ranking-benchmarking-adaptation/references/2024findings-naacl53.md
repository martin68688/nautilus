---
title: "CLEAN–EVAL: Clean Evaluation on Contaminated Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.53/"
categories: ['llm-ranking-benchmarking-adaptation', 'large-language-model-evaluation-augmentation']
tags: ['evaluation', 'data-contamination', 'benchmark']
venue: "NAACL 2024"
tldr: "Proposes a method for clean evaluation of LLMs on potentially contaminated benchmarks."
---

# CLEAN–EVAL: Clean Evaluation on Contaminated Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.53/](https://aclanthology.org/2024.findings-naacl.53/)

**TLDR**: Proposes a method for clean evaluation of LLMs on potentially contaminated benchmarks.

## Abstract

AbstractWe are currently in an era of fierce competition among various large language models (LLMs), continuously pushing the boundaries of benchmark performance. However, genuinely assessing the capabilities of these LLMs has become a challenging and critical issue due to potential data contamination. In this paper, we propose a novel and valuable method, Clean-Eval, which mitigates the issue of data contamination and evaluates the LLMs more cleanly. Clean-Eval employs a neural-based model to paraphrase and back-translate the contaminated data into a candidate set, generating expressions with the same meaning but in different surface forms. A semantic detector is then used to filter those generated low-quality samples to narrow down this candidate set. Candidates with moderate BLEURT scores against the original samples are selected as the final evaluation set. According to human assessment, this set is almost semantically equivalent to the original contamination set but expressed differently. We conduct experiments on 20 existing benchmarks across diverse tasks, and results demonstrate that Clean-Eval substantially restores the actual evaluation results on contaminated LLMs under both few-shot learning and fine-tuning scenarios.