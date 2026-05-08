---
title: "VLUE: A New Benchmark and Multi-task Knowledge Transfer Learning for Vietnamese Natural Language Understanding"
source: "https://aclanthology.org/2024.findings-naacl.15/"
categories: ['language-model-cultural-linguistic-evaluation']
tags: ['benchmark', 'multilingual', 'transfer-learning']
venue: "NAACL 2024"
tldr: "Introduces VLUE, a new benchmark and multi-task knowledge transfer learning framework for Vietnamese Natural Language Understanding."
---

# VLUE: A New Benchmark and Multi-task Knowledge Transfer Learning for Vietnamese Natural Language Understanding

**Source**: [https://aclanthology.org/2024.findings-naacl.15/](https://aclanthology.org/2024.findings-naacl.15/)

**TLDR**: Introduces VLUE, a new benchmark and multi-task knowledge transfer learning framework for Vietnamese Natural Language Understanding.

## Abstract

AbstractThe success of Natural Language Understanding (NLU) benchmarks in various languages, such as GLUE for English, CLUE for Chinese, KLUE for Korean, and IndoNLU for Indonesian, has facilitated the evaluation of new NLU models across a wide range of tasks. To establish a standardized set of benchmarks for Vietnamese NLU, we introduce the first Vietnamese Language Understanding Evaluation (VLUE) benchmark. The VLUE benchmark encompasses five datasets covering different NLU tasks, including text classification, span extraction, and natural language understanding. To provide an insightful overview of the current state of Vietnamese NLU, we then evaluate seven state-of-the-art pre-trained models, including both multilingual and Vietnamese monolingual models, on our proposed VLUE benchmark. Furthermore, we present CafeBERT, a new state-of-the-art pre-trained model that achieves superior results across all tasks in the VLUE benchmark. Our model combines the proficiency of a multilingual pre-trained model with Vietnamese linguistic knowledge. CafeBERT is developed based on the XLM-RoBERTa model, with an additional pretraining step utilizing a significant amount of Vietnamese textual data to enhance its adaptation to the Vietnamese language. For the purpose of future research, CafeBERT is made publicly available for research purposes.