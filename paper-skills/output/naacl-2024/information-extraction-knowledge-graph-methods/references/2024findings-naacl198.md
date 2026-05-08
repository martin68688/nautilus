---
title: "COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning"
source: "https://aclanthology.org/2024.findings-naacl.198/"
categories: ['llm-evaluation-summarization-opinion-analysis', 'information-extraction-knowledge-graph-methods']
tags: ['code-mixing', 'instruction-tuning', 'multilingual-qa']
venue: "NAACL 2024"
tldr: "Presents a code-mixing English-centric large language model for multilingual instruction tuning to improve low-resource language QA."
---

# COMMIT: Code-Mixing English-Centric Large Language Model for Multilingual Instruction Tuning

**Source**: [https://aclanthology.org/2024.findings-naacl.198/](https://aclanthology.org/2024.findings-naacl.198/)

**TLDR**: Presents a code-mixing English-centric large language model for multilingual instruction tuning to improve low-resource language QA.

## Abstract

AbstractRecently, instruction-tuned large language models (LLMs) are showing prominent performance on various tasks, such as question answering. However, the majority of instruction-tuned LLMs are English-centric, which hinders their application to low-resource language QA. In this paper, we propose COde-Mixed Multilingual Instruction Tuning (COMMIT) to adapt English-centric LLM to low-resource language QA. We point out two main causes of English-centricness: imbalance of unlabeled data, and English-centric instruction tuning datasets. To deviate from English-centric instruction tuning, we propose to specialize code-mixing for instruction tuning, which blocks code-mixing in English templates, to leverage the potential of its superiority. To overcome data imbalance, we perform cross-lingual alignment. The majority of cross-lingual alignment works focused on making representations similar, which is not desirable to decoder-based LLMs, such as LLaMA. Therefore, we propose code-mixed continual causal language modeling to align the decoder. COMMIT improves the exact match score of low-resourced language QA by up to 32x. Code is publicly available.