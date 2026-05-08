---
title: "Instruction Tuning with Human Curriculum"
source: "https://aclanthology.org/2024.findings-naacl.82/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['instruction-tuning', 'curriculum-learning', 'model-optimization']
venue: "NAACL 2024"
tldr: "Introduces Curriculum Instruction Tuning, a method using diverse curriculum strategies and a synthetic instruction-response generation framework to improve LLM training."
---

# Instruction Tuning with Human Curriculum

**Source**: [https://aclanthology.org/2024.findings-naacl.82/](https://aclanthology.org/2024.findings-naacl.82/)

**TLDR**: Introduces Curriculum Instruction Tuning, a method using diverse curriculum strategies and a synthetic instruction-response generation framework to improve LLM training.

## Abstract

AbstractIn this work, we (1) introduce Curriculum Instruction Tuning, (2) explore the potential advantages of employing diverse curriculum strategies, and (3) delineate a synthetic instruction-response generation framework that complements our theoretical approach. Distinct from the existing instruction tuning dataset, our generation pipeline is systematically structured to emulate the sequential and orderly characteristic of human learning. Additionally, we describe a methodology for generating instruction-response datasets that extensively span the various stages of human education, from middle school through the graduate level, utilizing educational subject catalogs.Before training, we meticulously organize the instruction data to ensure that questions escalate in difficulty regarding (A) the subject matter and (B) the intricacy of the instructions. The findings of our study reveal that substantial improvements in performance can be achieved through the mere application of curriculum ordering to instruction data—achieving gains of +4.76 on TruthfulQA, +2.98 on MMLU, +2.8 on OpenbookQA, and +1.28 on ARC-hard—compared to random shuffling. This enhancement is achieved without incurring additional computational expenses. Through comprehensive experimentation, we observe that the advantages of our proposed method are consistently evident across nine benchmarks.