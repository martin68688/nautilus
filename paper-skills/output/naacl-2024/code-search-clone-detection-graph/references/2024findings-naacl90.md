---
title: "SimSCOOD: Systematic Analysis of Out-of-Distribution Generalization in Fine-tuned Source Code Models"
source: "https://aclanthology.org/2024.findings-naacl.90/"
categories: ['llm-ranking-adaptation-benchmarking', 'code-search-clone-detection-graph']
tags: ['code-models', 'out-of-distribution', 'generalization']
venue: "NAACL 2024"
tldr: "Systematically analyzes out-of-distribution generalization in fine-tuned source code models."
---

# SimSCOOD: Systematic Analysis of Out-of-Distribution Generalization in Fine-tuned Source Code Models

**Source**: [https://aclanthology.org/2024.findings-naacl.90/](https://aclanthology.org/2024.findings-naacl.90/)

**TLDR**: Systematically analyzes out-of-distribution generalization in fine-tuned source code models.

## Abstract

AbstractLarge code datasets have become increasingly accessible for pre-training source code models. However, for the fine-tuning phase, obtaining representative training data that fully covers the code distribution for specific downstream tasks remains challenging due to the task-specific nature and limited labeling resources. These lead to out-of-distribution (OOD) generalization issues with unexpected model inference behaviors that have not been systematically studied yet.In this paper, we contribute the first systematic approach that simulates various OOD scenarios along different dimensions of source code data properties and study the fine-tuned model behaviors in such scenarios. We investigate the behaviors of models under different fine-tuning methodologies, including full fine-tuning and Low-Rank Adaptation (LoRA) fine-tuning methods. Our comprehensive analysis, conducted on four state-of-the-art pretrained models and applied to two code generation tasks, exposes multiple failure modes attributed to OOD generalization issues.