---
title: "Low-Rank Adaptation for Multilingual Summarization: An Empirical Study"
source: "https://aclanthology.org/2024.findings-naacl.77/"
categories: ['efficient-llm-training-optimization', 'information-extraction-knowledge-graph-methods']
tags: ['multilingual', 'summarization', 'low-rank-adaptation']
venue: "NAACL 2024"
tldr: "Investigates the use of Parameter-Efficient Fine-Tuning for low-resource multilingual summarization."
---

# Low-Rank Adaptation for Multilingual Summarization: An Empirical Study

**Source**: [https://aclanthology.org/2024.findings-naacl.77/](https://aclanthology.org/2024.findings-naacl.77/)

**TLDR**: Investigates the use of Parameter-Efficient Fine-Tuning for low-resource multilingual summarization.

## Abstract

AbstractAlthough the advancements of pre-trained Large Language Models have significantly accelerated recent progress in NLP, their ever-increasing size poses significant challenges for conventional fine-tuning, especially in memory-intensive tasks. We investigate the potential of Parameter-Efficient Fine-Tuning, focusing on Low-Rank Adaptation (LoRA), in the domain of multilingual summarization, a task that is both challenging (due to typically long inputs), and relatively unexplored. We conduct an extensive study across different data availability scenarios, including high- and low-data settings, and cross-lingual transfer, leveraging models of different sizes. Our findings reveal that LoRA is competitive with full fine-tuning when trained with high quantities of data, and excels in low-data scenarios and cross-lingual transfer. We also study different strategies for few-shot cross-lingual transfer, finding that continued LoRA tuning outperforms full fine-tuning and the dynamic composition of language-specific LoRA modules.