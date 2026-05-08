---
title: "Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?"
source: "https://aclanthology.org/2024.naacl-industry.33/"
categories: ['llm-evaluation-summarization-argument-extraction', 'llm-edge-distillation']
tags: ['summarization', 'efficiency', 'model-compression']
venue: "NAACL 2024"
tldr: "Investigates whether smaller, distilled LLMs can perform well on real-world meeting summarization tasks."
---

# Tiny Titans: Can Smaller Large Language Models Punch Above Their Weight in the Real World for Meeting Summarization?

**Source**: [https://aclanthology.org/2024.naacl-industry.33/](https://aclanthology.org/2024.naacl-industry.33/)

**TLDR**: Investigates whether smaller, distilled LLMs can perform well on real-world meeting summarization tasks.

## Abstract

AbstractLarge Language Models (LLMs) have demonstrated impressive capabilities to solve a wide range of tasks without being explicitly fine-tuned on task-specific datasets. However, deploying LLMs in the real world is not trivial, as it requires substantial computing resources. In this paper, we investigate whether smaller, Compact LLMs are a good alternative to the comparatively Larger LLMs to address significant costs associated with utilizing LLMs in the real world. In this regard, we study the meeting summarization task in a real-world industrial environment and conduct extensive experiments by comparing the performance of fine-tuned compact LLMs (FLAN-T5, TinyLLaMA, LiteLLaMA, etc.) with zero-shot larger LLMs (LLaMA-2, GPT-3.5, PaLM-2). We observe that most smaller LLMs, even after fine-tuning, fail to outperform larger zero-shot LLMs in meeting summarization datasets. However, a notable exception is FLAN-T5 (780M parameters), which achieves performance on par with zero-shot Larger LLMs (from 7B to above 70B parameters), while being significantly smaller. This makes compact LLMs like FLAN-T5 a suitable cost-efficient LLM for real-world industrial deployment.