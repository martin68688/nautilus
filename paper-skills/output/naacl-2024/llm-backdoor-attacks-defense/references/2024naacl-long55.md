---
title: "CoUDA: Coherence Evaluation via Unified Data Augmentation"
source: "https://aclanthology.org/2024.naacl-long.55/"
categories: ['llm-evaluation-summarization-argument-extraction', 'llm-backdoor-attacks-defense']
tags: ['coherence-evaluation', 'data-augmentation', 'unified-framework']
venue: "NAACL 2024"
tldr: "Proposes CoUDA, a unified data augmentation framework for training better coherence evaluation models."
---

# CoUDA: Coherence Evaluation via Unified Data Augmentation

**Source**: [https://aclanthology.org/2024.naacl-long.55/](https://aclanthology.org/2024.naacl-long.55/)

**TLDR**: Proposes CoUDA, a unified data augmentation framework for training better coherence evaluation models.

## Abstract

AbstractCoherence evaluation aims to assess the organization and structure of a discourse, which remains challenging even in the era of large language models. Due to the scarcity of annotated data, data augmentation is commonly used for training coherence evaluation models. However, previous augmentations for this task primarily rely on heuristic rules, lacking designing criteria as guidance.In this paper, we take inspiration from linguistic theory of discourse structure, and propose a data augmentation framework named CoUDA. CoUDA breaks down discourse coherence into global and local aspects, and designs augmentation strategies for both aspects, respectively.Especially for local coherence, we propose a novel generative strategy for constructing augmentation samples, which involves post-pretraining a generative model and applying two controlling mechanisms to control the difficulty of generated samples. During inference, CoUDA also jointly evaluates both global and local aspects to comprehensively assess the overall coherence of a discourse.Extensive experiments in coherence evaluation show that, with only 233M parameters, CoUDA achieves state-of-the-art performance in both pointwise scoring and pairwise ranking tasks, even surpassing recent GPT-3.5 and GPT-4 based metrics.