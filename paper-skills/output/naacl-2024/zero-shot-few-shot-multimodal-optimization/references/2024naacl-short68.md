---
title: "Language-Independent Representations Improve Zero-Shot Summarization"
source: "https://aclanthology.org/2024.naacl-short.68/"
categories: ['zero-shot-few-shot-multimodal-optimization']
tags: ['zero-shot', 'summarization', 'language-independent']
venue: "NAACL 2024"
tldr: "Improves zero-shot summarization by using language-independent representations to reduce catastrophic forgetting."
---

# Language-Independent Representations Improve Zero-Shot Summarization

**Source**: [https://aclanthology.org/2024.naacl-short.68/](https://aclanthology.org/2024.naacl-short.68/)

**TLDR**: Improves zero-shot summarization by using language-independent representations to reduce catastrophic forgetting.

## Abstract

AbstractFinetuning pretrained models on downstream generation tasks often leads to catastrophic forgetting in zero-shot conditions. In this work, we focus on summarization and tackle the problem through the lens of language-independent representations. After training on monolingual summarization, we perform zero-shot transfer to new languages or language pairs. We first show naively finetuned models are highly language-specific in both output behavior and internal representations, resulting in poor zero-shot performance. Next, we propose query-key (QK) finetuning to decouple task-specific knowledge from the pretrained language generation abilities. Then, after showing downsides of the standard adversarial language classifier, we propose a balanced variant that more directly enforces language-agnostic representations. Moreover, our qualitative analyses show removing source language identity correlates to zero-shot summarization performance. Our code is openly available.