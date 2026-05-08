---
title: "Psychometric Predictive Power of Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.129/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'large-language-model-evaluation-augmentation', 'causal-inference-with-language-models']
tags: ['evaluation', 'psychometrics', 'cognitive-modeling']
venue: "NAACL 2024"
tldr: "Evaluates the psychometric predictive power of LLMs, finding instruction tuning does not always make next-word probabilities more human-like."
---

# Psychometric Predictive Power of Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.129/](https://aclanthology.org/2024.findings-naacl.129/)

**TLDR**: Evaluates the psychometric predictive power of LLMs, finding instruction tuning does not always make next-word probabilities more human-like.

## Abstract

AbstractInstruction tuning aligns the response of large language models (LLMs) with human preferences.Despite such efforts in human–LLM alignment, we find that instruction tuning does not always make LLMs human-like from a cognitive modeling perspective. More specifically, next-word probabilities estimated by instruction-tuned LLMs are often worse at simulating human reading behavior than those estimated by base LLMs.In addition, we explore prompting methodologies for simulating human reading behavior with LLMs. Our results show that prompts reflecting a particular linguistic hypothesis improve psychometric predictive power, but are still inferior to small base models.These findings highlight that recent advancements in LLMs, i.e., instruction tuning and prompting, do not offer better estimates than direct probability measurements from base LLMs in cognitive modeling. In other words, pure next-word probability remains a strong predictor for human reading behavior, even in the age of LLMs.