---
title: "On-the-Fly Fusion of Large Language Models and Machine Translation"
source: "https://aclanthology.org/2024.findings-naacl.35/"
categories: ['bpe-subword-parsing-evaluation', 'human-llm-opinion-dynamics-moderation']
tags: ['machine-translation', 'llm-ensembling', 'on-the-fly-fusion']
venue: "NAACL 2024"
tldr: "Improves machine translation by on-the-fly ensembling of an NMT model with an LLM prompted on the same task."
---

# On-the-Fly Fusion of Large Language Models and Machine Translation

**Source**: [https://aclanthology.org/2024.findings-naacl.35/](https://aclanthology.org/2024.findings-naacl.35/)

**TLDR**: Improves machine translation by on-the-fly ensembling of an NMT model with an LLM prompted on the same task.

## Abstract

AbstractWe propose on-the-fly ensembling of a neural machine translation (NMT) model with a large language model (LLM), prompted on the same task and input. Through experiments on 4 language directions with varying data amounts, we find that a slightly weaker-at-translation LLM can improve translations of a NMT model, and such an ensemble can produce better translations than ensembling two stronger NMT models.We demonstrate that our ensemble method can be combined with various techniques from LLM prompting, such as in context learning and translation context.