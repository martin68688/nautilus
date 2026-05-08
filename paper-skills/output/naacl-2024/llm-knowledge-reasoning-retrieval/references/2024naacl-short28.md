---
title: "Contextualizing Argument Quality Assessment with Relevant Knowledge"
source: "https://aclanthology.org/2024.naacl-short.28/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization']
tags: ['argument-quality', 'contextualization', 'knowledge-retrieval', 'misinformation']
venue: "NAACL 2024"
tldr: "Proposes contextualizing argument quality assessment by retrieving and incorporating relevant knowledge, moving beyond analyzing arguments in isolation."
---

# Contextualizing Argument Quality Assessment with Relevant Knowledge

**Source**: [https://aclanthology.org/2024.naacl-short.28/](https://aclanthology.org/2024.naacl-short.28/)

**TLDR**: Proposes contextualizing argument quality assessment by retrieving and incorporating relevant knowledge, moving beyond analyzing arguments in isolation.

## Abstract

AbstractAutomatic assessment of the quality of arguments has been recognized as a challenging task with significant implications for misinformation and targeted speech. While real-world arguments are tightly anchored in context, existing computational methods analyze their quality in isolation, which affects their accuracy and generalizability. We propose SPARK: a novel method for scoring argument quality based on contextualization via relevant knowledge. We devise four augmentations that leverage large language models to provide feedback, infer hidden assumptions, supply a similar-quality argument, or give a counter-argument. SPARK uses a dual-encoder Transformer architecture to enable the original argument and its augmentation to be considered jointly. Our experiments in both in-domain and zero-shot setups show that SPARK consistently outperforms existing techniques across multiple metrics