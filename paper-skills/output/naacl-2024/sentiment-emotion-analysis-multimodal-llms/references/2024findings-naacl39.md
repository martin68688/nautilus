---
title: "LLM-Rec: Personalized Recommendation via Prompting Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.39/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'personalized-marketing-copy-generation']
tags: ['recommendation', 'LLM', 'prompting', 'personalization']
venue: "NAACL 2024"
tldr: "LLM-Rec performs personalized text-based recommendation by prompting LLMs with enriched item descriptions."
---

# LLM-Rec: Personalized Recommendation via Prompting Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.39/](https://aclanthology.org/2024.findings-naacl.39/)

**TLDR**: LLM-Rec performs personalized text-based recommendation by prompting LLMs with enriched item descriptions.

## Abstract

AbstractText-based recommendation holds a wide range of practical applications due to its versatility, as textual descriptions can represent nearly any type of item. However, directly employing the original item descriptions may not yield optimal recommendation performance due to the lack of comprehensive information to align with user preferences. Recent advances in large language models (LLMs) have showcased their remarkable ability to harness commonsense knowledge and reasoning. In this study, we introduce a novel approach, coined LLM-Rec, which incorporates four distinct prompting strategies of text enrichment for improving personalized text-based recommendations. Our empirical experiments reveal that using LLM-augmented text significantly enhances recommendation quality. Even basic MLP (Multi-Layer Perceptron) models achieve comparable or even better results than complex content-based methods. Notably, the success of LLM-Rec lies in its prompting strategies, which effectively tap into the language model’s comprehension of both general and specific item characteristics. This highlights the importance of employing diverse prompts and input augmentation techniques to boost the recommendation effectiveness of LLMs.