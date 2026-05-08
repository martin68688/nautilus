---
title: "The ART of LLM Refinement: Ask, Refine, and Trust"
source: "https://aclanthology.org/2024.naacl-long.327/"
categories: ['large-language-model-evaluation-augmentation']
tags: ['self-refinement', 'self-improvement', 'evaluation', 'trust']
venue: "NAACL 2024"
tldr: "Proposes an iterative self-refinement framework (Ask, Refine, Trust) for LLMs to judge and improve their own outputs."
---

# The ART of LLM Refinement: Ask, Refine, and Trust

**Source**: [https://aclanthology.org/2024.naacl-long.327/](https://aclanthology.org/2024.naacl-long.327/)

**TLDR**: Proposes an iterative self-refinement framework (Ask, Refine, Trust) for LLMs to judge and improve their own outputs.

## Abstract

AbstractLarge Language Models (LLMs) have demonstrated remarkable generative abilities, but can they judge the quality of their own generations and self-improve?A popular concept, referred to as *self-refinement*, postulates that LLMs can detect and correct the errors in their generations when asked to do so. However, recent empirical evidence points in the opposite direction, suggesting that LLMs often struggle to accurately identify errors when reasoning is involved. To address this, we propose a reasoning with a refinement strategy called *ART: Ask, Refine, and Trust*, which *asks* necessary questions to decide when an LLM should *refine* its output, and uses it to affirm or deny *trust* in its refinement by ranking the refinement and the initial prediction. On two multistep reasoning tasks of mathematical word problems (GSM8K) and question answering (StrategyQA), *ART* achieves a performance gain of +5 points over self-refinement baselines, while using a much smaller model as the decision maker. We believe that *ART* with smaller models, making refinement decisions can be a cost-effective alternative to fine-tuning LLMs.