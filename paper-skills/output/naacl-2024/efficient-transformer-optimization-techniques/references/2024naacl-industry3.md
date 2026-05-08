---
title: "SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling"
source: "https://aclanthology.org/2024.naacl-industry.3/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['model-scaling', 'depth-upscaling', 'llm']
venue: "NAACL 2024"
tldr: "Introduces a 10.7B parameter LLM scaled using a simple yet effective depth up-scaling method, achieving strong performance on various NLP tasks."
---

# SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling

**Source**: [https://aclanthology.org/2024.naacl-industry.3/](https://aclanthology.org/2024.naacl-industry.3/)

**TLDR**: Introduces a 10.7B parameter LLM scaled using a simple yet effective depth up-scaling method, achieving strong performance on various NLP tasks.

## Abstract

AbstractWe introduce SOLAR 10.7B, a large language model (LLM) with 10.7 billion parameters, demonstrating superior performance in various natural language processing (NLP) tasks. Inspired by recent efforts to efficiently up-scale LLMs, we present a method for scaling LLMs called depth up-scaling (DUS), which encompasses depthwise scaling and continued pretraining. In contrast to other LLM up-scaling methods that use mixture-of-experts, DUS does not require complex changes to train and inference efficiently. We show experimentally that DUS is simple yet effective in scaling up high-performance LLMs from small ones. Building on the DUS model, we additionally present SOLAR 10.7B-Instruct, a variant fine-tuned for instruction-following capabilities, surpassing Mixtral-8x7B-Instruct. SOLAR 10.7B is publicly available under the Apache 2.0 license, promoting broad access and application in the LLM field.