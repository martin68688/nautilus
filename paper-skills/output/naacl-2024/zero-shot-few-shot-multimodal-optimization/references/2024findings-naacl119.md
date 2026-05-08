---
title: "Prompt Space Optimizing Few-shot Reasoning Success with Large Language Models"
source: "https://aclanthology.org/2024.findings-naacl.119/"
categories: ['code-search-clone-detection', 'zero-shot-few-shot-multimodal-optimization']
tags: ['prompt-engineering', 'few-shot', 'reasoning', 'optimization']
venue: "NAACL 2024"
tldr: "Proposes optimizing the prompt space to improve the success rate of few-shot reasoning with large language models."
---

# Prompt Space Optimizing Few-shot Reasoning Success with Large Language Models

**Source**: [https://aclanthology.org/2024.findings-naacl.119/](https://aclanthology.org/2024.findings-naacl.119/)

**TLDR**: Proposes optimizing the prompt space to improve the success rate of few-shot reasoning with large language models.

## Abstract

AbstractPrompt engineering is an essential technique for enhancing the abilities of large language models (LLMs) by providing explicit and specific instructions. It enables LLMs to excel in various tasks, such as arithmetic reasoning, question answering, summarization, relation extraction, machine translation, and sentiment analysis. Researchers have been actively exploring different prompt engineering strategies, such as Chain of Thought (CoT), Zero-CoT, and In-context learning. However, an unresolved problem arises from the fact that current approaches lack a solid mathematical solution for determining optimal prompts. To address this issue in prompt engineering, we propose a new and effective approach called Prompt Space. Our methodology utilizes text embeddings to obtain basis vectors by matrix decomposition, and then constructs a space for representing all prompts. Prompt Space significantly outperforms state-of-the-art prompt paradigms on ten public reasoning benchmarks. Notably, without the help of the CoT method and the prompt “Let’s think step by step”, Prompt Space shows superior performance over the few-shot method. Overall, our approach provides a robust and effective mathematical framework for selecting simple and effective prompts. This advancement marks a significant step towards improving prompt engineering for a wide variety of applications in LLMs. Our code is publicly available at https://github.com/YouBLEI/Prompt-Space