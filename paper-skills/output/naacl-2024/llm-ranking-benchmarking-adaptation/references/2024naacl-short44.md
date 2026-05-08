---
title: "Is Prompt Transfer Always Effective? An Empirical Study of Prompt Transfer for Question Answering"
source: "https://aclanthology.org/2024.naacl-short.44/"
categories: ['zero-shot-few-shot-multimodal-optimization', 'llm-ranking-benchmarking-adaptation']
tags: ['prompt-tuning', 'transfer-learning', 'question-answering', 'evaluation']
venue: "NAACL 2024"
tldr: "Empirically studies the effectiveness of transferring trained soft prompts for question answering across datasets and model sizes."
---

# Is Prompt Transfer Always Effective? An Empirical Study of Prompt Transfer for Question Answering

**Source**: [https://aclanthology.org/2024.naacl-short.44/](https://aclanthology.org/2024.naacl-short.44/)

**TLDR**: Empirically studies the effectiveness of transferring trained soft prompts for question answering across datasets and model sizes.

## Abstract

AbstractPrompt tuning, which freezes all parameters of a pre-trained model and only trains a soft prompt, has emerged as a parameter-efficient approach. For the reason that the prompt initialization becomes sensitive when the model size is small, the prompt transfer that uses the trained prompt as an initialization for the target task has recently been introduced. Since previous works have compared tasks in large categories (e.g., summarization, sentiment analysis), the factors that influence prompt transfer have not been sufficiently explored. In this paper, we characterize the question answering task based on features such as answer format and empirically investigate the transferability of soft prompts for the first time. We analyze the impact of initialization during prompt transfer and find that the train dataset size of source and target tasks have the influence significantly. Furthermore, we propose a novel approach for measuring catastrophic forgetting and investigate how it occurs in terms of the amount of evidence. Our findings can help deeply understand transfer learning in prompt tuning.