---
title: "Self-Prompting Large Language Models for Zero-Shot Open-Domain QA"
source: "https://aclanthology.org/2024.naacl-long.17/"
categories: ['llm-knowledge-reasoning-retrieval', 'zero-shot-few-shot-multimodal-optimization', 'large-language-model-evaluation-augmentation']
tags: ['zero-shot', 'open-domain-qa', 'self-prompting']
venue: "NAACL 2024"
tldr: "A self-prompting method enables large language models to perform zero-shot open-domain question answering without training data."
---

# Self-Prompting Large Language Models for Zero-Shot Open-Domain QA

**Source**: [https://aclanthology.org/2024.naacl-long.17/](https://aclanthology.org/2024.naacl-long.17/)

**TLDR**: A self-prompting method enables large language models to perform zero-shot open-domain question answering without training data.

## Abstract

AbstractOpen-Domain Question Answering (ODQA) aims to answer questions without explicitly providing specific background documents. This task becomes notably challenging in a zero-shot setting where no data is available to train tailored retrieval-reader models.While recent Large Language Models (LLMs) like GPT-3 have demonstrated their effectiveness in zero-shot ODQA using direct prompting methods, these methods still fall short of fully harnessing the potential of LLMs when implicitly invoked.In this paper, we propose a Self-Prompting framework to explicitly utilize the massive knowledge encoded in the parameters of LLMs and their strong instruction understanding abilities. Concretely, we prompt LLMs step by step to generate multiple pseudo QA pairs with background passages and explanations entirely from scratch.These generated elements are then utilized for in-context learning. Experimental results show that our method significantly surpasses previous state-of-the-art zero-shot methods on three widely-used ODQA datasets and even achieves comparable performance with various customized fine-tuned models on full training data. Our code is available at https://github.com/lockon-n/self-prompting.