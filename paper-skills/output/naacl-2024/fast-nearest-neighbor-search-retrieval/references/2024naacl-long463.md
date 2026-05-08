---
title: "REPLUG: Retrieval-Augmented Black-Box Language Models"
source: "https://aclanthology.org/2024.naacl-long.463/"
categories: ['llm-reasoning-retrieval-evaluation', 'fast-nearest-neighbor-search-retrieval']
tags: ['retrieval-augmented', 'language-modeling', 'black-box', 'retrieval-tuning']
venue: "NAACL 2024"
tldr: "REPLUG is a retrieval-augmented language modeling framework that treats the LM as a black box and augments it with a tunable retrieval model, improving performance by simply prepending retrieved documents to input."
---

# REPLUG: Retrieval-Augmented Black-Box Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.463/](https://aclanthology.org/2024.naacl-long.463/)

**TLDR**: REPLUG is a retrieval-augmented language modeling framework that treats the LM as a black box and augments it with a tunable retrieval model, improving performance by simply prepending retrieved documents to input.

## Abstract

AbstractWe introduce REPLUG, a retrieval-augmented language modeling framework that treats the language model (LM) as a black box and augments it with a tuneable retrieval model. Unlike prior retrieval-augmented LMs that train language models with special cross-attention mechanisms to encode the retrieved text, REPLUG simply prepends retrieved documents to the input for the frozen black-box LM. This simple design can be easily applied to any existing language models. Furthermore, we show that the LM can be used to supervise the retrieval model, which can then find documents that help the LM make better predictions. Our experiments demonstrate that REPLUG with the tuned retriever significantly improves the performance of GPT-3 (175B) on language modeling by 6.3%, as well as the performance of Codex on five-shot MMLU by 5.1%. Code is publicly released at github.com/swj0419/REPLUG.