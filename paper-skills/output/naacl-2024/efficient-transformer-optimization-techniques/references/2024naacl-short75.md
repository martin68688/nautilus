---
title: "Breaking the Language Barrier: Can Direct Inference Outperform Pre-Translation in Multilingual LLM Applications?"
source: "https://aclanthology.org/2024.naacl-short.75/"
categories: ['efficient-large-model-training-optimization', 'efficient-transformer-optimization-techniques']
tags: ['efficient-inference', 'parallel-decoding', 'decoding-optimization']
venue: "NAACL 2024"
tldr: "Proposes Adaptive N-gram Parallel Decoding, a lossless method to accelerate LLM inference by leveraging n-gram predictions."
---

# Breaking the Language Barrier: Can Direct Inference Outperform Pre-Translation in Multilingual LLM Applications?

**Source**: [https://aclanthology.org/2024.naacl-short.75/](https://aclanthology.org/2024.naacl-short.75/)

**TLDR**: Proposes Adaptive N-gram Parallel Decoding, a lossless method to accelerate LLM inference by leveraging n-gram predictions.

## Abstract

AbstractLarge language models hold significant promise in multilingual applications. However, inherent biases stemming from predominantly English-centric pre-training have led to the widespread practice of pre-translation, i.e., translating non-English inputs to English before inference, leading to complexity and information loss. This study re-evaluates the need for pre-translation in the context of PaLM2 models, which have been established as highly performant in multilingual tasks. We offer a comprehensive investigation across 108 languages and 6 diverse benchmarks, including open-end generative tasks, which were excluded from previous similar studies. Our findings challenge the pre-translation paradigm established in prior research, highlighting the advantages of direct inference in PaLM2. Specifically, PaLM2-L consistently outperforms pre-translation in 94 out of 108 languages. These findings pave the way for more efficient and effective multilingual applications, alleviating the limitations associated with pre-translation and unlocking linguistic authenticity.