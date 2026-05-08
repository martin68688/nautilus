---
title: "REST: Retrieval-Based Speculative Decoding"
source: "https://aclanthology.org/2024.naacl-long.88/"
categories: ['human-llm-opinion-dynamics-moderation', 'efficient-transformer-optimization-techniques']
tags: ['speculative-decoding', 'retrieval']
venue: "NAACL 2024"
tldr: "REST speeds up LLM generation using retrieval-based speculative decoding."
---

# REST: Retrieval-Based Speculative Decoding

**Source**: [https://aclanthology.org/2024.naacl-long.88/](https://aclanthology.org/2024.naacl-long.88/)

**TLDR**: REST speeds up LLM generation using retrieval-based speculative decoding.

## Abstract

AbstractWe introduce Retrieval-Based Speculative Decoding (REST), a novel algorithm designed to speed up language model generation. The key insight driving the development of REST is the observation that the process of text generation often includes certain common phases and patterns. Unlike previous methods that rely on a draft language model for speculative decoding, REST harnesses the power of retrieval to generate draft tokens. This method draws from the reservoir of existing knowledge, retrieving and employing relevant tokens based on the current context. Its plug-and-play nature allows for seamless integration and acceleration of any language model, all without necessitating additional training. When benchmarked on 7B and 13B language models in a single-batch setting, REST achieves a significant speedup of 1.62 × to 2.36 × on code or text generation. The source code of REST is available at https://github.com/FasterDecoding/REST.