---
title: "SELF-[IN]CORRECT: LLMs Struggle with Discriminating Self-Generated Responses"
source: "https://www.semanticscholar.org/paper/e6f26f8feb3cd87a06ed87d5cddedf22b7f6d2a3"
categories: ['reasoning-learning-optimization-in-llms', 'llm-safety-adversarial-robustness']
tags: ['llm-evaluation', 'self-correction', 'self-discrimination']
venue: "AAAI 2024"
tldr: "Finds that LLMs struggle to discriminate and select the best among their own generated responses for self-correction."
---

# SELF-[IN]CORRECT: LLMs Struggle with Discriminating Self-Generated Responses

**Source**: [https://www.semanticscholar.org/paper/e6f26f8feb3cd87a06ed87d5cddedf22b7f6d2a3](https://www.semanticscholar.org/paper/e6f26f8feb3cd87a06ed87d5cddedf22b7f6d2a3)

**TLDR**: Finds that LLMs struggle to discriminate and select the best among their own generated responses for self-correction.

## Abstract

Can LLMs consistently improve their previous outputs for better results? For this to be true, LLMs would need to be better at discriminating among previously-generated alternatives, than generating initial responses. We explore the validity of this hypothesis in practice. We first formulate a unified framework that allows us to compare the generative and discriminative capability of any model on any task. In our resulting experimental analysis of several open-source and industrial LLMs, we observe that model’s are not reliably better at discriminating among previously-generated alternatives than generating initial responses. This finding challenges the notion that LLMs may be able to enhance their performance only through their own judgment.