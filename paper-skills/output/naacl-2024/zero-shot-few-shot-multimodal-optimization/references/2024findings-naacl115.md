---
title: "More Samples or More Prompts? Exploring Effective Few-Shot In-Context Learning for LLMs with In-Context Sampling"
source: "https://aclanthology.org/2024.findings-naacl.115/"
categories: ['bpe-subword-parsing-evaluation', 'zero-shot-few-shot-multimodal-optimization']
tags: ['in-context-learning', 'few-shot', 'prompting']
venue: "NAACL 2024"
tldr: "Explores using multiple prompts with in-context sampling to improve few-shot learning for LLMs."
---

# More Samples or More Prompts? Exploring Effective Few-Shot In-Context Learning for LLMs with In-Context Sampling

**Source**: [https://aclanthology.org/2024.findings-naacl.115/](https://aclanthology.org/2024.findings-naacl.115/)

**TLDR**: Explores using multiple prompts with in-context sampling to improve few-shot learning for LLMs.

## Abstract

AbstractWhile most existing works on LLM prompting techniques focus only on how to select a better set of data samples inside one single prompt input (In-Context Learning or ICL), why can not we design and leverage multiple prompts together to further improve the LLM’s performance? In this work, we propose In-Context Sampling (ICS), a low-resource LLM prompting technique to produce confident predictions by optimizing the construction of multiple ICL prompt inputs. Extensive experiments with three open-source LLMs (FlanT5-XL, Mistral-7B, and Mixtral-8x7B) on four NLI datasets (e-SNLI, Multi-NLI, ANLI, and Contract-NLI) and one QA dataset (CommonsenseQA) illustrate that ICS can consistently enhance LLMs’ performance. An in-depth evaluation with three data similarity-based ICS strategies suggests that these strategies can further elevate LLM’s performance, which sheds light on a new yet promising future research direction.