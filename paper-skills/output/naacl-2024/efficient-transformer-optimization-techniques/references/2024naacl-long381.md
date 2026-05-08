---
title: "Transformers Can Represent n-gram Language Models"
source: "https://aclanthology.org/2024.naacl-long.381/"
categories: ['efficient-transformer-optimization-techniques', 'llm-evaluation-summarization-argument-extraction']
tags: ['representational-capacity', 'ngram', 'transformers']
venue: "NAACL 2024"
tldr: "Shows that transformer models have the representational capacity to implement n-gram language models, analyzing their ability for language modeling."
---

# Transformers Can Represent n-gram Language Models

**Source**: [https://aclanthology.org/2024.naacl-long.381/](https://aclanthology.org/2024.naacl-long.381/)

**TLDR**: Shows that transformer models have the representational capacity to implement n-gram language models, analyzing their ability for language modeling.

## Abstract

AbstractPlenty of existing work has analyzed the abilities of the transformer architecture by describing its representational capacity with formal models of computation. However, the focus so far has been on analyzing the architecture in terms of language acceptance. We contend that this is an ill-suited problem in the study of language models (LMs), which are definitionally probability distributions over strings. In this paper, we focus on the relationship between transformer LMs and n-gram LMs, a simple and historically relevant class of language models. We show that transformer LMs using the hard or sparse attention mechanisms can exactly represent any n-gram LM, giving us a concrete lower bound on their probabilistic representational capacity. This provides a first step towards understanding the mechanisms that transformer LMs can use to represent probability distributions over strings.