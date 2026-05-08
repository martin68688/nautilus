---
title: "Revisiting subword tokenization: A case study on affixal negation in large language models"
source: "https://aclanthology.org/2024.naacl-long.284/"
categories: ['bpe-subword-parsing-evaluation', 'llm-backdoor-attacks-defense']
tags: ['tokenization', 'subword', 'morphology', 'negation', 'llm-evaluation']
venue: "NAACL 2024"
tldr: "Studies the impact of affixal negation on English LLMs, highlighting challenges posed by morphologically-implausible tokenizers."
---

# Revisiting subword tokenization: A case study on affixal negation in large language models

**Source**: [https://aclanthology.org/2024.naacl-long.284/](https://aclanthology.org/2024.naacl-long.284/)

**TLDR**: Studies the impact of affixal negation on English LLMs, highlighting challenges posed by morphologically-implausible tokenizers.

## Abstract

AbstractIn this work, we measure the impact of affixal negation on modern English large language models (LLMs). In affixal negation, the negated meaning is expressed through a negative morpheme, which is potentially challenging for LLMs as their tokenizers are often not morphologically plausible. We conduct extensive experiments using LLMs with different subword tokenization methods, which lead to several insights on the interaction between tokenization performance and negation sensitivity. Despite some interesting mismatches between tokenization accuracy and negation detection performance, we show that models can, on the whole, reliably recognize the meaning of affixal negation.