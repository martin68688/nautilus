---
title: "Tokenization Matters: Navigating Data-Scarce Tokenization for Gender Inclusive Language Technologies"
source: "https://aclanthology.org/2024.findings-naacl.113/"
categories: ['sentiment-emotion-analysis-multimodal-llms', 'social-bias-mitigation-in-language-models']
tags: ['tokenization', 'gender-inclusive', 'neopronouns']
venue: "NAACL 2024"
tldr: "Investigates how tokenization impacts LLM performance on gender-diverse neopronouns and proposes mitigation strategies."
---

# Tokenization Matters: Navigating Data-Scarce Tokenization for Gender Inclusive Language Technologies

**Source**: [https://aclanthology.org/2024.findings-naacl.113/](https://aclanthology.org/2024.findings-naacl.113/)

**TLDR**: Investigates how tokenization impacts LLM performance on gender-diverse neopronouns and proposes mitigation strategies.

## Abstract

AbstractGender-inclusive NLP research has documented the harmful limitations of gender binary-centric large language models (LLM), such as the inability to correctly use gender-diverse English neopronouns (e.g., xe, zir, fae). While data scarcity is a known culprit, the precise mechanisms through which scarcity affects this behavior remain underexplored. We discover LLM misgendering is significantly influenced by Byte-Pair Encoding (BPE) tokenization, the tokenizer powering many popular LLMs. Unlike binary pronouns, BPE overfragments neopronouns, a direct consequence of data scarcity during tokenizer training. This disparate tokenization mirrors tokenizer limitations observed in multilingual and low-resource NLP, unlocking new misgendering mitigation strategies. We propose two techniques: (1) pronoun tokenization parity, a method to enforce consistent tokenization across gendered pronouns, and (2) utilizing pre-existing LLM pronoun knowledge to improve neopronoun proficiency. Our proposed methods outperform finetuning with standard BPE, improving neopronoun accuracy from 14.1% to 58.4%. Our paper is the first to link LLM misgendering to tokenization and deficient neopronoun grammar, indicating that LLMs unable to correctly treat neopronouns as pronouns are more prone to misgender.