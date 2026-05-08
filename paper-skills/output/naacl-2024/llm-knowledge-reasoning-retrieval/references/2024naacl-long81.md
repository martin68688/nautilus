---
title: "Large Language Models Help Humans Verify Truthfulness – Except When They Are Convincingly Wrong"
source: "https://aclanthology.org/2024.naacl-long.81/"
categories: ['llm-knowledge-reasoning-retrieval', 'large-language-model-evaluation-augmentation']
tags: ['fact-checking', 'truthfulness', 'human-ai-interaction']
venue: "NAACL 2024"
tldr: "Finds LLMs can help humans verify truthfulness but can also mislead them when they are convincingly wrong."
---

# Large Language Models Help Humans Verify Truthfulness – Except When They Are Convincingly Wrong

**Source**: [https://aclanthology.org/2024.naacl-long.81/](https://aclanthology.org/2024.naacl-long.81/)

**TLDR**: Finds LLMs can help humans verify truthfulness but can also mislead them when they are convincingly wrong.

## Abstract

AbstractLarge Language Models (LLMs) are increasingly used for accessing information on the web. Their truthfulness and factuality are thus of great interest. To help users make the right decisions about the information they get, LLMs should not only provide information but also help users fact-check it. We conduct human experiments with 80 crowdworkers to compare language models with search engines (information retrieval systems) at facilitating fact-checking. We prompt LLMs to validate a given claim and provide corresponding explanations. Users reading LLM explanations are significantly more efficient than those using search engines while achieving similar accuracy. However, they over-rely on the LLMs when the explanation is wrong. To reduce over-reliance on LLMs, we ask LLMs to provide contrastive information—explain both why the claim is true and false, and then we present both sides of the explanation to users. This contrastive explanation mitigates users’ over-reliance on LLMs, but cannot significantly outperform search engines. Further, showing both search engine results and LLM explanations offers no complementary benefits compared to search engines alone. Taken together, our study highlights that natural language explanations by LLMs may not be a reliable replacement for reading the retrieved passages, especially in high-stakes settings where over-relying on wrong AI explanations could lead to critical consequences.